import copy
import os
from pathlib import Path
from typing import Union

import deft_hep as deft
import matplotlib.pyplot as plt
import numpy as np
from fitting import find_minimum, min_func
from plot import data_plot  # , grid_plot


def ln_prob(
    c: np.ndarray,
    pb: deft.PredictionBuilder,
    data: np.ndarray,
    icov,
) -> float:
    pred = pb.make_prediction(c)
    diff = pred - data

    ll = -np.dot(diff, np.dot(icov, diff))
    return ll


def grid_minimise(config: deft.ConfigReader, pb: deft.PredictionBuilder):
    from functools import partial
    from itertools import product

    assert pb.nOps == 2

    coeff_labels = config.tex_labels
    coeff_bounds = list(config.prior_limits.values())
    data = config.data
    icov = np.linalg.inv(config.cov)
    v_neg_ln_prob = np.vectorize(
        partial(min_func, pb=pb, data=data, icov=icov),
        otypes=[float],
        signature="(n)->()",
    )

    n = 200
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    c = np.array([[i, j] for (i, j) in product(x, y)])

    ll = -v_neg_ln_prob(c)
    l = np.exp(ll)

    plt.contourf(x, y, ll.reshape((n, n)).T, levels=100)
    plt.xlabel(coeff_labels[0])
    plt.ylabel(coeff_labels[1])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"$\ln L$", rotation=270)
    plt.savefig("minimise_grid_ll.png")
    plt.clf()

    plt.contourf(x, y, l.reshape((n, n)).T, levels=100)
    plt.xlabel(coeff_labels[0])
    plt.ylabel(coeff_labels[1])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"L", rotation=270)
    plt.savefig("minimise_grid_l.png")
    plt.clf()


def run_analysis(config_name: Union[str, Path], closure: bool = False):
    # avoid numpy parallelization
    os.environ["OMP_NUM_THREADS"] = "1"

    config = deft.ConfigReader(config_name)

    pb = deft.PredictionBuilder(config)

    if closure:
        print("Closure Test")
        pred_zero = pb.make_prediction(np.array([0.0] * pb.nOps))
        config_closure = copy.deepcopy(config)
        config_closure.data = pred_zero
        print(config_closure.data)
        fitter_closure = deft.MCMCFitter(
            config_closure,
            pb,
            initial_pos=np.array([1] * pb.nOps),
            initial_deviation=1,
            use_multiprocessing=False,
        )
        sampler_closure = fitter_closure.sampler
        mcmc_params_closure = np.mean(
            sampler_closure.get_chain(flat=True), axis=0
        )
        mcmc_params_cov_closure = np.atleast_1d(
            np.cov(np.transpose(sampler_closure.get_chain(flat=True)))
        )
        mcmc_params_error_closure = (
            np.sqrt(np.diag(mcmc_params_cov_closure))
            if len(mcmc_params_cov_closure) != 1
            else np.sqrt(mcmc_params_cov_closure)
        )

        sp = deft.SummaryPlotter(config, pb, fitter_closure)
        sp.fit_result(ylabel="Diff. XSec", show_plot=False, log_scale=True)
        sp.corner(show_plot=False)

        print("Fit Results")
        print("Coefficient Values:", mcmc_params_closure)
        print("Coefficient cov:", mcmc_params_cov_closure)
        print("Coefficient Err:", mcmc_params_error_closure)
        exit()

    if pb.nOps == 2:
        grid_minimise(config, pb)

    find_minimum(config, pb, config.data, config.cov, initial_c=[0] * pb.nOps)

    fitter = deft.MCMCFitter(
        config,
        pb,
        initial_pos=np.array([0] * pb.nOps),
        initial_deviation=1,
        use_multiprocessing=False,
    )
    sampler = fitter.sampler

    print(
        "Mean acceptance fraction: {0:.3f}".format(
            np.mean(sampler.acceptance_fraction)
        )
    )
    print(
        "Mean autocorrelation time: {0:.3f} steps".format(
            np.mean(sampler.get_autocorr_time())
        )
    )

    mcmc_params = np.mean(sampler.get_chain(flat=True), axis=0)
    mcmc_params_cov = np.atleast_1d(
        np.cov(np.transpose(sampler.get_chain(flat=True)))
    )
    mcmc_params_error = (
        np.sqrt(np.diag(mcmc_params_cov))
        if len(mcmc_params_cov) != 1
        else np.sqrt(mcmc_params_cov)
    )
    print("Fit Results")
    print("Coefficient Values:", mcmc_params)
    print("Coefficient cov:", mcmc_params_cov)
    print("Coefficient Err:", mcmc_params_error)

    icov = np.linalg.inv(config.cov)

    predictions = pb.make_prediction(mcmc_params)
    pred_zero_loc = config.samples.tolist().index(
        [1.0] + [0.0] * (len(mcmc_params))
    )
    pred_zero = config.predictions[pred_zero_loc]
    # pb.make_prediction([0.0 for _ in range(len(mcmc_params))])

    data_values = config.data
    data_var = np.array(config.cov).diagonal()
    diff = predictions - config.data
    ll = -np.dot(diff, np.dot(icov, diff))

    chi_sq = np.sum((data_values - predictions) ** 2 / data_var)
    chi_sq_dof = chi_sq / (len(data_values) - len(mcmc_params))
    print("MCMC Results")
    print("Prediction:", predictions)
    print("Chi squared:", chi_sq)
    print("Chi squared dof:", chi_sq_dof)
    print("LL:", ll)

    diff = pred_zero - config.data
    ll = -np.dot(diff, np.dot(icov, diff))
    chi_sq = np.sum((data_values - pred_zero) ** 2 / data_var)
    chi_sq_dof = chi_sq / len(data_values)
    print("MCMC with zero prediction Results")
    print("Chi squared Pred zero:", chi_sq)
    print("Chi squared dof:", chi_sq_dof)
    print("LL:", ll)

    sp = deft.SummaryPlotter(config, pb, fitter)
    sp.fit_result(ylabel="Diff. XSec", show_plot=False, log_scale=True)
    sp.corner(show_plot=False)

    model_label = "Model Pred.\n" + "\n".join(
        [
            "{}={:.2f}".format(l, p)
            for l, p in zip(config.tex_labels, mcmc_params)
        ]
    )
    config.tex_labels

    data_plot(
        config.data,
        np.array(config.cov),
        [
            (predictions, model_label),
            (pred_zero, f"SM Pred."),
        ],
        filename="ATLAS_model_result.png",
        ratio=True,
    )


def run_validation(config_name, test_name):

    config = deft.ConfigReader(config_name)
    pb = deft.PredictionBuilder(config)
    mv = deft.ModelValidator(pb)

    config_test = deft.ConfigReader(test_name)

    pred_samples, predictions = mv.validation_predictions(config_test)

    score = np.empty(len(predictions))
    res_avg = np.empty(len(predictions))
    for i, (model_sample, model_pred, true_sample, true_pred) in enumerate(
        zip(
            pred_samples,
            predictions,
            config_test.samples,
            config_test.predictions,
        )
    ):
        assert (model_sample == true_sample).all()
        err = true_pred * 0.01
        cov = np.diag(err ** 2)
        residuals = model_pred - true_pred
        rel_residuals = residuals / true_pred
        agreement = np.abs(residuals) < err

        # data_plot(
        # true_pred,
        # cov,
        # other_hists=[(model_pred, f"model {model_sample}")],
        # filename=f"./results/validation_{config.run_name}_{i}.png",
        # ratio=True,
        # data_label=f"MG5 {true_sample}",
        # )

        # x = range(len(model_pred))
        # plt.scatter(x, model_pred, marker=".")
        # plt.errorbar(x, true_pred, yerr=err, ls="None")
        # plt.title("{}".format(true_sample))
        # plt.show()

        # plt.errorbar(x, [1] * len(model_pred), yerr=(err / 2) / true_pred, ls="None")
        # plt.scatter(x, model_pred / true_pred, marker=".")
        # plt.show()

        res_avg[i] = rel_residuals.mean()
        score[i] = agreement.sum()
        print(
            "Sample:",
            model_sample,
            "Agree Score:",
            agreement.sum(),
            "Failed bins:",
            np.where(agreement == False)[0],
        )
    print(
        "Percentage:",
        score.mean() / len(predictions[0]),
        "+-",
        score.std() / len(predictions[0]),
    )

    from scipy.optimize import curve_fit

    gaus_func = (
        lambda x, A, mean, std: A
        * (1 / (std * (np.sqrt(2 * np.pi))))
        * (np.exp((-1.0 / 2.0) * (((x - mean) / std) ** 2)))
    )
    hist, edges = np.histogram(res_avg, bins=11, range=(-0.06, 0.06))
    centres = (edges[1:] + edges[:-1]) / 2
    p0 = [1, 0, 0.1]
    popt, pcov = curve_fit(f=gaus_func, xdata=centres, ydata=hist, p0=p0)
    print(popt, pcov)
    print(np.sqrt(np.diag(pcov)))

    fig = plt.figure(figsize=(3, 4))
    ax = fig.gca()
    x = np.linspace(-0.06, 0.06, 100)
    ax.hist(res_avg, bins=11, range=(-0.06, 0.06))
    ax.plot(x, gaus_func(x,*popt), "--k")
    ax.set_xlabel("Avg. Residuals")
    ax.set_ylabel("Num. Valid. tests")
    ax.text(-0.05, 25, "Mean={:.4f}\nStd={:.4f}".format(popt[1], popt[2]))
    plt.savefig("residuals_hist.png")
    plt.clf()



    # config_test.n_total = 10000
    # mv.validate(config_test)
