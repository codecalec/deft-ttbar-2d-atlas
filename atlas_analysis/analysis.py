from pathlib import Path
import os

import numpy as np
import deft_hep as deft
import matplotlib.pyplot as plt

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
    # cov = np.diag(np.diagonal(config.cov))
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

    plt.contourf(x, y, ll.reshape((n,n)).T, levels=100)
    plt.xlabel(coeff_labels[0])
    plt.ylabel(coeff_labels[1])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"$\ln L$", rotation=270)
    plt.savefig("minimise_grid_ll.png")
    plt.clf()

    plt.contourf(x, y, l.reshape((n,n)).T, levels=100)
    plt.xlabel(coeff_labels[0])
    plt.ylabel(coeff_labels[1])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"L", rotation=270)
    plt.savefig("minimise_grid_l.png")
    plt.clf()



def run_analysis(config_name: str or Path):
    # avoid numpy parallelization
    os.environ["OMP_NUM_THREADS"] = "1"

    config = deft.ConfigReader(config_name)

    pb = deft.PredictionBuilder(config)

    grid_minimise(config, pb)
    find_minimum(
        config, pb, config.data, config.cov, initial_c=[0.0] * pb.nOps
    )

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
    pred_zero = pb.make_prediction([0.0 for _ in range(len(mcmc_params))])

    data_values = config.data
    data_errors = np.sqrt(np.array(config.cov).diagonal())
    diff = predictions - config.data
    ll = -np.dot(diff, np.dot(icov, diff))

    chi_sq = np.sum((data_values - predictions) ** 2 / data_errors)
    chi_sq_dof = chi_sq / (len(data_values) - len(mcmc_params))
    print("MCMC Results")
    print("Prediction:", predictions)
    print("Chi squared:", chi_sq)
    print("Chi squared dof:", chi_sq_dof)
    print("LL:", ll)

    diff = pred_zero - config.data
    ll = -np.dot(diff, np.dot(icov, diff))
    chi_sq = np.sum((data_values - pred_zero) ** 2 / data_errors)
    chi_sq_dof = chi_sq / (len(data_values) - len(mcmc_params))
    print("MCMC with zero prediction Results")
    print("Chi squared Pred zero:", chi_sq)
    print("Chi squared dof:", chi_sq_dof)
    print("LL:", ll)

    sp = deft.SummaryPlotter(config, pb, fitter)
    sp.fit_result(ylabel="Diff. XSec", show_plot=False, log_scale=True)
    sp.corner(show_plot=False)

    data_plot(
        config.data,
        np.array(config.cov),
        [
            (predictions, f"Model Pred.\n{mcmc_params}"),
            (pred_zero, f"Zero Pred."),
        ],
        filename="ATLAS_model_result.png",
        ratio=True,
    )


def run_validation(config_name, test_name):
    from scipy.optimize import minimize

    config = deft.ConfigReader(config_name)
    pb = deft.PredictionBuilder(config)
    mv = deft.ModelValidator(pb)

    config_test = deft.ConfigReader(test_name)
    samples, model_preds = mv.validate(config_test)

    bounds = list(config_test.prior_limits.values())

    for sample, pred in zip(samples, model_preds):
        cov = config_test.cov  # np.diag(np.sqrt(pred))
        icov = np.linalg.inv(cov)  # use poisson erros
        initial_c = (
            sample[1:] + (np.random.random(len(sample[1:])) - 0.5) * 1.2
        )
        result = minimize(
            min_func,
            initial_c,
            args=(pb, pred, icov),
            bounds=bounds,
        )
        c = result.x
        c_cov = result.hess_inv.todense()
        c_err = np.sqrt(np.diag(c_cov))
        print(c, c_cov, sample)
        print(c_err)
        print(
            "Agrees:",
            (c - c_err < sample[1:]).all() and (sample[1:] < c + c_err).all(),
        )

    # fig, axs = plt.subplots(len(predictions)
    # for model_pred, mc_pred in zip(predictions, config_test.predictions):
    # plt.plot(range(len(model_pred)), model_pred, ".b", label="Model")
    # plt.plot(range(len(mc_pred)), mc_pred, ".r", label="MC")
    # plt.legend()
    # plt.show()
