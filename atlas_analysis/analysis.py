from pathlib import Path
import os

import numpy as np
import deft_hep as deft
import matplotlib.pyplot as plt


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


def run_analysis(config_name: str or Path):
    # avoid numpy parallelization
    os.environ["OMP_NUM_THREADS"] = "1"

    config = deft.ConfigReader(config_name)
    pb = deft.PredictionBuilder(config)
    fitter = deft.MCMCFitter(
        config,
        pb,
        initial_pos=np.array([0, 0, 0.5]),
        initial_deviation=0.3,
        use_multiprocessing=False,
    )
    sampler = fitter.sampler

    print(
        "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    )
    print(
        "Mean autocorrelation time: {0:.3f} steps".format(
            np.mean(sampler.get_autocorr_time())
        )
    )

    mcmc_params = np.mean(sampler.get_chain(flat=True), axis=0)
    mcmc_params_cov = np.cov(np.transpose(sampler.get_chain(flat=True)))
    mcmc_params_error = np.sqrt(np.diag(mcmc_params_cov))
    print("Fit Results")
    print("Coefficient Values:", mcmc_params)
    print("Coefficient cov:", mcmc_params_cov)
    print("Coefficient Err:", mcmc_params_cov)

    # if (chain := sampler.get_chain(flat=True, thin=25).T).ndim == 1:
    # plt.hist(chain)
    # plt.xlabel("$C_{tG}$")
    # plt.show()
    # elif chain.ndim == 2:
    # for c in chain:
    # print(c)
    # plt.hist(c)
    # plt.show()

    # print(chain[0].shape, chain[1].shape)
    # hist = np.histogramdd(chain.T)
    # print(hist)

    # plt.hist2d(chain[0], chain[1])
    # plt.xlabel("$C_{tG}$")
    # plt.ylabel("$C_{t\\phi}$")
    # plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # icov = np.linalg.inv(config.cov)

    # ax1.set_xlabel("Iteration")
    # ax1.set_ylabel("$C_{tG}$")
    # ax2.set_xlabel("Iteration")
    # ax2.set_ylabel("Log Likelihood")

    # for i, chain in enumerate(sampler.get_chain(thin=5).T[0]):
    # x = np.arange(len(chain)) * 5
    # ax1.plot(x, chain, label=f"Chain {i}")

    # ll = [ln_prob([c], pb, config.data, icov) for c in chain]
    # ax2.plot(x, ll, label=f"Chain {i}")
    # plt.savefig("fit_likelihood.png")
    # plt.clf()

    predictions = pb.make_prediction(mcmc_params)
    pred_zero = pb.make_prediction([0.0 for _ in range(len(mcmc_params))])
    data_values = config.params["config"]["data"]["central_values"]
    data_errors = np.sqrt(
        np.array(config.params["config"]["data"]["covariance_matrix"]).diagonal()
    )

    chi_sq = np.sum((data_values - predictions)**2 / data_errors)
    chi_sq_dof = chi_sq / (len(data_values) - len(mcmc_params))
    print("Chi squared:", chi_sq)
    print("Chi squared dof:", chi_sq_dof)

    chi_sq = np.sum((data_values - pred_zero)**2 / data_errors)
    chi_sq_dof = chi_sq / (len(data_values) - len(mcmc_params))
    print("Chi squared:", chi_sq)
    print("Chi squared dof:", chi_sq_dof)

    x = range(len(data_values))
    plt.errorbar(x, data_values, yerr=data_errors, fmt=".k", label="ATLAS Data")
    plt.plot(
        x,
        predictions,
        "o",
        label="Fit"
        # label="Fit [$c_{tG}"
        # + f"{mcmc_params[0]:.2f} \\pm {mcmc_params_error:.2f}"
        # + "$]",
    )
    plt.plot(x, pred_zero, "x", label="Zero Model")
    plt.xlabel("Bin Index")
    plt.ylabel(r"$d^{2}\sigma/dm_{t\bar{t}}dp^{T}_{t_{H}}$")
    plt.legend()
    plt.yscale("log")
    plt.savefig("fit_result.png")
    plt.clf()

    sp = deft.SummaryPlotter(config, pb, fitter)
    sp.fit_result(ylabel="Diff. XSec", show_plot=False, log_scale=True)
    sp.corner(show_plot=False)


def run_validation(config_name, test_name):
    config = deft.ConfigReader(config_name)
    pb = deft.PredictionBuilder(1, config.samples, config.predictions)
    mv = deft.ModelValidator(pb)

    config_test = deft.ConfigReader(test_name)
    samples, predictions = mv.validate(config_test)
    print(samples, predictions)
    print(config_test.samples, config_test.predictions)

    fig, axs = plt.subplots(2)
    for ax, model_pred, mc_pred in zip(axs, predictions, config_test.predictions):
        ax.plot(range(len(model_pred)), model_pred, ".b", label="Model")
        ax.plot(range(len(mc_pred)), mc_pred, ".r", label="MC")
        ax.legend()
    plt.show()
