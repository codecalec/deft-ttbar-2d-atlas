import sys
import getopt
from pathlib import Path
from typing import List
import logging

from data import collect_MC_data, extract_data, cov_matrix
from config import generate_json

import numpy as np
import deft_hep as deft
import matplotlib.pyplot as plt
from matplotlib import rcParams, axes

plt.style.use("science")
rcParams["savefig.dpi"] = 200


DATA_PATH = Path("/home/agv/Documents/Honours/Project/data/1908.07305")
MC_PATH = Path("/home/agv/Documents/Honours/Project/data_generation/atlas_ttbar_2D")


def plot_comparison(
    ax: axes.Axes,
    data: np.ndarray,
    error: np.ndarray,
    mc: np.ndarray,
    ctg: float,
    bin_left: np.ndarray,
    bin_right: np.ndarray,
    label=None,
):

    centres = bin_left + (bin_right - bin_left) / 2
    ax.errorbar(centres, data, yerr=error, fmt=".r", label="ATLAS data")
    ax.errorbar(
        centres,
        mc,
        xerr=(bin_right - bin_left) / 2,
        fmt=".k",
        label="MadGraph [$C_{tg}" + f"={ctg}$]",
    )

    ax.set_xlabel(r"$p_t^{T}$ [GeV]")
    ax.set_ylabel(r"$d\sigma/d m_{ttbar}$")
    ax.legend()


def plot_comparison_multiple_operator(
    ax: axes.Axes,
    data: np.ndarray,
    error: np.ndarray,
    mc: List[np.ndarray],
    ctg: List[float],
    bin_left: np.ndarray,
    bin_right: np.ndarray,
):
    from matplotlib.colors import Normalize

    centres = bin_left + (bin_right - bin_left) / 2

    colours = plt.get_cmap("viridis")(Normalize()(ctg))
    for m, c, colour in zip(mc, ctg, colours):
        ax.errorbar(
            centres,
            m,
            xerr=(bin_right - bin_left) / 2,
            fmt=".",
            ecolor="k",
            # color=colour,
            label="$C_{tg}" + f"={c}$",
        )

    ax.errorbar(centres, data, yerr=error, fmt=".r", label="ATLAS data")

    ax.set_xlabel(r"$p_t^{T}$ [GeV]")
    ax.set_ylabel(r"$d\sigma/d m_{ttbar}$ [pb GeV$^{-2}$]")
    ax.legend()


def run_analysis(config_name: Path):
    config = deft.ConfigReader(config_name)
    pb = deft.PredictionBuilder(1, config.samples, config.predictions)
    fitter = deft.MCMCFitter(config, pb)
    sampler = fitter.sampler

    print(
        "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    )
    print(
        "Mean autocorrelation time: {0:.3f} steps".format(
            np.mean(sampler.get_autocorr_time())
        )
    )

    mcmc_params = np.mean(sampler.flatchain, axis=0)
    mcmc_params_cov = np.cov(np.transpose(sampler.flatchain))
    print("Fit Results")
    print(mcmc_params)
    print(mcmc_params_cov)


    predictions = pb.make_prediction(mcmc_params)
    data_values = config.params["config"]["data"]["central_values"]
    data_errors = np.sqrt(
        np.array(
            config.params["config"]["data"]["covariance_matrix"]
        ).diagonal()
    )
    x = range(15)
    plt.errorbar(x, data_values, yerr=data_errors, fmt=".k")
    plt.plot(x, predictions, "o")
    plt.show()


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


if __name__ == "__main__":

    logging.basicConfig(filename="analysis.log", encoding="utf-8", level=logging.DEBUG)
    mpl_logger = logging.getLogger("matplotlib.texmanager")
    mpl_logger.setLevel(logging.WARNING)

    data_files = sorted(DATA_PATH.glob("PTT_MTTBAR_?.csv"))
    data_values = np.array([])
    for f in data_files:
        data_values = np.append(data_values, extract_data(f))

    logging.debug(f"Data Values:\n{data_values}\n{data_values.shape}")

    cov_files = sorted(DATA_PATH.glob("PTT_MTTBAR_COV_*.csv"))
    values_cov = cov_matrix(cov_files)
    logging.debug(f"Covariance Matrix:\n{values_cov}\n{values_cov.shape}")

    mc_files = sorted(MC_PATH.glob("run_0?_LO/MADatNLO.HwU"))
    ctg_values = [-2.0, 0.0001, 2.0, -1.0, 1.0]
    bin_left, bin_right, mc_data = collect_MC_data(mc_files, ctg_values)

    fig = plt.figure()
    axs = fig.subplots(2, 2)
    indices = [(0, 3), (3, 7), (7, 12), (12, 15)]
    for ax, (left, right) in zip(axs.flatten(), indices):
        error = np.sqrt(np.diagonal(values_cov[left:right, left:right]))
        trimmed_mc_data = [m[left:right] for m in mc_data]
        plot_comparison_multiple_operator(
            ax,
            data_values[left:right],
            error,
            trimmed_mc_data,
            ctg_values,
            bin_left[left:right],
            bin_right[left:right],
        )
    plt.show()

    generate_json(data_values, values_cov, mc_data, ctg_values, Path("atlas_2D.json"))
    run_analysis(Path("atlas_2D.json"))

    # run_validation(Path("atlas_2D.json"), Path("test_atlas_2D.json"))
