from pathlib import Path
import logging
from config import *
from data import extract_data, cov_matrix, collect_MC_data
from analysis import run_analysis
from plot import plot_comparison, plot_comparison_multiple_operator

import numpy as np
import deft_hep as deft
import matplotlib.pyplot as plt
from matplotlib import rcParams, axes

plt.style.use("science")
rcParams["savefig.dpi"] = 200


DATA_PATH = Path("/home/agv/Documents/Honours_Project/data/1908.07305")
MC_PATH_CTG = Path("/home/agv/Documents/Honours_Project/data_generation/atlas_ttbar_2D")
MC_PATH_MULTIPLE = Path(
    "/home/agv/Documents/Honours_Project/data_generation/atlas_ttbar_2D_multiple"
)
MC_PATH_THREE_OP = Path(
    "/home/agv/Documents/Honours_Project/data_generation/atlas_ttbar_2D_all"
)


def ctg_analysis(data_values, values_cov):
    mc_files = sorted(MC_PATH_CTG.glob("run_0?_LO/MADatNLO.HwU"))
    ctg_values = [-2.0, 0.0001, 2.0, -1.0, 1.0]
    bin_left, bin_right, mc_data = collect_MC_data(mc_files)

    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(2, 2)
    indices = [(0, 3), (3, 7), (7, 12), (12, 15)]
    mttbar_values = [(350, 500), (500, 700), (700, 1000), (1000, 2000)]
    mttbar_labels = ["$m_{ttbar}=" + f"{i}-{j}$GeV" for (i, j) in mttbar_values]
    for ax, (left, right), label in zip(axs.flatten(), indices, mttbar_labels):
        error = np.sqrt(np.diagonal(values_cov[left:right, left:right]))
        trimmed_mc_data = mc_data[1][left:right]  # ctg =0.001
        plot_comparison(
            ax,
            data_values[left:right],
            error,
            trimmed_mc_data,
            ctg_values[1],
            bin_left[left:right],
            bin_right[left:right],
            label=label,
        )
    plt.savefig("ATLAS_MC_comparison.png")
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(2, 2)
    indices = [(0, 3), (3, 7), (7, 12), (12, 15)]
    for ax, (left, right), label in zip(axs.flat, indices, mttbar_labels):
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
            label=label,
        )
    plt.savefig("ATLAS_MC_multi_comparison.png")
    plt.close()

    # Just take first 3x3
    # data_values = data_values[:3]
    # values_cov = values_cov[:3,:3]
    # mc_data = [a[:3] for a in mc_data]

    # Just take second 4x4
    # data_values = data_values[12:]
    # values_cov = values_cov[12:, 12:]
    # mc_data = [a[12:] for a in mc_data]

    from scipy.linalg import block_diag

    data_values = np.append(data_values[:3], data_values[12:])
    values_cov = block_diag(values_cov[:3, :3], values_cov[12:, 12:])
    mc_data = [np.append(a[:3], a[12:]) for a in mc_data]
    print(data_values)
    print(values_cov)
    print(mc_data)

    generate_json_ctg(
        data_values, values_cov, mc_data, ctg_values, Path("atlas_2D.json")
    )
    run_analysis(Path("atlas_2D.json"))
    # run_validation(Path("atlas_2D.json"), Path("test_atlas_2D.json"))


def multiple_analysis(data, covariance):
    mc_files = sorted(MC_PATH_MULTIPLE.glob("run_0?_LO/MADatNLO.HwU"))
    ctg_values = [-2.0, 0.0001, 2.0]
    ctp_values = [-2.0, 0.0001, 2.0]
    bin_left, bin_right, mc_data = collect_MC_data(mc_files)

    from data import NLO_TO_NNLO_k_factor, LO_PATH, NLO_PATH, k_factor

    # k = (NLO_TO_NNLO_k_factor * k_factor(LO_PATH, NLO_PATH)).tolist()
    k = 1.0

    generate_json_multiple(
        data,
        covariance,
        mc_data,
        k,
        ctg_values,
        ctp_values,
        filename=Path("atlas_2D_multiple.json"),
    )
    run_analysis(Path("atlas_2D_multiple.json"))

def three_op_analysis(data, covariance):
    mc_files = sorted(MC_PATH_THREE_OP.glob("run_??_LO/MADatNLO.HwU"))
    ctg_values = [-2.0, 0.0001, 2.0]
    ctp_values = [-2.0, 0.0001, 2.0]
    ctq8_values = [-2.0, 0.0001, 2.0]
    bin_left, bin_right, mc_data = collect_MC_data(mc_files)

    k = 1

    # data = data[-8:]
    # covariance = covariance[-8:,-8:]
    # mc_data = [a[-8:] for a in mc_data]
    # covariance = np.diag(np.diagonal(covariance))

    generate_json_three_op(
        data,
        covariance,
        mc_data,
        k,
        ctp_values,
        ctg_values,
        ctq8_values,
        filename=Path("atlas_2D_three_op.json"),
    )
    run_analysis(Path("atlas_2D_three_op.json"))


if __name__ == "__main__":

    logging.basicConfig(filename="analysis.log", encoding="utf-8", level=logging.DEBUG)
    mpl_logger = logging.getLogger("matplotlib.texmanager")
    mpl_logger.setLevel(logging.WARNING)
    mpl_logger = logging.getLogger("matplotlib.dviread")
    mpl_logger.setLevel(logging.WARNING)

    data_files = [DATA_PATH / f"Table70{i}.csv" for i in range(1, 5)]
    assert len(data_files) == 4
    data_values = np.array([])
    for f in data_files:
        data_values = np.append(data_values, extract_data(f))

    logging.debug(f"Data Values:\n{data_values}\n{data_values.shape}")

    cov_files = [
        DATA_PATH / f"Table70{i}.csv" if i < 10 else DATA_PATH / f"Table7{i}.csv"
        for i in range(5, 15)
    ]
    assert len(cov_files) == 10, f"Supplied {len(cov_files)} files"
    values_cov = cov_matrix(cov_files)

    logging.debug(f"Covariance Matrix:\n{values_cov}\n{values_cov.shape}")

    three_op_analysis(data_values, values_cov)

    # multiple_analysis(data_values, values_cov)

    # ctg_analysis(data_values, values_cov)
