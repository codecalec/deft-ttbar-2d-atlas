import logging
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from analysis import run_analysis, run_validation
from config import *
from data import (
    NLO_TO_NNLO_k_factor,
    collect_MC_data,
    cov_matrix,
    extract_data,
    scale_variation,
    k_factor,
    verify_cov_matrix,
)
from matplotlib import rcParams
from plot import data_plot, plot_comparison, plot_comparison_multiple_operator

plt.style.use("science")
rcParams["savefig.dpi"] = 200

PROJECT_PATH = Path("/home/agv/Documents/Honours_Project")
DATA_PATH = PROJECT_PATH / "data/1908.07305"
DATA_GEN_PATH = PROJECT_PATH / "data_generation"
MC_PATH_CTG = DATA_GEN_PATH / "atlas_ttbar_2D"
MC_PATH_MULTIPLE = DATA_GEN_PATH / "atlas_ttbar_2D_multiple"
MC_PATH_THREE_OP = DATA_GEN_PATH / "atlas_ttbar_2D_all"
MC_PATH_THREE_OP_VALID = DATA_GEN_PATH / "atlas_ttbar_2D_all_validation"
MC_PATH_CTG_CTQ = DATA_GEN_PATH / "atlas_ttbar_2D_ctg_ctq_big"
MC_PATH_CTG_CTQ_VALID = DATA_GEN_PATH / "atlas_ttbar_2D_ctg_ctq_validation"

MC_PATH_OP_COMPARISON = DATA_GEN_PATH / "atlas_ttbar_2D_single_op"
MC_PATH_SM = DATA_GEN_PATH / "atlas_ttbar_2D_sm"


def operator_comparison(
    sm_path: Path,
    ctp_paths: List[Path],
    ctg_paths: List[Path],
    ctq_paths: List[Path],
    data: np.ndarray,
    covariance: np.ndarray,
):
    sm_data = collect_MC_data([sm_path])[2][0]  # get sm data as ndarray

    ctp_data = collect_MC_data(ctp_paths)[2]
    ctg_data = collect_MC_data(ctg_paths)[2]
    ctq_data = collect_MC_data(ctq_paths)[2]
    c_values = [-2, 2]

    for datasets, c, label in zip(
        [ctp_data, ctg_data, ctq_data],
        ["ctp", "ctg", "ctq"],
        [r"$C_{t\phi}$", r"$C_{tG}$", r"$C_{tq}$"],
    ):
        filename = f"data_plot_{c}.png"
        data_plot(
            data,
            covariance,
            [
                *[
                    (x, label + f"={str(value)}")
                    for x, value in zip(datasets, c_values)
                ],
                (sm_data, "MG5@LO SM"),
            ],
            filename=filename,
            ratio=True,
        )

    # indices = [6, 7, 8, 9, 10, 11, 12, 13, 14]
    # # indices= np.arange(15)
    # data_err = np.sqrt(np.diag(covariance))
    # z = (sm_data[indices] - data[indices]) / data_err[indices]
    # print("z", z)

    # for data_list, label in zip([ctp_data, ctg_data, ctq_data], ["ctp", "ctg", "ctq"]):
    # for y, value in zip(data_list, c_values):
    # plt.scatter(
    # x[indices], y[indices], marker="_", label=f"MG5 {label}={value}"
    # )

    # plt.scatter(x[indices], sm_data[indices], marker="_", label="MG5 sm")
    # plt.errorbar(
    # x[indices],
    # data[indices],
    # fmt=".k",
    # yerr=data_err[indices],
    # label="ATLAS data",
    # )
    # plt.ylabel(r"$\partial^2 \sigma / \partial m_{t\bar{t}} \partial p_{t}^{T}$")
    # plt.legend()
    # plt.show()


def ctg_analysis(data_values, values_cov):
    mc_files = sorted(MC_PATH_CTG.glob("run_0?_LO/MADatNLO.HwU"))
    ctg_values = [-2.0, 0, 2.0, -1.0, 1.0]
    bin_left, bin_right, mc_data = collect_MC_data(mc_files)

    # fig = plt.figure(figsize=(10, 10))
    # axs = fig.subplots(2, 2)
    # indices = [(0, 3), (3, 7), (7, 12), (12, 15)]
    # mttbar_values = [(350, 500), (500, 700), (700, 1000), (1000, 2000)]
    # mttbar_labels = ["$m_{ttbar}=" + f"{i}-{j}$GeV" for (i, j) in mttbar_values]
    # for ax, (left, right), label in zip(axs.flatten(), indices, mttbar_labels):
    # error = np.sqrt(np.diagonal(values_cov[left:right, left:right]))
    # trimmed_mc_data = mc_data[1][left:right]  # ctg =0.001
    # plot_comparison(
    # ax,
    # data_values[left:right],
    # error,
    # trimmed_mc_data,
    # ctg_values[1],
    # bin_left[left:right],
    # bin_right[left:right],
    # label=label,
    # )
    # plt.savefig("ATLAS_MC_comparison.png")
    # plt.close()

    # fig = plt.figure(figsize=(10, 10))
    # axs = fig.subplots(2, 2)
    # indices = [(0, 3), (3, 7), (7, 12), (12, 15)]
    # for ax, (left, right), label in zip(axs.flat, indices, mttbar_labels):
    # error = np.sqrt(np.diagonal(values_cov[left:right, left:right]))
    # trimmed_mc_data = [m[left:right] for m in mc_data]
    # plot_comparison_multiple_operator(
    # ax,
    # data_values[left:right],
    # error,
    # trimmed_mc_data,
    # ctg_values,
    # bin_left[left:right],
    # bin_right[left:right],
    # label=label,
    # )
    # plt.savefig("ATLAS_MC_multi_comparison.png")
    # plt.close()

    # data_values = data_values[5:]
    # values_cov = values_cov[5:,5:]
    # mc_data = [a[5:] for a in mc_data]
    # data_err = np.sqrt(np.diag(values_cov))
    # print("z", (mc_data[1] - data_values) / data_err)

    # from scipy.linalg import block_diag

    # data_values = np.append(data_values[:2], data_values[3:])
    # values_cov = block_diag(values_cov[:2, :2], values_cov[3:, 3:])
    # mc_data = [np.append(a[:2], a[3:]) for a in mc_data]
    # print(data_values)
    # print(values_cov)
    # print(mc_data)


    values_cov = np.diag(np.diagonal(values_cov))

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

    from data import LO_PATH, NLO_PATH, NLO_TO_NNLO_k_factor, k_factor

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

    mc_files = list(MC_PATH_THREE_OP.glob(r"run_*_LO/MADatNLO.HwU"))
    pattern = re.compile(r"run_(\d+)_LO")
    order = {f: int(re.search(pattern, str(f)).group(1)) - 1 for f in mc_files}
    mc_files = sorted(mc_files, key=lambda x: order[x])

    assert len(mc_files) == 9 ** 3
    _, _, mc_data = collect_MC_data(mc_files)

    ctp_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    ctg_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    ctq8_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]

    k = 1

    # indices = [12,13,14]
    indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    # indices = [6,10,11,12,13,14]
    # When no 6, double solution
    # indices = [7, 8, 9, 10, 11, 12, 13, 14]
    # indices = [7, 8, 9, 10, 11, 12, 13, 14]
    # indices = [0,1,2,3,4,5,6,7,8,9,10]
    # indices = [0,1,3,5,6,7,9,10,11,12,14]
    # indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    data = data[indices]
    covariance = covariance[indices][:, indices]
    mc_data = [a[indices] for a in mc_data]

    covariance = np.diag(np.diagonal(covariance))

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
    run_analysis(Path("atlas_2D_three_op.json"), closure=False)

    # valid_data = [a[indices] for a in valid_data]

    ctp_values = [-1.5, -0.5, 0.5, 1.5]
    ctg_values = [-1.5, -0.5, 0.5, 1.5]
    ctq8_values = [-1.5, -0.5, 0.5, 1.5]

    validation_files = sorted(MC_PATH_THREE_OP_VALID.glob("run_??_LO/MADatNLO.HwU"))
    assert len(validation_files) == 64
    _, _, valid_data = collect_MC_data(validation_files)

    generate_json_three_op(
        data,
        covariance,
        valid_data,
        k,
        ctp_values,
        ctg_values,
        ctq8_values,
        filename=Path("atlas_2D_three_op_test.json"),
    )

    # run_validation(Path("atlas_2D_three_op.json"), Path("atlas_2D_three_op_test.json"))


def ctg_ctq_analysis(data, covariance, scale_variance=1):

    # mc_files = MC_PATH_THREE_OP.glob(r"run_*_LO/MADatNLO.HwU")
    # pattern = re.compile(r"run_(\d+)_LO")

    # mc_files = sorted(
        # filter(
            # lambda f: int(re.search(pattern, str(f)).group(1)) in range(51, 76),
            # mc_files,
        # )
    # )

    mc_files = list(MC_PATH_CTG_CTQ.glob(r"run_*_LO/MADatNLO.HwU"))
    pattern = re.compile(r"run_(\d+)_LO")
    order = {f: int(re.search(pattern, str(f)).group(1)) - 1 for f in mc_files}
    mc_files = sorted(mc_files, key=lambda x: order[x])

    assert len(mc_files) == 169
    _, _, mc_data = collect_MC_data(mc_files)

    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    data = data[indices]
    covariance = covariance[indices][:, indices]
    mc_data = [a[indices] for a in mc_data]
    covariance = np.diag(np.diagonal(covariance))

    ctg_list = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]
    ctq8_list = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]

    filename = Path("atlas_2D_ctg_ctq.json")
    # if (scale_variance > 1.0).any() :
        # filename = Path("atlas_2D_ctg_ctq_higher.json")
    # elif (scale_variance < 1.0).any() :
        # filename = Path("atlas_2D_ctg_ctq_lower.json")
    mc_data = mc_data * scale_variance
    # mc_data = [a * scale_variance for a in mc_data]

    generate_json_ctg_ctq(
        data,
        covariance,
        mc_data,
        k_factor=1,
        ctg_list=ctg_list,
        ctq8_list=ctq8_list,
        filename=filename,
    )
    run_analysis(filename)

    ctg_values = [-2.5, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.5]
    ctq8_values = [-2.5, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.5]


    validation_files = list(MC_PATH_CTG_CTQ_VALID.glob("run_*_LO/MADatNLO.HwU"))
    assert len(validation_files) == 100
    order = {f: int(re.search(pattern, str(f)).group(1)) - 1 for f in validation_files}
    validation_files = sorted(validation_files, key=lambda x: order[x])

    _, _, valid_data = collect_MC_data(validation_files)
    valid_data = [a[indices] for a in valid_data]

    generate_json_ctg_ctq(
        data,
        covariance,
        valid_data,
        k_factor=1,
        ctg_list=ctg_values,
        ctq8_list=ctq8_values,
        filename=Path("atlas_2D_ctg_ctq_test.json"),
    )
    run_validation(Path("atlas_2D_ctg_ctq.json"), Path("atlas_2D_ctg_ctq_test.json"))


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

    sm_path = MC_PATH_SM / "run_01_LO" / "MADatNLO.HwU"
    scale_lower = 1 - scale_variation(sm_path)
    scale_upper = 1 + scale_variation(sm_path)

    ctg_ctq_analysis(data_values, values_cov, scale_variance=scale_upper)

    # result = verify_cov_matrix(values_cov)

    # print(result[:3][:,:3])
    # print(result[3:7][:,3:7])
    # print(result[7:12][:,7:12])
    # print(result[12:][:,12:])

    # three_op_analysis(data_values, values_cov)

    # multiple_analysis(data_values, values_cov)

    # ctg_analysis(data_values, values_cov)

    # comparison_paths = list(MC_PATH_OP_COMPARISON.glob("run_??_LO/MADatNLO.HwU"))
    # assert len(comparison_paths) == 6
    # ctp_paths = comparison_paths[:2]
    # ctg_paths = comparison_paths[2:4]
    # ctq_paths = comparison_paths[4:]

    # sm_path = list(MC_PATH_SM.glob("run_??_LO/MADatNLO.HwU"))[0]
    # operator_comparison(
        # sm_path, ctp_paths, ctg_paths, ctq_paths, data_values, values_cov
    # )
