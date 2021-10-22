import re
from itertools import product
from pathlib import Path

from config import generate_json_ctg, generate_json_ctg_ctq
from data import collect_MC_data_1D
from analysis import run_analysis, run_validation

import numpy as np
import matplotlib.pyplot as plt

PROJECT_PATH = Path("/home/agv/Documents/Honours_Project")
DATA_PATH = PROJECT_PATH / "data/1908.07305"
DATA_GEN_PATH = PROJECT_PATH / "data_generation"

MC_PATH_CTG_CTQ_1D = DATA_GEN_PATH / "atlas_ttbar_1D_ctg_ctq_big"
MC_PATH_CTG_CTQ_1D_VALID = DATA_GEN_PATH / "atlas_ttbar_1D_ctg_ctq_validation"

def ctg_analysis_1D(data, covariance, scale_variance=1):
    mc_files = list(MC_PATH_CTG_CTQ_1D.glob(r"run_*_LO/MADatNLO.HwU"))
    pattern = re.compile(r"run_(\d+)_LO")
    order = {f: int(re.search(pattern, str(f)).group(1)) - 1 for f in mc_files}
    mc_files = sorted(mc_files, key=lambda x: order[x])

    assert len(mc_files) == 169
    _, _, mc_data = collect_MC_data_1D(mc_files)
    mc_data = mc_data * scale_variance

    covariance = np.diag(np.diagonal(covariance))

    mc_ctg_data = []
    ctg_list = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]
    ctq8_list = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]

    for mc, (ctg, ctq8) in zip(mc_data, product(ctg_list, ctq8_list)):
        if ctq8 == 0:
            mc_ctg_data.append(mc)

    filename = Path("atlas_1D_ctg.json")
    generate_json_ctg(
        data,
        covariance,
        mc_ctg_data,
        ctg_list=ctg_list,
        filename=filename,
    )
    run_analysis(filename, is1D=True)
    # run_validation(Path("atlas_1D_ctg.json"), Path("atlas_1D_ctg.json"))


def ctg_ctq_analysis_1D(data, covariance, scale_variance=1):
    mc_files = list(MC_PATH_CTG_CTQ_1D.glob(r"run_*_LO/MADatNLO.HwU"))
    pattern = re.compile(r"run_(\d+)_LO")
    order = {f: int(re.search(pattern, str(f)).group(1)) - 1 for f in mc_files}
    mc_files = sorted(mc_files, key=lambda x: order[x])

    assert len(mc_files) == 169
    _, _, mc_data = collect_MC_data_1D(mc_files)

    covariance = np.diag(np.diagonal(covariance))

    mc_data = mc_data * scale_variance

    ctg_list = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]
    ctq8_list = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]

    for mc, (ctg, ctq8) in zip(mc_data, product(ctg_list, ctq8_list)):
        if ctg == 0:
            if isinstance(ctq8, int):
                plt.scatter(range(len(mc)), mc, label=f"{ctq8}")
    err = np.sqrt(np.diagonal(covariance))
    plt.errorbar(range(len(data)), data, yerr=err)
    plt.legend()
    plt.show()

    filename = Path("atlas_1D_ctg_ctq.json")
    generate_json_ctg_ctq(
        data,
        covariance,
        mc_data,
        k_factor=1,
        ctg_list=ctg_list,
        ctq8_list=ctq8_list,
        filename=filename,
    )
    run_analysis(filename, is1D=True)

    ctg_val_list = [-2.5, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.5]
    ctq8_val_list = [-2.5, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.5]

    validation_files = list(MC_PATH_CTG_CTQ_1D_VALID.glob("run_*_LO/MADatNLO.HwU"))
    assert len(validation_files) == 100
    order = {f: int(re.search(pattern, str(f)).group(1)) - 1 for f in validation_files}
    validation_files = sorted(validation_files, key=lambda x: order[x])
    _, _, valid_data = collect_MC_data_1D(validation_files)

    filename_val = Path("atlas_1D_ctg_ctq_val.json")
    generate_json_ctg_ctq(
        data,
        covariance,
        valid_data,
        k_factor=1,
        ctg_list=ctg_val_list,
        ctq8_list=ctq8_val_list,
        filename=filename_val,
    )

    run_validation(filename, filename_val)
