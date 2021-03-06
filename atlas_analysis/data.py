import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from deft_hep.helper import convert_hwu_to_numpy

LO_PATH = Path(
    "/home/agv/Documents/Honours_Project/data_generation/atlas_ttbar_2D_sm/run_01_LO/MADatNLO.HwU"
)
NLO_PATH = Path(
    "/home/agv/Documents/Honours_Project/data_generation/atlas_ttbar_2D_sm/run_01/MADatNLO.HwU"
)

LO_1D_PATH = Path(
    "/home/agv/Documents/Honours_Project/data_generation/atlas_ttbar_1D_sm/run_01_LO/MADatNLO.HwU"
)
NLO_1D_PATH = Path(
    "/home/agv/Documents/Honours_Project/data_generation/atlas_ttbar_1D_sm/run_01/MADatNLO.HwU"
)


NLO_XSEC = 6.741e2  #  +- 0.023e2 from https://arxiv.org/abs/1405.0301
NNLO_XSEC = 831.76  # from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
NLO_TO_NNLO_k_factor = NNLO_XSEC / NLO_XSEC


def _per_bin_k_factor(LO_path: Path, NLO_path: Path, num_hist: int = 15):

    _, _, LO_values = convert_hwu_to_numpy(LO_path, num_hist)
    _, _, NLO_values = convert_hwu_to_numpy(NLO_path, num_hist)
    assert len(LO_values) == num_hist

    k = NLO_values / LO_values

    return k


k_factor = _per_bin_k_factor(LO_PATH, NLO_PATH) * NLO_TO_NNLO_k_factor
k_factor_1D = _per_bin_k_factor(LO_1D_PATH, NLO_1D_PATH, 9) * NLO_TO_NNLO_k_factor


def scale_variation(file: Path, verbose: bool = False):
    values = []

    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "<histogram>" in line:
                hist_info = lines[i + 1].strip().split()
                dy = float(hist_info[3])
                central = float(hist_info[2])
                variation = (dy / 2) / central
                assert variation < 1.0
                values.append(variation)
    if verbose:
        print(f"Scale variation from {file}")
        print(values)
    return np.array(values)


def collect_MC_data(
    files: List[Path],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:

    mttbar_bin_widths = np.array([175] * 3 + [200] * 4 + [300] * 5 + [1000] * 3)

    if not files:
        raise Exception("No files available")

    mc_data = []
    for f in files:
        bin_left, bin_right, values = convert_hwu_to_numpy(f, 15)

        scaled_values = values / (bin_right - bin_left) / mttbar_bin_widths * k_factor

        mc_data.append(scaled_values)
    return bin_left, bin_right, mc_data


def extract_data(filename: Path) -> np.ndarray:

    values = []
    with filename.open() as f:
        for _ in range(11):
            next(f)

        while (line := next(f)) != "\n":
            comma = 0
            index = 0
            while comma < 4:
                index = line.find(",", index + 1)
                comma += 1
            line = line[:index].strip().split(",")
            values.append(float(line[-1]))
    return np.array(values)


def cov_matrix(files: List[Path]) -> np.ndarray:
    """Return the constructed covariance matrix from list of csv files"""

    # Convert files to ndarrays
    file_index = 0
    arr_list = []
    for col in range(4):
        for row in range(col + 1):
            f = files[file_index]
            file_index += 1
            logging.debug(f"reading file:{f} {col} {row}")
            arr = _extract_cov(f, col, row)
            arr_list.append(arr)

    # Concat matrices together
    item = 0
    heights = [arr.shape[0] for arr in arr_list[-4:]]
    cov_matrix = None
    for col in range(4):
        col_arr = None
        for row in range(col + 1):
            arr = arr_list[item]
            item += 1
            if col_arr is not None:
                col_arr = np.concatenate((col_arr, arr), axis=0)
            else:
                col_arr = arr

        for height in heights[(col + 1) :]:
            empty = np.zeros((height, col_arr.shape[1]))
            col_arr = np.concatenate((col_arr, empty), axis=0)

        if cov_matrix is not None:
            cov_matrix = np.concatenate((cov_matrix, col_arr), axis=1)
        else:
            cov_matrix = col_arr

    assert cov_matrix.shape[0] == cov_matrix.shape[1]
    for row in range(cov_matrix.shape[0]):
        for col in range(row, cov_matrix.shape[1]):
            cov_matrix[col][row] = cov_matrix[row][col]

    assert np.allclose(cov_matrix, cov_matrix.T)

    return cov_matrix


def _extract_cov(f: Path, col: int, row: int) -> np.ndarray:
    """Hard coded matrix sizes for sub covariant matrices"""
    if col == 0:
        arr = _read_cov_matrix(f, 3, 3)
    elif col == 1:
        if row == 0:
            arr = _read_cov_matrix(f, 4, 3)
        else:
            arr = _read_cov_matrix(f, 4, 4)
    elif col == 2:
        if row == 0:
            arr = _read_cov_matrix(f, 5, 3)
        elif row == 1:
            arr = _read_cov_matrix(f, 5, 4)
        else:
            arr = _read_cov_matrix(f, 5, 5)
    elif col == 3:
        if row == 0:
            arr = _read_cov_matrix(f, 3, 3)
        elif row == 1:
            arr = _read_cov_matrix(f, 3, 4)
        elif row == 2:
            arr = _read_cov_matrix(f, 3, 5)
        else:
            arr = _read_cov_matrix(f, 3, 3)
    return arr


def _read_cov_matrix(path: Path, width: int, height: int) -> np.ndarray:
    cov_data = np.genfromtxt(path, delimiter=",", skip_header=10)[
        :, -1
    ]  # get last column

    if width * height is not len(cov_data):
        raise ValueError(
            f"File does not have correct number of values [{width*height}]"
        )

    return cov_data.reshape((width, height)).T


def verify_cov_matrix(cov: np.ndarray) -> np.ndarray:
    from itertools import product

    err = np.sqrt(np.diag(cov))
    result = np.empty(cov.shape)
    for (i, err_i), (j, err_j) in product(enumerate(err), enumerate(err)):
        result[i, j] = cov[i, j] / (err_i * err_j)
    return result


def collect_MC_data_1D(
    files: List[Path],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:

    if not files:
        raise Exception("No files available")

    mc_data = []
    for f in files:
        bin_left, bin_right, values = convert_hwu_to_numpy(f, 9)

        scaled_values = values / (bin_right - bin_left) * k_factor_1D

        mc_data.append(scaled_values)
    return bin_left, bin_right, mc_data


def extract_data_mttbar(filename: Path) -> np.ndarray:
    return np.loadtxt(filename, delimiter=",", skiprows=23)[:, 4]


def cov_matrix_mttbar(filename: Path) -> np.ndarray:
    return np.loadtxt(filename, delimiter=",", skiprows=10)[:, 6].reshape((9, 9))
