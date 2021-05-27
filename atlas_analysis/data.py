from typing import List
from pathlib import Path

import pandas as pd
import numpy as np

import logging


def extract_data(filename: Path) -> np.array:

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

    return cov_data.reshape((height, width))
