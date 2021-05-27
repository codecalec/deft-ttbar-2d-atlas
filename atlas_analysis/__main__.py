import sys
import getopt
from pathlib import Path
from typing import List
import logging

from data import extract_data, cov_matrix

import numpy as np

# import deft_hep as deft
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use("science")
rcParams["savefig.dpi"] = 200

DATA_PATH = Path("/home/agv/Documents/Honours/Project/data/1908.07305")
MC_PATH = Path("/home/agv/Documents/Honours/Project/data_generation/ttbar_2D")

if __name__ == "__main__":

    logging.basicConfig(filename="analysis.log", encoding="utf-8", level=logging.DEBUG)
    DATA_PATH = Path("/home/agv/Documents/Honours/Project/data/1908.07305")

    data_files = sorted(DATA_PATH.glob("PTT_MTTBAR_?.csv"))
    values = np.array([])
    for f in data_files:
        values = np.append(values, extract_data(f))
    print(values)

    cov_files = sorted(DATA_PATH.glob("PTT_MTTBAR_COV_*.csv"))
    values_cov = cov_matrix(cov_files)
    print(values_cov.shape)
