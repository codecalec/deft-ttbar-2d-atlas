import itertools
import json
from pathlib import Path
from typing import List, Union

import numpy as np


def generate_json_ctg(
    data: np.ndarray,
    covariance: List[List[float]],
    MC_signal: List[np.ndarray],
    ctg_list: List[float],
    filename: Path,
):

    output_dict = {
        "config": {
            "run_name": "ATLAS-ctg",
            "data": {"observable": "$p_{t}^{T}$"},
            "model": {
                "input": "numpy",
                "inclusive_k_factor": 1,
                "c_i_benchmark": 2,
                "max_coeff_power": 1,
                "cross_terms": False,
            },
            "fit": {"n_burnin": 1000, "n_total": 10000, "n_walkers": 16},
        }
    }

    # test_output_dict = {
        # "config": {
            # "run_name": "ATLAS-ctg",
            # "data": {"observable": "$p_{t}^{T}$"},
            # "model": {
                # "input": "numpy",
                # "inclusive_k_factor": 1,
                # "c_i_benchmark": 2,
                # "max_coeff_power": 1,
                # "cross_terms": False,
            # },
            # "fit": {"n_burnin": 500, "n_total": 3000, "n_walkers": 100},
        # }
    # }
    # MC_signal_model = MC_signal[:3]
    # MC_signal_test = MC_signal[3:]
    # ctg_list_model = ctg_list[:3]
    # ctg_list_test = ctg_list[3:]

    # Data Section
    output_dict["config"]["data"]["bins"] = list(range(0, len(data) + 1))
    output_dict["config"]["data"]["central_values"] = data.tolist()
    # test_output_dict["config"]["data"]["bins"] = [0] * len(data)
    # test_output_dict["config"]["data"]["central_values"] = data.tolist()

    output_dict["config"]["data"]["covariance_matrix"] = covariance.tolist()
    # test_output_dict["config"]["data"]["covariance_matrix"] = covariance.tolist()

    # Model Section
    samples = [[1, ctg] for ctg in ctg_list]
    samples_test = [[1, ctg] for ctg in ctg_list]

    output_dict["config"]["model"]["samples"] = samples
    # test_output_dict["config"]["model"]["samples"] = samples_test

    predictions = [mc.tolist() for mc in MC_signal]
    # predictions_test = [mc.tolist() for mc in MC_signal_test]

    output_dict["config"]["model"]["predictions"] = predictions
    # test_output_dict["config"]["model"]["predictions"] = predictions_test

    output_dict["config"]["model"]["prior_limits"] = {"c_{tG}": [-5.0, 5.0]}
    # test_output_dict["config"]["model"]["prior_limits"] = {"c_{tG}": [-5.0, 5.0]}

    with open(filename, "w") as f:
        json.dump(output_dict, f, indent=4)

    # with open(filename.parent / f"test_{filename.name}", "w") as test_f:
        # json.dump(test_output_dict, test_f, indent=4)


def generate_json_multiple(
    data: np.ndarray,
    covariance: List[List[float]],
    MC_signal: List[np.ndarray],
    k_factor: Union[float, List[np.ndarray]],
    ctg_list: List[float],
    ctp_list: List[float],
    filename: Path,
):
    """Make .json for ctg and ctp MC generated events"""

    output_dict = {
        "config": {
            "run_name": "ATLAS-ctg-ctp",
            "data": {"observable": "$m_{t\\bar{t}}$"},
            "model": {
                "input": "numpy",
            },
            "fit": {"n_burnin": 500, "n_total": 20000, "n_walkers": 10},
        }
    }
    predictions = [mc.tolist() for mc in MC_signal]
    output_dict["config"]["data"]["bins"] = list(range(0, len(data) + 1))
    output_dict["config"]["data"]["central_values"] = data.tolist()

    output_dict["config"]["data"]["covariance_matrix"] = covariance.tolist()

    samples = [[1, ctg, ctp] for ctg, ctp in itertools.product(ctg_list, ctp_list)]
    output_dict["config"]["model"]["samples"] = samples

    predictions = [mc.tolist() for mc in MC_signal]
    output_dict["config"]["model"]["predictions"] = predictions
    output_dict["config"]["model"]["prior_limits"] = {
        "c_{tG}": [-5.0, 5.0],
        r"c_{t\phi}": [-5.0, 5.0],
    }

    output_dict["config"]["model"]["inclusive_k_factor"] = k_factor

    with open(filename, "w") as f:
        json.dump(output_dict, f, indent=4)


def generate_json_three_op(
    data: np.ndarray,
    covariance: List[List[float]],
    MC_signal: List[np.ndarray],
    k_factor: Union[float, List[np.ndarray]],
    ctp_list: List[float],
    ctg_list: List[float],
    ctq8_list: List[float],
    filename: Path,
):
    """Make .json for ctp, ctG, and ctq8 MC generated events"""

    output_dict = {
        "config": {
            "run_name": "ATLAS-ctg-ctp-ctq8",
            "data": {"observable": "$p_{t}^{T}$"},
            "model": {
                "input": "numpy",
            },
            "fit": {"n_burnin": 1000, "n_total": 50000, "n_walkers": 10},
        }
    }
    predictions = [mc.tolist() for mc in MC_signal]
    output_dict["config"]["data"]["bins"] = list(range(0, len(data) + 1))
    output_dict["config"]["data"]["central_values"] = data.tolist()

    output_dict["config"]["data"]["covariance_matrix"] = covariance.tolist()

    samples = [
        [1, ctp, ctg, ctq]
        for ctp, ctg, ctq in itertools.product(ctp_list, ctg_list, ctq8_list)
    ]
    output_dict["config"]["model"]["samples"] = samples

    predictions = [mc.tolist() for mc in MC_signal]
    output_dict["config"]["model"]["predictions"] = predictions
    output_dict["config"]["model"]["prior_limits"] = {
        r"c_{t\phi}": [-12.0, 6.0],
        "c_{tG}": [-5.0, 5.0],
        "c_{tq}^{8}": [-5.0, 5.0],
    }

    output_dict["config"]["model"]["inclusive_k_factor"] = k_factor

    with open(filename, "w") as f:
        json.dump(output_dict, f, indent=4)


def generate_json_ctg_ctq(
    data: np.ndarray,
    covariance: List[List[float]],
    MC_signal: List[np.ndarray],
    k_factor: Union[float, List[np.ndarray]],
    ctg_list: List[float],
    ctq8_list: List[float],
    filename: Path,
):
    output_dict = {
        "config": {
            "run_name": "ATLAS-ctg-ctq8",
            "data": {"observable": "$p_{t}^{T}$"},
            "model": {
                "input": "numpy",
            },
            "fit": {"n_burnin": 1000, "n_total": 80000, "n_walkers": 20},
        }
    }
    predictions = [mc.tolist() for mc in MC_signal]
    output_dict["config"]["data"]["bins"] = list(range(0, len(data) + 1))
    output_dict["config"]["data"]["central_values"] = data.tolist()

    output_dict["config"]["data"]["covariance_matrix"] = covariance.tolist()

    samples = [[1, ctg, ctq] for ctg, ctq in itertools.product(ctg_list, ctq8_list)]
    output_dict["config"]["model"]["samples"] = samples

    predictions = [mc.tolist() for mc in MC_signal]
    output_dict["config"]["model"]["predictions"] = predictions
    output_dict["config"]["model"]["prior_limits"] = {
        "c_{tG}": [-5.0, 5.0],
        "c_{tq}^{8}": [-5.0, 5.0],
    }

    output_dict["config"]["model"]["inclusive_k_factor"] = k_factor

    with open(filename, "w") as f:
        json.dump(output_dict, f, indent=4)
