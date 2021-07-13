from pathlib import Path
from typing import List

import numpy as np


def generate_json(
    data: np.ndarray,
    covariance: List[List[float]],
    MC_signal: List[np.ndarray],
    ctg_list: List[float],
    filename: Path,
):
    import json

    output_dict = {
        "config": {
            "run_name": "CMS-TOP-ctg",
            "data": {"observable": "$m_{t\\bar{t}}$"},
            "model": {
                "input": "numpy",
                "inclusive_k_factor": 1,
                "c_i_benchmark": 2,
                "max_coeff_power": 1,
                "cross_terms": False,
            },
            "fit": {"n_burnin": 500, "n_total": 5000, "n_walkers": 16},
        }
    }

    test_output_dict = {
        "config": {
            "run_name": "CMS-TOP-ctg",
            "data": {"observable": "$m_{t\\bar{t}}$"},
            "model": {
                "input": "numpy",
                "inclusive_k_factor": 1,
                "c_i_benchmark": 2,
                "max_coeff_power": 1,
                "cross_terms": False,
            },
            "fit": {"n_burnin": 500, "n_total": 3000, "n_walkers": 100},
        }
    }
    MC_signal_model = MC_signal[:3]
    MC_signal_test = MC_signal[3:]
    ctg_list_model = ctg_list[:3]
    ctg_list_test = ctg_list[3:]

    # Data Section
    output_dict["config"]["data"]["bins"] = [0] * len(data)
    output_dict["config"]["data"]["central_values"] = data.tolist()
    test_output_dict["config"]["data"]["bins"] = [0] * len(data)
    test_output_dict["config"]["data"]["central_values"] = data.tolist()

    output_dict["config"]["data"]["covariance_matrix"] = covariance.tolist()
    test_output_dict["config"]["data"][
        "covariance_matrix"
    ] = covariance.tolist()

    # Model Section
    samples = [[1, ctg] for ctg in ctg_list_model]
    samples_test = [[1, ctg] for ctg in ctg_list_test]

    output_dict["config"]["model"]["samples"] = samples
    test_output_dict["config"]["model"]["samples"] = samples_test

    predictions = [mc.tolist() for mc in MC_signal_model]
    predictions_test = [mc.tolist() for mc in MC_signal_test]

    output_dict["config"]["model"]["predictions"] = predictions
    test_output_dict["config"]["model"]["predictions"] = predictions_test

    output_dict["config"]["model"]["prior_limits"] = {"$c_{tG}$": [-2.0, 2.0]}
    test_output_dict["config"]["model"]["prior_limits"] = {
        "$c_{tG}$": [-2.0, 2.0]
    }

    with open(filename, "w") as f:
        json.dump(output_dict, f, indent=4)

    with open(filename.parent / f"test_{filename.name}", "w") as test_f:
        json.dump(test_output_dict, test_f, indent=4)
