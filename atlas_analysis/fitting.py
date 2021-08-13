from typing import Optional
import itertools

import numpy as np
from deft_hep import ConfigReader, PredictionBuilder


def _min_func(c: np.ndarray, pb: PredictionBuilder, data: np.ndarray, icov: np.ndarray):
    pred = pb.make_prediction(c)
    diff = pred - data
    return np.dot(diff, np.dot(icov, diff)) + len(data) * np.log(1 / np.linalg.det(icov))


def find_minimum(
    config: ConfigReader,
    pb: PredictionBuilder,
    data: np.ndarray,
    covariance: np.ndarray,
    initial_c: Optional[np.ndarray] = None,
    verbose: bool = True,
):

    from scipy.optimize import minimize

    if initial_c is None:
        initial_c = np.random.random(pb.nOps) - 0.5
        # initial_c = [-1.51,-1.09,-0.65]

    bounds = list(config.prior_limits.values())
    icov = np.linalg.inv(covariance)
    result = minimize(
        _min_func,
        initial_c,
        args=(pb, data, icov),
        bounds=bounds,
        # options={"gtol": 1e-10, "ftol": 10 * np.finfo(float).eps},
    )
    if verbose:
        print("Opt Param:", result.x)
        print("Hess", result.hess_inv.todense())
        print("Err:", np.sqrt(np.diag(result.hess_inv.todense())))
    return result
