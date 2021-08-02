from typing import List

import numpy as np
from matplotlib import axes


def plot_comparison(
    ax: axes.Axes,
    data: np.ndarray,
    error: np.ndarray,
    mc: np.ndarray,
    c,
    bin_left: np.ndarray,
    bin_right: np.ndarray,
    label=None,
):

    centres = bin_left + (bin_right - bin_left) / 2
    ax.errorbar(
        centres,
        data,
        xerr=(bin_right - bin_left) / 2,
        yerr=error,
        fmt=".k",
        label="ATLAS data",
    )
    ax.stairs(
        mc,
        np.append(0, bin_right),
        color="r",
        label="MadGraph [$C_{tg}" + f"={c}$]",
    )

    if label:
        ax.plot([], [], "", label=f"{label}")

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
    label=None,
):

    centres = bin_left + (bin_right - bin_left) / 2

    for m, c in zip(mc, ctg):
        ax.errorbar(
            centres,
            m,
            xerr=(bin_right - bin_left) / 2,
            fmt=".",
            ecolor="k",
            label="$C_{tg}" + f"={c}$",
        )

    ax.errorbar(
        centres,
        data,
        xerr=(bin_right - bin_left) / 2,
        yerr=error,
        fmt=".k",
        label="ATLAS data",
    )
    if label:
        ax.plot([], [], "", label=f"{label}")

    ax.set_xlabel(r"$p_t^{T}$ [GeV]")
    ax.set_ylabel(r"$d\sigma/d m_{ttbar}$ [pb GeV$^{-2}$]")
    ax.legend()
