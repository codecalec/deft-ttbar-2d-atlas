from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
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


def data_plot(
    atlas_data: np.ndarray,
    atlas_cov: np.ndarray,
    other_hists: List[Tuple[np.ndarray, str]],
    filename: Optional[Union[str, Path]] = None,
    ratio: bool = False,
):

    bins_1 = [(0, 90), (90, 180), (180, 1000)]
    bins_2 = [(0, 80), (80, 170), (170, 280), (280, 1000)]
    bins_3 = [(0, 80), (80, 170), (170, 270), (270, 370), (370, 1000)]
    bins_4 = [(0, 180), (180, 280), (280, 1000)]

    if ratio:
        fig, ((ax1, ax2, ax3, ax4), (ax1r, ax2r, ax3r, ax4r)) = plt.subplots(
            2,
            4,
            figsize=(9.5, 3.5),
            sharey="row",
            sharex="col",
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax1r.set_ylabel("Ratio")
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(9.5, 3.5), sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    # ax1.set_title(r"$\bf{ATLAS}$", loc="left")
    # ax1.set_title(r"$\sqrt{s}=13\mathrm{TeV}, 36.1\mathrm{fb}^{-1}$", loc="right")
    # ax4.set_xlabel("$p_{t}^{T}$ [GeV]", loc="right")
    # fig.supylabel(
    # r"$\mathrm{d}^2 \sigma / \mathrm{d}p_{T}^{t} \mathrm{d}m_{t\bar{t}}$ [pb/GeV$^{2}$]"
    # )

    ax1.set_title(r"$\bf{dEFT}$", loc="left")
    ax4.set_title(r"$\sqrt{s}=13\mathrm{TeV}, 36.1\mathrm{fb}^{-1}$", loc="right")

    ax1.set_ylabel(
        r"$\mathrm{d}^2 \sigma / \mathrm{d}p_{T}^{t} \mathrm{d}m_{t\bar{t}}$ [pb/GeV$^{2}$]",
        loc="top",
    )
    fig.supxlabel("$p_{t}^{T}$ [GeV]")

    mttbar_labels = [("325", "500"), ("500", "700"), ("700", "1000"), ("1000", "2000")]
    for ax, (low, high) in zip([ax1, ax2, ax3, ax4], mttbar_labels):
        ax.set_yscale("log")
        ax.text(
            # 0.8, 0.5,
            0.6,
            0.9,
            low + r"$< m_{t \bar{t}}$ [GeV]$\leq$" + high,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    atlas_err = np.sqrt(np.diag(atlas_cov))

    get_widths = lambda x: np.array([(j - i) for (i, j) in x])
    widths_1 = get_widths(bins_1)
    widths_2 = get_widths(bins_2)
    widths_3 = get_widths(bins_3)
    widths_4 = get_widths(bins_4)

    get_centres = lambda x, widths: [
        left + width / 2 for (left, _), width in zip(x, widths)
    ]
    centres_1 = get_centres(bins_1, widths_1)
    centres_2 = get_centres(bins_2, widths_2)
    centres_3 = get_centres(bins_3, widths_3)
    centres_4 = get_centres(bins_4, widths_4)

    ax1.errorbar(
        centres_1,
        atlas_data[:3],
        yerr=atlas_err[:3],
        fmt=".k",
        label="Atlas Data",
    )

    ax2.errorbar(
        centres_2,
        atlas_data[3:7],
        yerr=atlas_err[3:7],
        fmt=".k",
    )

    ax3.errorbar(
        centres_3,
        atlas_data[7:12],
        yerr=atlas_err[7:12],
        fmt=".k",
    )

    ax4.errorbar(
        centres_4,
        atlas_data[12:],
        yerr=atlas_err[12:],
        fmt=".k",
    )

    for hist_data, label in other_hists:
        ax1.errorbar(
            centres_1,
            hist_data[:3],
            xerr=widths_1 / 2,
            linestyle="None",
            label=label,
        )

        ax2.errorbar(
            centres_2,
            hist_data[3:7],
            xerr=widths_2 / 2,
            linestyle="None",
        )

        ax3.errorbar(
            centres_3,
            hist_data[7:12],
            xerr=widths_3 / 2,
            linestyle="None",
        )

        ax4.errorbar(
            centres_4,
            hist_data[12:],
            xerr=widths_4 / 2,
            linestyle="None",
        )

        if ratio:
            for axr in [ax1r, ax2r, ax3r, ax4r]:
                axr.axhline(1.0, ls="-", color="k")

            ax1r.errorbar(
                centres_1,
                [1] * 3,
                yerr=atlas_err[:3] / hist_data[:3],
                linestyle="None",
                fmt="k",
            )

            # breakpoint()
            # ax1r.fill_between(
            # [x2 for (_, x2) in bins_1].insert(0,0),
            # y1=np.append(0.0, atlas_err[:3] / hist_data[:3]),
            # y2=np.append(0.0, - atlas_err[:3] / hist_data[:3]),
            # color="r",
            # )

            ax1r.errorbar(
                centres_1,
                atlas_data[:3] / hist_data[:3],
                xerr=widths_1 / 2,
                linestyle="None",
            )

            ax2r.errorbar(
                centres_2,
                [1] * 4,
                yerr=atlas_err[3:7] / hist_data[3:7],
                linestyle="None",
                fmt="k",
            )
            ax2r.errorbar(
                centres_2,
                atlas_data[3:7] / hist_data[3:7],
                xerr=widths_2 / 2,
                linestyle="None",
            )

            ax3r.errorbar(
                centres_3,
                [1] * 5,
                yerr=atlas_err[7:12] / hist_data[7:12],
                linestyle="None",
                fmt="k",
            )
            ax3r.errorbar(
                centres_3,
                atlas_data[7:12] / hist_data[7:12],
                xerr=widths_3 / 2,
                linestyle="None",
            )

            ax4r.errorbar(
                centres_4,
                [1] * 3,
                yerr=atlas_err[12:] / hist_data[12:],
                linestyle="None",
                fmt="k",
            )
            ax4r.errorbar(
                centres_4,
                atlas_data[12:] / hist_data[12:],
                xerr=widths_4 / 2,
                linestyle="None",
            )

    fig.legend(bbox_to_anchor=(0.9, 0.8), loc="upper left")
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.savefig("data_plot.png", bbox_inches="tight")
    plt.clf()
