"""Utilities for data visualisation."""
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIGSIZE = (4, 4)
FONTSIZE = 12


def plot_calibration(
    predicted_pi: np.ndarray,
    observed_pi: np.ndarray,
    fname: Optional[Union[str, Path]] = None,
):
    """Plot miscalibration curve."""
    plt.figure(figsize=FIGSIZE)
    ax_ideal = sns.lineplot([0, 1], [0, 1], label="ideal")
    ax_ideal.lines[0].set_linestyle("--")

    sns.lineplot(predicted_pi, observed_pi)
    plt.fill_between(
        predicted_pi, predicted_pi, observed_pi, alpha=0.2, label="miscalibration area",
    )

    ax_ideal.set_xlabel("Expected cumulative distribution")
    ax_ideal.set_ylabel("Observed cumulative distribution")
    ax_ideal.set_xlim([0, 1])
    ax_ideal.set_ylim([0, 1])

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def plot_sharpness(
    stdevs: np.ndarray,
    sharpness: float,
    coeff_var: float,
    fname: Optional[Union[str, Path]] = None,
):
    """Plot standard deviation distribution and sharpness."""
    plt.figure(figsize=FIGSIZE)
    ax_sharp = sns.distplot(stdevs, kde=False, norm_hist=True)
    ax_sharp.set_xlim(left=0.0)
    ax_sharp.set_xlabel("Predicted standard deviations (eV)")
    ax_sharp.set_ylabel("Normalized frequency")
    ax_sharp.set_yticklabels([])
    ax_sharp.set_yticks([])

    ax_sharp.axvline(x=sharpness, label="sharpness")

    xlim = ax_sharp.get_xlim()
    if sharpness < (xlim[0] + xlim[1]) / 2:
        text = f"\n  Sharpness = {sharpness:.2f} eV\n  C$_v$ = {coeff_var:.2f}"
        h_align = "left"
    else:
        text = f"\nSharpness = {sharpness:.2f} eV  \nC$_v$ = {coeff_var:.2f}  "
        h_align = "right"

    ax_sharp.text(
        x=sharpness,
        y=ax_sharp.get_ylim()[1],
        s=text,
        verticalalignment="top",
        horizontalalignment=h_align,
        fontsize=FONTSIZE,
    )

    if fname:
        plt.savefig(fname)
    else:
        plt.show()
