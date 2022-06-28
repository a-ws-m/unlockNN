"""Tools for visualising benchmarking results."""
import re
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

Pathy = Union[str, Path]


sns.set_style()


def calc_pis(
    residuals: np.ndarray, stddevs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate probability intervals for given set of residuals and standard deviations."""
    norm_resids = residuals / stddevs
    predicted_pi = np.linspace(0, 1, 100)
    bounds = norm.ppf(predicted_pi)
    observed_pi = np.array([np.count_nonzero(norm_resids <= bound) for bound in bounds])
    observed_pi = observed_pi / norm_resids.size
    return predicted_pi, observed_pi


def parity_plot(
    prob_df: pd.DataFrame,
    fname: str,
    target_name: str,
    top_padding: float = 1.0,
    bottom_padding: float = 0.5,
):
    """Plot the parity with 95% CI error bars for a dataset."""
    PREDICTED_NAME = f"Predicted {target_name}"
    OBSERVED_NAME = f"Observed {target_name}"

    # MARKER_COLOUR = tuple(np.array([168, 190, 240]) / 255)

    plot_df = pd.DataFrame(
        {
            PREDICTED_NAME: prob_df["Predicted value"],
            OBSERVED_NAME: prob_df["True value"],
            "CI": prob_df["Predicted standard deviation"] * 2,
            "# Inducing index points": prob_df["# Inducing points"],
            "Fold": prob_df["Fold"],
        }
    )
    Y_LIMS = (
        min(plot_df[PREDICTED_NAME].min(), plot_df[OBSERVED_NAME].min())
        - bottom_padding,
        max(plot_df[PREDICTED_NAME].max(), plot_df[OBSERVED_NAME].max()) + top_padding,
    )
    g = sns.FacetGrid(
        data=plot_df, height=7, col="# Inducing index points", margin_titles=True, hue="Fold"
    )
    g.map_dataframe(
        plt.errorbar,
        OBSERVED_NAME,
        PREDICTED_NAME,
        "CI",
        marker="o",
        # color=MARKER_COLOUR,
        linestyle="",
        alpha=0.7,
        markeredgewidth=1,
        markeredgecolor="black",
    )

    g.map(
        sns.lineplot,
        x=Y_LIMS,
        y=Y_LIMS,
        label="Ideal",
        color="black",
        linestyle="--",
        marker="",
    )
    g.set(xlim=Y_LIMS, ylim=Y_LIMS, aspect="equal")
    g.set_titles(col_template="{col_name} points")
    plt.savefig(
        fname,
        # transparent=True,
        bbox_inches="tight"
    )


def plot_calibration(prob_df: pd.DataFrame, fname):
    """Plot a calibration curve for a given dataset."""
    PREDICTED_NAME = "Predicted cumulative distribution"
    OBSERVED_NAME = "Observed cumulative distribution"

    # LINE_COLOUR = tuple(np.array([0, 40, 85]) / 255)
    # FILL_COLOUR = tuple(np.array([203, 216, 246]) / 255)

    data = None
    for num_inducing_points, subdf in prob_df.groupby("# Inducing points"):
        predicted_pi, observed_pi = calc_pis(
            subdf["True value"] - subdf["Predicted value"],
            subdf["Predicted standard deviation"],
        )
        sub_plot_df = pd.DataFrame(
            {PREDICTED_NAME: predicted_pi, OBSERVED_NAME: observed_pi}
        )
        sub_plot_df["# Inducing index points"] = num_inducing_points
        if data is None:
            data = sub_plot_df
        else:
            data = pd.concat([data, sub_plot_df], ignore_index=True)

    g = sns.relplot(
        data=data,
        x=PREDICTED_NAME,
        y=OBSERVED_NAME,
        kind="line",
        color="black",
        col="# Inducing index points",
        label="Actual",
        facet_kws={"margin_titles": True},
    )
    g.map(
        sns.lineplot,
        x=(0, 1),
        y=(0, 1),
        color="black",
        linestyle="--",
        marker="",
        label="Ideal",
    )
    g.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    for num_inducing_points, ax in g.axes_dict.items():
        data_slice = data[data["# Inducing index points"] == num_inducing_points]
        ax.fill_between(
            data_slice[PREDICTED_NAME],
            data_slice[PREDICTED_NAME],
            data_slice[OBSERVED_NAME],
            alpha=0.7,
            # color=FILL_COLOUR,
        )
    plt.savefig(
        fname,
        # transparent=True,
        bbox_inches="tight"
    )


def read_prob_results(
    results_dirs: Dict[int, List[Path]], metric_fname: str = "metrics.csv"
) -> pd.DataFrame:
    """Read results from files into a DataFrame."""
    data = defaultdict(list)
    for points, dirs in results_dirs.items():
        for fold, path in enumerate(dirs):
            metric_path = path / metric_fname
            metric_series: pd.Series = pd.read_csv(metric_path, index_col=0).squeeze(
                "columns"
            )

            for entry in metric_series.index:
                data[entry].append(metric_series[entry])

            data["fold"].append(fold)
            data["points"].append(points)
            data["Model name"].append(f"Unlock-{points}")

    return pd.DataFrame(data)


def read_base_results(
    results_dirs: List[Path],
    base_model_name: str = "Base",
    metric_fname: str = "metrics.csv",
):
    """Read results from files into a DataFrame."""
    data = defaultdict(list)
    for fold, path in enumerate(results_dirs):
        metric_path = path / metric_fname
        metric_series: pd.Series = pd.read_csv(metric_path, index_col=0).squeeze(
            "columns"
        )

        for entry in metric_series.index:
            data[entry].append(metric_series[entry])

        data["fold"].append(fold)
        data["Model name"].append(base_model_name)

    return pd.DataFrame(data)


class ResultsLoader:
    def __init__(
        self,
        base_model_results: Pathy,
        prob_model_results: Pathy,
        num_folds: int,
        model_name_pattern: str = r"\D*(?:-(?P<points>\d+)-)?(?P<fold>\d+)$",
        base_model_name: str = "Base",
        metric_fname: str = "metrics.csv",
    ) -> None:
        """Initialize log paths."""
        self.base_model_results = Path(base_model_results)
        self.prob_model_results = Path(prob_model_results)
        self.model_name_pattern = model_name_pattern
        self.num_folds = num_folds
        self.base_model_name = base_model_name
        self.metric_fname = metric_fname

    def prob_results_directories(self, with_fold: bool=False) -> Dict[int, List[Path]]:
        """Get the results directories for the model.

        Returns:
            A dictionary of ``{num_inducing_points: [path]}`` with the ``path``s
            ordered by fold index.

        """
        model_dirs = defaultdict(list)
        for direct in self.prob_model_results.iterdir():
            if not direct.is_dir():
                continue

            match = re.match(self.model_name_pattern, direct.name)
            if match:
                num_points = match.group("points")
                if num_points:
                    model_dirs[int(num_points)].append(
                        (int(match.group("fold")), direct)
                    )

        for key, val in model_dirs.items():
            # Sort based on fold
            model_dirs[key] = list(sorted(val, key=itemgetter(0)))
            if not with_fold:
                model_dirs[key] = [tup[1] for tup in model_dirs[key]]

        return dict(model_dirs)

    def base_results_directories(self) -> List[Path]:
        """Get the results directories for the baseline model."""
        model_dirs = []
        for direct in self.base_model_results.iterdir():
            if not direct.is_dir():
                continue
            match = re.match(self.model_name_pattern, direct.name)
            if match:
                model_dirs.append((int(match.group("fold")), direct))
        return [item[1] for item in sorted(model_dirs, key=itemgetter(0))]

    def get_all_prob_predictions(self) -> pd.DataFrame:
        """Get a dataframe with the combined test set predictions across all folds."""
        template_df = None
        for num_inducing_points, entries in self.prob_results_directories(with_fold=True).items():
            for fold, path_ in entries:
                loaded_df = pd.read_csv(path_ / "predictions.csv", index_col=0)
                loaded_df["# Inducing points"] = num_inducing_points
                loaded_df["Fold"] = fold
                if template_df is None:
                    template_df = loaded_df
                else:
                    template_df = pd.concat([template_df, loaded_df])
        return template_df

    def calibration_plot(self, fname: str):
        """Make a calibration plot for the test set predictions."""
        predictions_df = self.get_all_prob_predictions()
        plot_calibration(predictions_df, fname)

    def parity_plot(
        self,
        fname: str,
        target_name: str,
        num_scatter: int,
        top_padding: float = 1.0,
        bottom_padding: float = 0.5,
    ):
        """Make a parity plot for the test set predictions."""
        predictions_df = self.get_all_prob_predictions()
        predictions_df["data_idx"] = predictions_df.index
        num_test_points = int(predictions_df["data_idx"].max())
        idxs_to_plot = np.random.randint(0, num_test_points + 1, num_scatter)
        plot_df = predictions_df[[data_idx in idxs_to_plot for data_idx in predictions_df["data_idx"]]]
        parity_plot(plot_df, fname, target_name, top_padding, bottom_padding)

    def plot_metric(
        self,
        fname: Pathy,
        metric: str = "mae",
        metric_label: str = "Mean absolute error",
        include_base: bool = False,
        title: str = "",
        zero_lim: bool = True,
        target: Optional[float] = None,
        transparent: bool = False,
    ):
        """Plot a comparison of the model's errors versus the baseline model.

        Args:
            fname: Where to save the plot.
            target: An optional target to plot as a vertical dashed line.

        """
        prob_result_dirs = self.prob_results_directories()
        plot_order = [f"Unlock-{points}" for points in sorted(prob_result_dirs.keys())]
        prob_df = read_prob_results(prob_result_dirs, self.metric_fname)
        if include_base:
            base_df = read_base_results(
                self.base_results_directories(),
                base_model_name=self.base_model_name,
                metric_fname=self.metric_fname,
            )
            plot_df = pd.concat([base_df, prob_df], ignore_index=True)
            plot_order.insert(0, self.base_model_name)
        else:
            plot_df = prob_df

        plot_df[metric_label] = plot_df[metric]

        plt.figure()
        sns.stripplot(
            data=plot_df,
            x=metric_label,
            y="Model name",
            dodge=0.2,
            size=5,
            color=".8",
            linewidth=0,
            marker="X",
            palette="pastel",
            order=plot_order,
        )
        sns.pointplot(
            data=plot_df,
            x=metric_label,
            y="Model name",
            dodge=None,
            scale=1.25,
            markers="d",
            ci=None,
            palette="dark",
            order=plot_order,
        )
        if target:
            plt.axvline(x=target, ls="--", c=".3")
        if zero_lim:
            plt.xlim(left=0)
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, transparent=transparent)
