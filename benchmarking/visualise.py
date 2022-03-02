"""Tools for visualising benchmarking results."""
from collections import defaultdict
from operator import itemgetter
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

Pathy = Union[str, Path]


sns.set_style()


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

    def prob_results_directories(self) -> Dict[int, List[Path]]:
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
            model_dirs[key] = [tup[1] for tup in sorted(val, key=itemgetter(0))]

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
        prob_df = read_prob_results(prob_result_dirs)
        if include_base:
            base_df = read_base_results(
                self.base_results_directories(), base_model_name=self.base_model_name
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
