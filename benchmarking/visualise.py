"""Tools for visualising benchmarking results."""
from collections import defaultdict
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

Pathy = Union[str, Path]


def read_results(results_dirs: List[Path], metric_fname: str = "metrics.csv") -> pd.DataFrame:
    """Read results from files into a DataFrame."""
    data = defaultdict(list())
    for fold, path in enumerate(results_dirs):
        metric_path = path / metric_fname
        metric_series: pd.Series = pd.read_csv(metric_path, index_col=0, squeeze=True)

        for entry in metric_series.index:
            data[entry].append(metric_series[entry])

        data["fold"].append(fold)

    return pd.DataFrame(data)


class ResultsLoader:
    def __init__(
        self,
        base_model_results: Pathy,
        prob_model_results: Pathy,
        num_folds: int,
        model_name_pattern: str = r"\D*(?:-(?P<points>\d+)-)?(?P<fold>\d+)$",
        metric_fname: str = "metrics.csv",
    ) -> None:
        """Initialize log paths."""
        self.base_model_results = Path(base_model_results)
        self.prob_model_results = Path(prob_model_results)
        self.model_name_pattern = model_name_pattern
        self.num_folds = num_folds
        self.metric_fname = metric_fname

    def results_directories(self, base_model: bool = False) -> Union[List[Path], Tuple[List[Path], List[int]]]:
        """Get the results directories for the model.
        
        Args:
            base_model: If True, get the results for the unmodified model.
                Default (False) gives results for the probabilistic model.
        
        Returns:
            A list of paths, ordered by fold index.

            A list of inducing index points, only if ``base_model = False``.
        
        """
        model_dirs = {}
        results_dir = self.base_model_results if base_model else self.prob_model_results
        for direct in results_dir.iterdir():
            if not direct.is_dir():
                continue
            match = re.match(self.model_name_pattern, direct.name)
            if match:
                model_dirs[int(match.group("fold"))] = direct

        return [model_dirs[idx] for idx in range(self.num_folds)]

    def plot_