"""Plot the perovskites results."""
from pathlib import Path

from ... import visualise

if __name__ == "__main__":
    result_dir = Path(__file__).parents[1] / "results" / "metrics" / "perovskites"
    meg_res_dir = result_dir / "megnet"
    prob_res_dir = result_dir / "prob"
    rl = visualise.ResultsLoader(meg_res_dir, prob_res_dir, 5, base_model_name="MEGNet")
    rl.plot_metric(
        "perovskites-mae-change.png",
        title="Matbench Perovskites: Adding UnlockNN",
        target=0.0288,
        include_base=True,
    )
    rl.plot_metric(
        "perovskites-mae-comp.png",
        title="Matbench Perovskites: Effect of Inducing Points",
        target=0.0288,
    )
    rl.plot_metric(
        "perovskites-nll.png",
        metric="nll",
        metric_label="Negative Log Likelihood",
        title="Matbench Perovskites: Uncertainty Goodness",
        zero_lim=False,
    )
