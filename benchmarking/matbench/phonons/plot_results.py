"""Plot the perovskites results."""
from pathlib import Path

from ... import visualise

if __name__ == "__main__":
    result_dir = Path(__file__).parents[1] / "results" / "metrics" / "phonons"
    meg_res_dir = result_dir / "megnet"
    prob_res_dir = result_dir / "prob"
    rl = visualise.ResultsLoader(meg_res_dir, prob_res_dir, 5, base_model_name="MEGNet")
    rl.get_average_metrics()
    rl.plot_metric(
        "phonons-mae-change.png",
        title="Matbench Phonons: Adding UnlockNN",
        target=29.5385,
        include_base=True,
    )
    rl.plot_metric(
        "phonons-mae-comp.png",
        title="Matbench Phonons: Effect of Inducing Points",
        target=29.5385,
    )
    rl.plot_metric(
        "phonons-nll.png",
        metric="nll",
        metric_label="Negative Log Likelihood",
        title="Matbench Phonons: Uncertainty Goodness",
        zero_lim=False,
    )
    rl.parity_plot(
        "phonons-parity.png",
        target_name="$\omega^{max} phonons$",
        num_scatter=150,
        title="Matbench Phonons: Probabilistic parity plots"
    )
    rl.calibration_plot(
        "phonons-calibration.png",
        "Matbench Phonons: Calibration curves"
    )
