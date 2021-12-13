"""Benchmarking script for the matbench phonons dataset."""
from pathlib import Path

from .. import matbench_utils

if __name__ == "__main__":
    trainer = matbench_utils.MatbenchTrainer(
        "matbench_perovskites",
        "https://ml.materialsproject.org/projects/matbench_perovskites.json.gz",
        "e_form",
    )
    trainer.execute()
