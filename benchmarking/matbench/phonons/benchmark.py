"""Benchmarking script for the matbench phonons dataset."""
from pathlib import Path

from .. import matbench_utils

if __name__ == "__main__":
    trainer = matbench_utils.MatbenchTrainer(
        "matbench_phonons",
        "https://ml.materialsproject.org/projects/matbench_phonons.json.gz",
        "last phdos peak",
        batch_size=8,
    )
    trainer.execute()
