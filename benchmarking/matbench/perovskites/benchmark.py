"""Benchmarking script for the matbench phonons dataset."""
from ..matbench import MatbenchTrainer

if __name__ == "__main__":
    trainer = MatbenchTrainer(
        "matbench_perovskites",
        "https://ml.materialsproject.org/projects/matbench_perovskites.json.gz",
        "e_form",
    )
    trainer.execute()
