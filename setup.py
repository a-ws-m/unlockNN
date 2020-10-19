"""Setup script for package."""
import pathlib
from setuptools import find_namespace_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="unlockGNN",
    version="1.0.0",
    description="Uncertainty quantification for graph neural network models of chemical systems",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/a-ws-m/unlockGNN",
    author="Alexander Moriarty",
    author_email="amoriarty14@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_namespace_packages(include=["unlockgnn.*"]),
    include_package_data=False,
    install_requires=[
        "cloudpickle",
        "numpy",
        "pandas",
        "pyarrow",
        "pymatgen",
        "matminer",
        "megnet",
        "scipy",
        "smact",
        "tensorflow-probability",
    ],
    extras_require={
        "choice of tensorflow install": ["tensorflow"],
        "UQ metrics": ["matplotlib", "seaborn"],
        "progress bar": ["tqdm"],
    },
)
