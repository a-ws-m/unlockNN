"""Tools for adding uncertainty quantification to neural networks."""
import warnings

# Modules with deprecation warnings
DEPR_MODULES = ["tensorflow", "tensorflow_probability", "megnet"]
PENDING_DEPR_MODULES = ["ruamel", "pymatgen", "monty"]

for module in DEPR_MODULES:
    warnings.filterwarnings("ignore", "DeprecationWarning", module=module)

for module in PENDING_DEPR_MODULES:
    warnings.filterwarnings("ignore", "PendingDeprecationWarning", module=module)

warnings.filterwarnings("ignore", r"UserWarning: Unable to detect statically whether the number of index_points is 1.")
warnings.filterwarnings("ignore", r"UserWarning: Converting sparse .* to a dense Tensor of unknown shape.")
warnings.filterwarnings("ignore", r"CustomMaskWarning: Custom mask layers require a config and must override get_config.")

from .kernel_layers import *
from .model import *
