"""Load dataframes and calculate concatenation layer outputs."""
import numpy as np
import pyarrow.feather as feather
import pymatgen
from megnet import MEGNetModel

from sse_gnn.datalib.preprocessing import LayerScaler

from .config import DB_DIR, MODELS_DIR

MIN_LOSS_FILE = MODELS_DIR / "megnet" / "*.hdf5"

# * Calculate concatenation layers for all the SSE data and save to disk
print("Loading data...")
train_df = feather.read_feather(DB_DIR / "train_df.fthr")
test_df = feather.read_feather(DB_DIR / "test_df.fthr")

print("Loading MEGNet model...")
model = MEGNetModel.from_file(MIN_LOSS_FILE)

print("Instantiating GP data parser...")
processor = LayerScaler(model, training_df=train_df)

# Save scaling factor to file
np.savetxt("megnet_model/sf.txt", processor.sf)

# Save training data (with calculated outputs) to file
assert processor.training_data is not None
feather.write_feather(processor.training_data, DB_DIR / "gp_train_df.fthr")

# Compute test data values
print("Computing test data layer outputs...")
test_structures = [
    pymatgen.Structure.from_str(struct, "json") for struct in test_df["structure"]
]
test_df["layer_out"] = processor.structures_to_input(test_structures)

# And save to file
feather.write_feather(test_df, DB_DIR / "gp_test_df.fthr")
