"""A test module for dissecting MEGNet models."""
from pathlib import Path

import pandas as pd

from tensorflow.keras import backend as K
from megnet.models import MEGNetModel

CONCAT_IND = -4  # Concatenate layer index

# Model file locations
model_dir = Path("megnet_model/")
model_name = "val_mae_00996_0.014805.hdf5"
model_loc = str(model_dir / model_name)

# Data file locations
data_dir = Path("dataframes/")
train_data_loc = str(data_dir / "train_df.pickle")
test_data_loc = str(data_dir / "test_df.pickle")

# Load model
model = MEGNetModel.from_file(model_loc)

# Load data
# train_df = pd.read_pickle(train_data_loc)
test_df = pd.read_pickle(test_data_loc)

# Get the concatenation layer
concat_layer = model.layers[CONCAT_IND]
concat_out = concat_layer.output  # Its output

print(type(concat_out))

# Define a function to acquire the output of the concat layer
concat_eval = K.function([model.input], [concat_out])


def convert_struct_to_inp(struct):
    """Convert a pymatgen structure to an input for the model."""
    graph = model.graph_converter.convert(struct)
    return model.graph_converter.graph_to_input(graph)


# Let's see what the output looks like for an example case...
test_case = test_df.sample()
test_struct = test_df["structure"][0]
concat_example = concat_eval([convert_struct_to_inp(test_struct)])

print(concat_example[0])
