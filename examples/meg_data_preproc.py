"""Convert MEGNet pre-trained model data to layer outputs for a given model."""
from megnet.utils.models import load_model
from pyarrow import feather
from sklearn.model_selection import train_test_split

from sse_gnn.datalib.preprocessing import LayerScaler, convert_graph_df
from sse_gnn.utilities import serialize_array

from .config import DB_DIR


TRAIN_OUT = DB_DIR / "train_megnet_dat.fthr"
TEST_OUT = DB_DIR / "test_megnet_dat.fthr"


def split_data(data):
    """Split data into training and validation sets."""
    return train_test_split(data, random_state=2020)


model = load_model("Bandgap_MP_2018")

data = feather.read_feather(DB_DIR / "parsed_megnet_dat.fthr")

train, test = split_data(data)

scaler = LayerScaler(model, training_df=train, use_structs=False)

with open(DB_DIR / "github_data_sf", "wb") as f:
    scaler.sf.tofile(f)

train = scaler.training_data
test["layer_out"] = scaler.graphs_to_input(convert_graph_df(test))

train.loc[:, "layer_out"] = train["layer_out"].apply(serialize_array)
test.loc[:, "layer_out"] = test["layer_out"].apply(serialize_array)

feather.write_feather(train, TRAIN_OUT)
feather.write_feather(test, TEST_OUT)
