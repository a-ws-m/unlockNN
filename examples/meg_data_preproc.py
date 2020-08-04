"""Convert MEGNet pre-trained model data to layer outputs for a given model."""
from megnet.utils.models import load_model
from pyarrow import feather

from sse_gnn.utils import serialize_array, deserialize_array

from .config import DB_DIR
from sse_gnn.datalib.preprocessing import LayerScaler

model = load_model("Bandgap_MP_2018")

data = feather.read_feather(DB_DIR / "parsed_megnet_dat.fthr")

# Deserialize arrays
for col in array_cols:
    data[col] = data[col].apply(deserialize_array)

scaler = LayerScaler(model, training_df=data,)

