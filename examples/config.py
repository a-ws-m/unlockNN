"""Useful persistent variables for examples."""
from pathlib import Path
import os

DB_DIR = Path("dataframes/")
DB_DOWN_LOC = DB_DIR / "mp_download.fthr"  # Download location for raw MP data
DB_SMACT_LOC = DB_DIR / "smact_db.fthr"  # Location for data with SmactStructures
SSE_DB_LOC = DB_DIR / "sse_db.fthr"  # Location for completely preprocessed data
MODELS_DIR = Path("models/")

LAST_TO_FIRST = [
    SSE_DB_LOC,
    DB_SMACT_LOC,
    DB_DOWN_LOC,
]  # List of files in reverse processing order

# Environment variable that contains the Materials Project API key
MP_API_VAR = "MPI_KEY"

# Get the Materials Project API
MP_API_KEY = os.environ.get(MP_API_VAR)

N_EPOCHS = os.environ["N_EPOCHS"]  # Number of training epochs
PATIENCE = os.environ["PATIENCE"]  # Patience
PREV_MODEL = os.environ.get("PREV_MODEL")  # Location of the previous model
NEW_MODEL = os.environ["NEW_MODEL"]  # Where to save the new model
CKPT_PATH = os.environ.get("CKPT_PATH")  # Checkpoint path
n_inducing = os.environ.get("NUM_INDUCING")  # How many inducing points for the VGP
NUM_INDUCING = int(n_inducing) if n_inducing is not None else None
RUN_NAME = os.environ.get("RUN_NAME")
