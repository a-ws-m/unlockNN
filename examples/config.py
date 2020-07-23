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
