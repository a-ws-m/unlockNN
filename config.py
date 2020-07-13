"""Configuration file containing globally useful variables."""
from pathlib import Path
import os

DB_DIR = Path("dataframes")
DB_DOWN_LOC = DB_DIR / "mp_download.pickle"  # Download location for raw MP data
DB_SMACT_LOC = DB_DIR / "smact_db.pickle"  # Location for data with SmactStructures
SSE_DB_LOC = DB_DIR / "sse_db.pickle"  # Location for completely preprocessed data

LAST_TO_FIRST = [
    SSE_DB_LOC,
    DB_SMACT_LOC,
    DB_DOWN_LOC,
]  # List of files in reverse processing order

# Environment variable that contains the Materials Project API key
MP_API_VAR = "MPI_KEY"

# Get the Materials Project API
MP_API_KEY = os.environ.get(MP_API_VAR)
