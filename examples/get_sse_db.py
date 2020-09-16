"""Example code demonstrating how to build up a database containing structures and SSEs."""
import pyarrow.feather as feather

from unlockgnn.datalib.mining import (
    add_smact_structs,
    download_structures,
    extract_sse_data,
)

from .config import DB_DOWN_LOC, DB_SMACT_LOC, LAST_TO_FIRST, MP_API_KEY, SSE_DB_LOC

working_db = None  # Which, if any, is the most complete database in the filesystem
for DB in LAST_TO_FIRST:
    try:
        df = feather.read_feather(DB)

        if not df.empty:  # Got a working DB
            working_db = DB
            print(f"Found already processed database: {working_db}")
            break

    except Exception as e:  # DB doesn't exist yet
        print(f"Can't import {DB}: {e}")
        continue

if working_db is None:
    print("Downloading structures.")
    assert MP_API_KEY is not None
    df = download_structures(MP_API_KEY, DB_DOWN_LOC)
    working_db = DB_DOWN_LOC

if working_db == DB_DOWN_LOC:
    print("Converting to SMACT structures...")
    df = add_smact_structs(df, DB_SMACT_LOC)
    working_db = DB_SMACT_LOC

if working_db == DB_SMACT_LOC:
    print("Adding SSE columns...")
    df = extract_sse_data(df, SSE_DB_LOC)
    working_db = SSE_DB_LOC

if working_db == SSE_DB_LOC:
    print(f"Database found/created in {working_db}")
