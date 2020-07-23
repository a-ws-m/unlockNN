"""Tools to scrape relevant compound data from Materials Project and label their ions' SSEs appropriately."""
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import pyarrow.feather as feather
import pymatgen
import smact.data_loader as smact_data
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from smact.structure_prediction.structure import SmactStructure

from .config import (
    DB_DOWN_LOC,
    DB_SMACT_LOC,
    LAST_TO_FIRST,
    MP_API_KEY,
    MP_API_VAR,
    SSE_DB_LOC,
)

if not MP_API_KEY:
    raise TypeError(
        f"Environment variable {MP_API_VAR} not set. "
        "Please set it to your Materials Project API key."
    )


def download_structures(file: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Create database of MP IDs and structures.

    Queries for binary compounds with experimentally determined crystal structures
    that have been reported to the ICSD.
    Optionally, writes a database of their `material_id`s and their structures as a feather
    to the specified file.

    Args:
        file (str, optional): The path to the file, to which to write the feathered data.
            If omitted (None), does not write to a file.

    Returns:
        :obj:`pd.DataFrame`: A dataframe containting the materials IDs and structures
            of the compounds.

    """
    df_mp = MPDataRetrieval(MP_API_KEY).get_dataframe(
        criteria={"icsd_ids.0": {"$exists": True}, "nelements": 2},
        properties=["material_id", "structure"],
        index_mpid=False,  # Index isn't preserved when writing to file
    )

    df_mp["structure"] = [structure.to("json") for structure in df_mp["structure"]]
    if file is not None:
        feather.write_feather(df_mp, file)

    return df_mp


def get_smact_struct(py_struct) -> Union[SmactStructure, None]:
    """Get a SMACT structure from a pymatgen structure and handle errors.

    If bond valencey can't be determined, returns `None`.

    Args:
        py_struct (:obj:`pymatgen.Structure`): The structure to convert.

    Returns:
        :obj:`SmactStructure`, optional: The SmactStructure object, or `None`
            if the bond valency could not be determined.

    """
    try:
        return SmactStructure.from_py_struct(py_struct)
    except ValueError:
        return None


def add_smact_structs(
    df: pd.DataFrame, file: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Add species columns to a DataFrame containing pymatgen `Structure`s.

    Species column is entitled 'species'. Removes rows that cannot be
    converted due to ambiguous bond valency. Optionally outputs the new DataFrame to a file.

    Args:
        df (:obj:`pd.DataFrame`): A DataFrame containing pymatgen structures in a
            column entitled 'structure'.
        file (str, optional): The path to a file, to which to write the feathered data.
            If omitted (None), does not write to a file.

    Returns:
        df (:obj:`pd.DataFrame`): The DataFrame with added species columns.

    """
    structures = (
        pymatgen.Structure.from_str(struct, "json") for struct in df["structure"]
    )
    with Pool() as pool:
        df["smact_struct"] = pool.map(get_smact_struct, structures)

    df.dropna(inplace=True)
    df["smact_struct"] = [struct.as_poscar() for struct in df["smact_struct"]]

    if file is not None:
        feather.write_feather(df, file)

    return df


def lookup_sse(symbol: str, charge: int) -> Optional[float]:
    """Lookup the SSE of an ion, given its symbol and charge.

    Args:
        symbol (str): The elemental symbol of the species.
        charge (int): The oxidation state of the ion.

    Returns:
        SSE (float or None): The SSE of the ion, or `None` if it
            is not in the SMACT database.

    """
    data_2015 = smact_data.lookup_element_sse2015_data(symbol, copy=False)
    try:
        for data in data_2015:
            if data["OxidationState"] == charge:
                return data["SolidStateEnergy2015"]
    except TypeError:
        pass  # Got `None` from sse2015 lookup

    data_pauli = smact_data.lookup_element_sse_pauling_data(symbol)
    try:
        for data in data_pauli:
            if data["OxidationState"] == charge:
                return data["SolidStateEnergy2015"]
    except TypeError:
        pass  # Got `None` from sse_pauli lookup

    return None


def get_cat_an_sse(
    smact_struct: SmactStructure,
) -> Tuple[Optional[float], Optional[float]]:
    """Return the cation and anion SSEs of ions in a binary compound.

    If there are multiple oxidation states per element, this function
    returns `None` for both cation and anion SSE.

    Args:
        smact_struct (:obj:`SmactStructure`): The `SmactStructure` of the compound.

    Returns:
        cation_sse (float): The SSE of the cation, or `None` if compound
            is not binary.
        anion_sse (float): The SSE of the anion, or `None` if compound
            is not binary.

    """
    if len(smact_struct.species) != 2:
        # More than two combinations of element and charge exist in the compound
        return (None, None)

    # Determine anion and cation by sorting on charge
    anion, cation = tuple(sorted(smact_struct.species, key=itemgetter(1)))
    anion_sse = lookup_sse(anion[0], anion[1])
    cation_sse = lookup_sse(cation[0], cation[1])
    return cation_sse, anion_sse


def extract_sse_data(
    df: pd.DataFrame, file: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Add columns for SSEs to a DataFrame containing `SmactStructure`s.

    Cation SSE contained in 'cat_sse', anion sse contained in 'an_sse'.
    Optionally outputs the new DataFrame to a file.

    Args:
        df (:obj:`pd.DataFrame`): A DataFrame containing `SmactStructure`s in a
            column entitled 'smact_struct'.
        file (str, optional): The path to a file, to which to write the feathered data.
            If omitted (None), does not write to a file.

    Returns:
        df (:obj:`pd.DataFrame`): The DataFrame with added `SmactStructure`s.

    """
    smact_structs = map(SmactStructure.from_poscar, df["smact_struct"])
    cat_an_sses = list(map(get_cat_an_sse, smact_structs))

    df["cat_sse"] = list(map(itemgetter(0), cat_an_sses))
    df["an_sse"] = list(map(itemgetter(1), cat_an_sses))

    df.dropna(inplace=True)

    if file is not None:
        feather.write_feather(df, file)

    return df


if __name__ == "__main__":
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
        df = download_structures(DB_DOWN_LOC)
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
