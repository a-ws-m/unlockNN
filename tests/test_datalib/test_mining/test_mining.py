"""Test the data mining suite."""
import pandas as pd
import pymatgen as mg
import pytest

from smact.structure_prediction.structure import SmactStructure

from unlockgnn.datalib import mining

sse_lookup_combos = [
    (("Ag", 1), -4.1),
    (("C", -4), -6.48),
    # Pauli only
    (("H", 1), -4.28),
    (("H", -1), -4.28),
    (("H", 1, False), None),
    # Neither DB
    (("spam", 42), None),
]


@pytest.mark.parametrize("test_input,expected", sse_lookup_combos)
def test_lookup_sse(test_input, expected):
    """Test `lookup_sse` functionality."""
    assert mining.lookup_sse(*test_input) == expected


cubic_lattice = mg.Lattice.cubic(4.2)
cscl = mg.Structure(cubic_lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
nonsense_struct = mg.Structure(cubic_lattice, ["H", "He"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_get_smact_struct():
    """Test `get_smact_struct` functionality."""
    assert isinstance(mining.get_smact_struct(cscl), SmactStructure)
    assert mining.get_smact_struct(nonsense_struct) is None


# * Database testing schema


def test_add_smact_structs():
    """Test adding SMACT structures to a dataframe."""
    test_df = pd.DataFrame({"structure": [cscl.to_json(), nonsense_struct.to_json()]})
    test_df = mining.add_smact_structs(test_df)

    assert set(test_df.columns) == {"structure", "smact_struct"}

    # Only one structure should compile and we should have
    # dropped the invalid one
    assert len(test_df.index) == 1

    # Should be a POSCAR representation
    assert isinstance(test_df["smact_struct"][0], str)


cs_sse = 0.6
cl_sse = -8.61
cs_cl_sses = (cs_sse, cl_sse)
cs_cl_species = [("Cs", 1, 1), ("Cl", -1, 1)]


def test_get_cat_an_sse(mocker):
    """Test getting SSEs using different mock `SmactStructure`s."""
    m = mocker.Mock()

    # First, with pre-sorted species
    m.species = cs_cl_species
    assert mining.get_cat_an_sse(m) == cs_cl_sses

    # Unsorted
    m.species = m.species[::-1]
    assert mining.get_cat_an_sse(m) == cs_cl_sses

    # Improper number of SSEs
    m.species.append(("spam", 1, 1))
    assert mining.get_cat_an_sse(m) == (None, None)


def test_sse_extract(mocker):
    """Test extracting SSE data into a `DataFrame`."""
    # Make a mock SmactStructure
    cscl_mock = mocker.Mock()
    cscl_mock.species = cs_cl_species

    # Make a mock SmactStructure with invalid species
    inval_mock = mocker.Mock()
    inval_mock.species = [("African", 1, 1), ("European", 1, 1)]

    cscl_label = "CsCl"
    inval_label = "swallow"
    test_df = pd.DataFrame({"smact_struct": [cscl_label, inval_label]})

    def get_smact_mock(poscar_label):
        """Get the `SmactStructure` `Mock` instance."""
        if poscar_label == cscl_label:
            return cscl_mock
        elif poscar_label == inval_label:
            return inval_mock

    def get_sse_mock(structure):
        """Mimic getting cation and anion SSEs.

        Operates with expected behaviour for `get_cat_an_sse`.

        """
        if set(structure.species) == set(cs_cl_species):
            return cs_cl_sses
        else:
            return (None, None)

    # Patch constructor to give us the mock
    mocker.patch(
        "smact.structure_prediction.structure.SmactStructure.from_poscar",
        get_smact_mock,
    )
    # Patch sse calculator
    mocker.patch("unlockgnn.datalib.mining.get_cat_an_sse", get_sse_mock)

    test_df = mining.extract_sse_data(test_df)

    assert set(test_df.columns) == {"smact_struct", "cat_sse", "an_sse"}
    assert set(test_df["cat_sse"]) == {cs_sse}
    assert set(test_df["an_sse"]) == {cl_sse}
