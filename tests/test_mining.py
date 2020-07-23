"""Test the data mining suite."""
import pytest

from sse_gnn.datalib import mining

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
