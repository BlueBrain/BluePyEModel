from pathlib import Path

from bluepyemodel.tools.mechanisms import get_mechanism_ion

TEST_ROOT = Path(__file__).parents[1]
DATA = TEST_ROOT / "test_data"


def test_get_mechanism_ion():
    # writes ion current
    ions = get_mechanism_ion(DATA / "mechanisms" / "Ca_HVA2.mod")
    assert ions == ["ica"]
    # writes ionic concentration
    ions = get_mechanism_ion(DATA / "mechanisms" / "CaDynamics_DC0.mod")
    assert ions == []
