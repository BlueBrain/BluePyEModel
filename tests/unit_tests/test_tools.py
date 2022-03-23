from pathlib import Path

from bluepyemodel.tools.mechanisms import get_mechanism_currents

TEST_ROOT = Path(__file__).parents[1]
DATA = TEST_ROOT / "test_data"


def test_get_mechanism_currents():
    # writes ion current
    ion_currs, nonspec_currs = get_mechanism_currents(DATA / "mechanisms" / "Ca_HVA2.mod")
    assert ion_currs == ["ica"]
    assert nonspec_currs == []
    # writes ionic concentration
    ion_currs, nonspec_currs = get_mechanism_currents(DATA / "mechanisms" / "CaDynamics_DC0.mod")
    assert ion_currs == []
    assert nonspec_currs == []
    # writes non-specific current
    ion_currs, nonspec_currs = get_mechanism_currents(DATA / "mechanisms" / "Ih.mod")
    assert ion_currs == []
    assert nonspec_currs == ["ihcn"]
