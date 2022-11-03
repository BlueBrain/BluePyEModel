from pathlib import Path

import pytest

from bluepyemodel.tools.mechanisms import get_mechanism_currents
from bluepyemodel.tools.utils import are_same_protocol
from bluepyemodel.tools.utils import format_protocol_name_to_list

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


def test_format_protocol_name_to_list():
    # str case
    name, amp = format_protocol_name_to_list("APWaveform_140")
    assert name == "APWaveform"
    assert amp == 140.0

    name, amp = format_protocol_name_to_list("APWaveform_140.0")
    assert name == "APWaveform"
    assert amp == 140.0

    name, amp = format_protocol_name_to_list("APWaveform")
    assert name == "APWaveform"
    assert amp is None

    # list case
    name, amp = format_protocol_name_to_list(["APWaveform", 140])
    assert name == "APWaveform"
    assert amp == 140.0

    # error case
    with pytest.raises(TypeError, match="protocol_name should be a string or a list."):
        format_protocol_name_to_list(None)


def test_are_same_protocol():

