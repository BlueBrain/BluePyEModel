"""
Copyright 2023-2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest

from bluepyemodel.tools.mechanisms import get_mechanism_currents
from bluepyemodel.tools.utils import are_same_protocol
from bluepyemodel.tools.utils import format_protocol_name_to_list
from bluepyemodel.tools.utils import select_rec_for_thumbnail
from tests.utils import DATA


def test_get_mechanism_currents():
    # writes ion current
    ion_currs, nonspec_currs, ionic_concentrations = get_mechanism_currents(
        DATA / "mechanisms" / "Ca_HVA2.mod"
    )
    assert ion_currs == ["ica"]
    assert nonspec_currs == []
    assert ionic_concentrations == ["cai"]
    # writes ionic concentration
    ion_currs, nonspec_currs, ionic_concentrations = get_mechanism_currents(
        DATA / "mechanisms" / "CaDynamics_DC0.mod"
    )
    assert ion_currs == []
    assert nonspec_currs == []
    assert ionic_concentrations == ["cai"]
    # writes non-specific current
    ion_currs, nonspec_currs, ionic_concentrations = get_mechanism_currents(
        DATA / "mechanisms" / "Ih.mod"
    )
    assert ion_currs == []
    assert nonspec_currs == ["ihcn"]
    assert ionic_concentrations == []


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
    # None case
    assert not are_same_protocol("APWaveform_140", None)
    assert not are_same_protocol(None, "APWaveform_140")

    # str case
    assert not are_same_protocol("APWaveform_140", "IDRest_100")
    assert not are_same_protocol("APWaveform_140", "APWaveform_120")
    assert not are_same_protocol("APWaveform_140", "IDRest_140")
    assert are_same_protocol("APWaveform_140", "APWaveform_140")
    assert are_same_protocol("APWaveform_140", "APWaveform_140.0")

    # list case
    assert not are_same_protocol(["APWaveform", 140], ["APWaveform", 120])
    assert not are_same_protocol(["APWaveform", 140], ["IDRest", 140])
    assert are_same_protocol(["APWaveform", 140], ["APWaveform", 140])
    assert are_same_protocol(["APWaveform", 140], ["APWaveform", 140.0])

    # mixed case
    assert not are_same_protocol("APWaveform_140", ["APWaveform", 120])
    assert are_same_protocol("APWaveform_140", ["APWaveform", 140])
    assert are_same_protocol("APWaveform_140", ["APWaveform", 140.0])
    assert are_same_protocol("APWaveform_140.0", ["APWaveform", 140])


def test_select_rec_for_thumbnail():
    """Test for select_rec_for_thumbnail function."""
    # normal case: select step protocol with lowest positive amplitude
    rec_names = [
        "IDrest_200.soma.v",
        "IDrest_-40.soma.v",
        "IDrest_130.soma.v",
        "sAHP_40.soma.v",
    ]
    assert select_rec_for_thumbnail(rec_names) == "IDrest_130.soma.v"

    # empty recordings case
    with pytest.raises(
        ValueError, match="No recording in recording_names. Can not plot thumbnail."
    ):
        select_rec_for_thumbnail([])

    # no step protocols case: return 1st protocol
    assert select_rec_for_thumbnail(["sAHP_40.soma.v", "sAHP_20.soma.v"]) == "sAHP_40.soma.v"

    # step protocol present but not positive case: return 1st protocol
    assert select_rec_for_thumbnail(["sAHP_40.soma.v", "IDrest_-20.soma.v"]) == "sAHP_40.soma.v"

    # additional step protocol names case
    rec_names = ["IDrest_200.soma.v", "MyStep_100.soma.v"]
    other_step_prot_names = ["MyStep", "Test"]
    assert (
        select_rec_for_thumbnail(rec_names, additional_step_prots=other_step_prot_names)
        == "MyStep_100.soma.v"
    )

    # thumbnail_rec case
    rec_names = ["IDrest_130.soma.v", "sAHP_40.soma.v"]
    assert select_rec_for_thumbnail(rec_names, thumbnail_rec="sAHP_40.soma.v") == "sAHP_40.soma.v"

    # thumbnail_rec not in rec_names:
    rec_names = ["sAHP_40.soma.v", "IDrest_130.soma.v"]
    assert (
        select_rec_for_thumbnail(rec_names, thumbnail_rec="sAHP_20.soma.v") == "IDrest_130.soma.v"
    )
