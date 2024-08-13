"""Tests for plotting utils functions."""

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

import copy
import numpy

from bluepyemodel.ecode.iv import IV
from bluepyemodel.emodel_pipeline.plotting_utils import (
    binning,
    fill_in_IV_curve_evaluator,
    get_ordered_currentscape_keys,
    get_recording_names,
    get_title,
    get_traces_ylabel,
    get_traces_names_and_float_responses,
    rel_to_abs_amplitude,
)
from bluepyemodel.evaluation.efel_feature_bpem import eFELFeatureBPEM
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.evaluation.evaluator import (
    define_threshold_based_optimisation_protocol,
    soma_loc,
)
from bluepyemodel.evaluation.protocols import ThresholdBasedProtocol
from bluepyemodel.evaluation.recordings import LooseDtRecordingCustom
from bluepyopt.ephys.objectives import SingletonWeightObjective
from bluepyopt.ephys.responses import TimeVoltageResponse


def test_get_traces_ylabel():
    """Test get_traces_ylabel."""
    assert get_traces_ylabel("v") == "Voltage (mV)"
    assert get_traces_ylabel("i") == "Current (pA)"
    assert get_traces_ylabel("ica") == "Current (pA)"
    assert get_traces_ylabel("cai") == "Ionic concentration (mM)"
    assert get_traces_ylabel("unknown") == ""


def test_recording_names(db):
    """Test get_recording_names."""
    fitness_calculator_configuration = db.get_fitness_calculator_configuration(
        record_ions_and_currents=True
    )
    prot_runner, _ = define_threshold_based_optimisation_protocol(fitness_calculator_configuration)
    stimuli = prot_runner.protocols
    recording_names = get_recording_names(
        db.get_fitness_calculator_configuration().protocols,
        stimuli,
    )

    assert recording_names == {
        "Step_280.soma.v",
        "bAP.ca_ais.cai",
        "bAP.ca_prox_basal.cai",
        "Step_200.soma.v",
        "IV_-100.soma.v",
        "SpikeRec_600.soma.v",
        "RMPProtocol.soma.v",
        "bAP.ca_soma.cai",
        "APWaveform_320.soma.v",
        "bAP.dend2.v",
        "bAP.soma.v",
        "SearchThresholdCurrent.soma.v",
        "bAP.dend1.v",
        "bAP.ca_prox_apic.cai",
        "SearchHoldingCurrent.soma.v",
        "Step_150.soma.v",
        "RinProtocol.soma.v",
    }

    fitness_calculator_configuration = db.get_fitness_calculator_configuration(
        record_ions_and_currents=False
    )
    prot_runner, _ = define_threshold_based_optimisation_protocol(fitness_calculator_configuration)
    stimuli = prot_runner.protocols
    recording_names_no_extra_rec = get_recording_names(
        db.get_fitness_calculator_configuration().protocols,
        stimuli,
    )

    assert recording_names == recording_names_no_extra_rec


def test_get_traces_names_and_float_responses():
    """Test get_traces_names_and_float_responses."""
    # test with one absolute protocol
    recording_names = {
        "Step_0.1234.soma.v",
    }
    responses = {
        "Step_0.1234.soma.v": TimeVoltageResponse(
            "test_response", [0.0, 0.1, 0.2], [-80.0, -80.1, -79.9]
        )
    }

    test_resp = get_traces_names_and_float_responses(responses, recording_names)
    assert test_resp == (["Step_0.1234.soma.v"], None, None, None, None)

    # test with threshold-based protocols
    recording_names = {
        "Step_200.soma.v",
        "IV_-100.soma.v",
        "RMPProtocol.soma.v",
        "SearchThresholdCurrent.soma.v",
        "RinProtocol.soma.v",
        "SearchHoldingCurrent.soma.v",
    }
    responses = {
        "Step_200.soma.v": TimeVoltageResponse("", [], []),
        "IV_-100.soma.v": TimeVoltageResponse("", [], []),
        "RMPProtocol.soma.v": TimeVoltageResponse("", [], []),
        "SearchThresholdCurrent.soma.v": TimeVoltageResponse("", [], []),
        "RinProtocol.soma.v": TimeVoltageResponse("", [], []),
        "SearchHoldingCurrent.soma.v": TimeVoltageResponse("", [], []),
        "bpo_threshold_current": 0.2,
        "bpo_holding_current": -0.08,
        "bpo_rmp": -80.0,
        "bpo_rin": 2357.09,
    }

    test_resp = get_traces_names_and_float_responses(responses, recording_names)
    assert test_resp == (
        [
            "Step_200.soma.v",
            "IV_-100.soma.v",
            "RMPProtocol.soma.v",
            "SearchThresholdCurrent.soma.v",
            "RinProtocol.soma.v",
            "SearchHoldingCurrent.soma.v",
        ],
        0.2,
        -0.08,
        -80.0,
        2357.09,
    )


def test_get_title():
    """Test get_title."""
    assert get_title("bNAC", None, None) == "bNAC"
    assert get_title("bNAC", "test", None) == "bNAC ; iteration = test"
    assert get_title("bNAC", None, 42) == "bNAC ; seed = 42"
    assert get_title("bNAC", "test", 42) == "bNAC ; iteration = test ; seed = 42"


def test_rel_to_abs_amplitude():
    """Test rel_to_abs_amplitude."""
    # absolute amplitude case
    numpy.testing.assert_equal(rel_to_abs_amplitude(0.1234, {}), numpy.nan)

    # relative amplitude case
    assert (
        rel_to_abs_amplitude(150, {"bpo_threshold_current": 0.5, "bpo_holding_current": -0.2})
        == 0.55
    )


def test_binning():
    """Test binning."""
    # lists smaller than n_bin
    assert binning([0, 1], [2, 3]) == ([0, 1], [2, 3], [0, 0])

    # lists larger than n_bin
    x = [0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.6, 0.6, 0.7, 0.7, 0.7, 1.0]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    new_x, new_y, y_err = binning(x, y)
    numpy.testing.assert_allclose(new_x, [0.1, 0.3, 0.5, 0.7, 0.9])
    numpy.testing.assert_allclose(new_y, [3.5, 7.0, 11.0, 15.0, 17.0])
    numpy.testing.assert_allclose(y_err, [1.70782513, 2.0, 1.41421356, 0.816496581, 0.0])


def test_fill_in_IV_curve_evaluator(db):
    """Test fill_in_IV_curve_evaluator."""
    evaluator = get_evaluator_from_access_point(db)
    # copy evaluator
    new_evaluator = copy.deepcopy(evaluator)
    # add an iv with 0 < amp <100 to new evaluator
    new_protocol = ThresholdBasedProtocol(
        name="IV_40",
        stimulus=IV(
            location=soma_loc,
            thresh_perc=40,
        ),
        recordings=[LooseDtRecordingCustom("IV_40.soma.v", soma_loc, "v")],
    )
    new_evaluator.fitness_protocols["main_protocol"].protocols["IV_40"] = new_protocol
    new_evaluator.fitness_protocols["main_protocol"].execution_order = (
        new_evaluator.fitness_protocols["main_protocol"].compute_execution_order()
    )

    # should also append a random feature associated to IV_40 protocol. BPEM does not expect feature-less protocols
    feat_name = "IV_40.soma.v.voltage_base"
    new_evaluator.fitness_calculator.objectives.append(
        SingletonWeightObjective(
            feat_name,
            eFELFeatureBPEM(
                feat_name,
                efel_feature_name="voltage_base",
                recording_names={"": "IV_40.soma.v"},
                stim_start=700,
                stim_end=2700,
                exp_mean=-80.0,  # fodder: not used
                exp_std=1.0,  # fodder: not used
                stimulus_current=new_protocol.amplitude,
            ),
            1.0,
        )
    )

    # now evaluator has IV both for amp < 0 case and for 0 < amp < 100 case
    # get new evaluator from function
    updated_evaluator = fill_in_IV_curve_evaluator(new_evaluator, {}, "iv")

    # test that new features have appeared
    assert "IV_-100.soma.v.voltage_deflection_vb_ssse" not in [
        obj.name for obj in evaluator.fitness_calculator.objectives
    ]
    assert "IV_-100.soma.v.voltage_deflection_vb_ssse" not in [
        obj.features[0].name for obj in evaluator.fitness_calculator.objectives
    ]
    assert "IV_40.soma.v.maximum_voltage_from_voltagebase" not in [
        obj.name for obj in evaluator.fitness_calculator.objectives
    ]
    assert "IV_40.soma.v.maximum_voltage_from_voltagebase" not in [
        obj.features[0].name for obj in evaluator.fitness_calculator.objectives
    ]

    assert "IV_-100.soma.v.voltage_deflection_vb_ssse" in [
        obj.name for obj in updated_evaluator.fitness_calculator.objectives
    ]
    assert "IV_-100.soma.v.voltage_deflection_vb_ssse" in [
        obj.features[0].name for obj in updated_evaluator.fitness_calculator.objectives
    ]
    assert "IV_40.soma.v.maximum_voltage_from_voltagebase" in [
        obj.name for obj in updated_evaluator.fitness_calculator.objectives
    ]
    assert "IV_40.soma.v.maximum_voltage_from_voltagebase" in [
        obj.features[0].name for obj in updated_evaluator.fitness_calculator.objectives
    ]


def test_get_ordered_currentscape_keys():
    """Test get_ordered_currentscape_keys."""
    keys = [
        "RMPProtocol.soma.v",
        "Step_300.soma.cai",
        "Step_300.soma.ica_TC_iL",
        "Step_300.soma.v",
    ]
    expected_keys = {
        "Step_300": {
            "soma": {
                "voltage_key": "Step_300.soma.v",
                "current_keys": ["Step_300.soma.ica_TC_iL"],
                "current_names": ["ica_TC_iL"],
                "ion_conc_keys": ["Step_300.soma.cai"],
                "ion_conc_names": ["cai"],
            }
        }
    }
    ordered_keys = get_ordered_currentscape_keys(keys)
    assert ordered_keys == expected_keys
