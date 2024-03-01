"""
Copyright 2023, EPFL/Blue Brain Project

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


from bluepyemodel.emodel_pipeline.plotting import get_ordered_currentscape_keys, get_recording_names
from bluepyemodel.evaluation.evaluator import define_threshold_based_optimisation_protocol


def test_get_ordered_currentscape_keys():
    keys = ["RMPProtocol.soma.v", "Step_300.soma.cai", "Step_300.soma.ica_TC_iL", "Step_300.soma.v"]
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


def test_recording_names(db):
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
        'Step_280.soma.v',
        'bAP.ca_ais.cai',
        'bAP.ca_prox_basal.cai',
        'Step_200.soma.v',
        'IV_-100.soma.v',
        'SpikeRec_600.soma.v',
        'RMPProtocol.soma.v',
        'bAP.ca_soma.cai',
        'APWaveform_320.soma.v',
        'bAP.dend2.v',
        'bAP.soma.v',
        'SearchThresholdCurrent.soma.v',
        'bAP.dend1.v',
        'bAP.ca_prox_apic.cai',
        'SearchHoldingCurrent.soma.v',
        'Step_150.soma.v',
        'RinProtocol.soma.v'
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
