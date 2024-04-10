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

import logging

import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point


def test_protocols(db, tmp_path):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    params = db.get_emodel().parameters
    evaluator = get_evaluator_from_access_point(access_point=db)

    responses = evaluator.run_protocols(
        protocols=evaluator.fitness_protocols.values(), param_values=params
    )

    assert_allclose(responses["bpo_rmp"], -77.232155, rtol=1e-06)
    assert_allclose(responses["bpo_holding_current"], -0.146875, rtol=1e-06)
    assert_allclose(responses["bpo_rin"], 37.32179555, rtol=1e-06)
    assert_allclose(responses["bpo_threshold_current"], 0.4765729735, rtol=1e-06)

    for prot_name in [
        "RMPProtocol.soma.v",
        "RinProtocol.soma.v",
        "bAP.soma.v",
        "bAP.dend1.v",
        "bAP.dend2.v",
        "bAP.ca_prox_apic.cai",
        "bAP.ca_prox_basal.cai",
        "bAP.ca_soma.cai",
        "bAP.ca_ais.cai",
        "Step_200.soma.v",
        "Step_280.soma.v",
        "APWaveform_320.soma.v",
        "IV_-100.soma.v",
        "SpikeRec_600.soma.v",
    ]:
        responses[prot_name].response.to_csv(f"{tmp_path}/test_{prot_name}.csv", index=False)
        expected_df = pd.read_csv(f"{tmp_path}/test_{prot_name}.csv")
        response = responses[prot_name].response
        assert_frame_equal(response, expected_df)
