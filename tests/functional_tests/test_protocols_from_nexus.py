"""
Copyright 2024, EPFL/Blue Brain Project

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


def test_protocols(db_from_nexus, tmp_path):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    params = db_from_nexus.get_emodel().parameters
    evaluator = get_evaluator_from_access_point(access_point=db_from_nexus)

    responses = evaluator.run_protocols(
        protocols=evaluator.fitness_protocols.values(), param_values=params
    )

    assert_allclose(responses["bpo_rmp"], -82.61402706564716, rtol=1e-06)
    assert_allclose(responses["bpo_holding_current"], -0.05, rtol=1e-06)
    assert_allclose(responses["bpo_rin"], 41.151498, rtol=1e-06)
    assert_allclose(responses["bpo_threshold_current"], 0.3755498847141163, rtol=1e-06)

    for prot_name in [
        "APWaveform_280.soma.v",
        "IDrest_150.soma.v",
        "IDrest_250.soma.v",
        "IV_-100.soma.v",
        "RinProtocol.soma.v",
        "RMPProtocol.soma.v",
        "SearchHoldingCurrent.soma.v",
        "SearchThresholdCurrent.soma.v",
    ]:
        output_path = f"{tmp_path}/test_{prot_name}.csv"
        responses[prot_name].response.to_csv(output_path, index=False)
        expected_df = pd.read_csv(output_path)
        response = responses[prot_name].response
        assert_frame_equal(response, expected_df)
