import pathlib
import os
from pathlib import Path
import pytest
import logging

import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.access_point import get_access_point

TEST_ROOT = Path(__file__).parents[1]
DATA = TEST_ROOT / "test_data"


@pytest.fixture
def api_config():
    return {
        "emodel": "cADpyr_L5TPC",
        "emodel_dir": DATA,
        "recipes_path": DATA / "config/recipes.json",
    }


@pytest.fixture
def db(api_config):
    return get_access_point("local", **api_config)


@pytest.fixture
def evaluator(db):

    if pathlib.Path("x86_64").is_dir():
        os.popen("rm -rf x86_64").read()
    os.popen(f"nrnivmodl {DATA}/mechanisms").read()

    db.get_mechanisms_directory = lambda: None
    return get_evaluator_from_access_point(access_point=db)


def ___test_protocols(db, evaluator):

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    params = db.get_emodel().parameters

    responses = evaluator.run_protocols(
        protocols=evaluator.fitness_protocols.values(), param_values=params
    )

    assert_allclose(responses["bpo_rin"], 37.372735, rtol=1e-06)
    assert_allclose(responses["bpo_rmp"], -77.232155, rtol=1e-06)
    assert_allclose(responses["bpo_holding_current"], -0.14453125, rtol=1e-06)
    assert_allclose(responses["bpo_threshold_current"], 0.482622, rtol=1e-06)

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
        #responses[prot_name].response.to_csv(f"{DATA}/test_{prot_name}.csv", index=False)
        expected_df = pd.read_csv(f"{DATA}/test_{prot_name}.csv")
        response = responses[prot_name].response
        assert_frame_equal(response, expected_df)

def test_reproducible_protocols(db, evaluator):

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    params = db.get_emodel().parameters
    print(params)

    responses = evaluator.run_protocols(
        protocols=evaluator.fitness_protocols.values(), param_values=params
    )
    scores = evaluator.objective_list(evaluator.fitness_calculator.calculate_scores(responses))
    print(scores, sum(scores))#responses["bpo_rin"], responses['bpo_holding_current'])
    for p in params:
        if p != 'gK_Pstbar_K_Pst.axonal':
            params[p] *= 1.01
    responses = evaluator.run_protocols(
        protocols=evaluator.fitness_protocols.values(), param_values=params
    )
    scores = evaluator.objective_list(evaluator.fitness_calculator.calculate_scores(responses))
    print(scores, sum(scores))#responses["bpo_rin"], responses['bpo_holding_current'])

    for p in params:
        if p != 'gK_Pstbar_K_Pst.axonal':
            params[p] /= 1.01
    responses = evaluator.run_protocols(
        protocols=evaluator.fitness_protocols.values(), param_values=params
    )
    scores = evaluator.objective_list(evaluator.fitness_calculator.calculate_scores(responses))
    print(scores, sum(scores))#responses["bpo_rin"], responses['bpo_holding_current'])


    #assert_allclose(responses["bpo_rin"], 37.372735, rtol=1e-06)
    #assert_allclose(responses["bpo_rmp"], -77.232155, rtol=1e-06)
    #assert_allclose(responses["bpo_holding_current"], -0.14453125, rtol=1e-06)
    #assert_allclose(responses["bpo_threshold_current"], 0.482622, rtol=1e-06)


