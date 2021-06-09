import pathlib
import os
from pathlib import Path
import pytest

import pandas as pd
from numpy.testing import assert_allclose
from pandas.util.testing import assert_frame_equal

from bluepyemodel.evaluation.evaluation import get_evaluator_from_db
from bluepyemodel.api import get_db

TEST_ROOT = Path(__file__).parent
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
    return get_db("singlecell", **api_config)


@pytest.fixture
def evaluator(db):
    if pathlib.Path("x86_64").is_dir():
        os.popen("rm -rf x86_64").read()
    os.popen(f"nrnivmodl {DATA}/mechanisms").read()
    return get_evaluator_from_db(emodel=db.emodel, db=db)


def test_protocols(db, evaluator):
    params = db.get_emodel()["parameters"]
    responses = evaluator.run_protocols(
        protocols=evaluator.fitness_protocols.values(), param_values=params
    )

    assert_allclose(responses["bpo_rin"], 37.372735)
    assert_allclose(responses["bpo_rmp"], -77.23215465876982)
    assert_allclose(responses["bpo_holding_current"], -0.14453125)
    assert_allclose(responses["bpo_threshold_current"], 0.482622125935)

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
        responses[prot_name].response.to_csv(f"{DATA}/test_{prot_name}.csv", index=False)
        expected_df = pd.read_csv(f"{DATA}/test_{prot_name}.csv")
        response = responses[prot_name].response
        assert_frame_equal(response, expected_df)
