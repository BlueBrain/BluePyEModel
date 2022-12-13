import pytest
import numpy

from bluepyopt.ephys.locations import NrnSeclistCompLocation
from bluepyopt.ephys.recordings import CompRecording

from bluepyemodel.ecode.idrest import IDrest
from bluepyemodel.evaluation.protocol_configuration import ProtocolConfiguration
from bluepyemodel.evaluation.efeature_configuration import EFeatureConfiguration
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.evaluation.evaluator import define_location, define_protocol, define_efeature
from bluepyemodel.evaluation.protocols import ThresholdBasedProtocol
from tests.unit_tests.test_local_access_point import api_config
from tests.unit_tests.test_local_access_point import db

def test_define_location():

    definition = {
        "type": "CompRecording",
        "location": "soma"
    }

    location = define_location(definition)

    assert isinstance(location, NrnSeclistCompLocation)
    assert location.seclist_name == "somatic"


@pytest.fixture
def protocol():

    protocol_configuration = ProtocolConfiguration(
        **{
            "name": "Step_250",
            "stimuli": [{
                "delay": 700.0,
                "amp": None,
                "thresh_perc": 250.0,
                "duration": 2000.0,
                "totduration": 3000.0,
                "holding_current": None
            }],
            "recordings": [{
                "type": "CompRecording",
                "name": f"Step_250.soma.v",
                "location": "soma",
                "variable": "v"
            }],
            "validation": False
        }
    )

    return define_protocol(protocol_configuration, stochasticity=True)


def test_define_protocol(protocol):

    assert isinstance(protocol, ThresholdBasedProtocol)
    assert isinstance(protocol.stimulus, IDrest)
    assert len(protocol.recordings) == 1
    assert isinstance(protocol.recordings[0], CompRecording)


def test_define_efeature(protocol):

    feature_config = EFeatureConfiguration(
        efel_feature_name="AP_amplitude",
        protocol_name="Step_250",
        recording_name="soma.v",
        mean=66.64,
        std=5.0,
        efel_settings={"stim_start": 300.},
        threshold_efeature_std=0.1
    )

    global_efel_settings = {"stim_start": 500.}

    feature = define_efeature(feature_config, protocol, global_efel_settings)

    numpy.testing.assert_almost_equal(feature.exp_std, 6.664)
    assert feature.stim_start == 300.
    assert feature.recording_names[''] == "Step_250.soma.v"


def test_start_from_emodel(db):
    """Test start_from_emodel in get_evaluator_from_access_point"""
    eva = get_evaluator_from_access_point(db)
    assert len(eva.param_names) == 31
    assert 'constant.distribution_decay' in eva.param_names
    assert eva.cell_model.params['constant.distribution_decay'].frozen is False
    assert eva.cell_model.params['constant.distribution_decay'].bounds is not None

    db.pipeline_settings.start_from_emodel = {"emodel": "cADpyr_L5TPC"}
    db.pipeline_settings.parameters_to_freeze = []
    eva = get_evaluator_from_access_point(db)
    assert len(eva.param_names) == 0
    assert eva.cell_model.params['constant.distribution_decay'].frozen is True
    assert eva.cell_model.params['constant.distribution_decay'].bounds is None

    db.pipeline_settings.parameters_to_freeze = [
        'constant.distribution_decay', 'g_pas.all', 'e_pas.all', 'gIhbar_Ih.somadend'
    ]
    eva = get_evaluator_from_access_point(db)
    assert len(eva.param_names) == 27
    assert 'constant.distribution_decay' not in eva.param_names
    assert eva.cell_model.params['constant.distribution_decay'].frozen is True
    assert eva.cell_model.params['constant.distribution_decay'].bounds is None
    assert eva.cell_model.params['gK_Tstbar_K_Tst.axonal'].frozen is False
    assert eva.cell_model.params['gK_Tstbar_K_Tst.axonal'].bounds is not None
