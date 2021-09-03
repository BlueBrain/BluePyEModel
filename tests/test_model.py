import pytest
from bluepyemodel.evaluation import model
import bluepyopt.ephys as ephys


def test_multi_locations():

    section = "alldend"
    locations = model.multi_locations(section, {})

    assert isinstance(locations[0], ephys.locations.NrnSeclistLocation)
    assert len(locations) == 2


def test_define_parameters():

    definitions = {
        "distributions": {
            "exp": {"fun": "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}"},
            "decay": {
                "fun": "math.exp({distance}*{constant})*{value}",
                "parameters": ["constant"],
            },
        },
        "parameters": {
            "global": [{"name": "celsius", "val": 34}],
            "distribution_decay": [{"name": "constant", "val": [-0.1, 0.0]}],
            "somadend": [{"name": "gIhbar_Ih", "val": [0, 2e-4], "dist": "exp"}],
        },
    }

    parameters = model.define_parameters(definitions)

    for param in parameters:
        assert isinstance(param, ephys.parameters.NrnParameter)

        if param.name == "distribution_decay":
            assert param.bounds[0] == -0.1
            assert param.bounds[1] == -0.0
