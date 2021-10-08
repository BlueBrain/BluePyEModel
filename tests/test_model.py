import pytest
import bluepyopt.ephys as ephys

from bluepyemodel.evaluation import model
from bluepyemodel.model_configuration.parameter_configuration import ParameterConfiguration
from bluepyemodel.model_configuration.distribution_configuration import DistributionConfiguration

def test_multi_locations():

    section = "alldend"
    locations = model.multi_locations(section, {})

    assert isinstance(locations[0], ephys.locations.NrnSeclistLocation)
    assert len(locations) == 2


def test_define_parameters():

    parameters = [
        {
            'location': "global",
            'name': "celsius",
            'value': 34,
        },
        {
            'location': "distribution_decay",
            'name': "constant",
            'value': [-0.1, 0.0],            
        },
        {
            'location': "somadend",
            'name': "gIhbar_Ih",
            'value': [0, 2e-4],  
            'distribution': "exp"
        }
    ]

    distributions = [
        DistributionConfiguration(
            "exp",
            "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}" 
        ),
        DistributionConfiguration(
            "decay",
            "math.exp({distance}*{constant})*{value}",
            ["constant"]
        )
    ]

    distributions = model.define_distributions(distributions)
    parameters = model.define_parameters(
        [ParameterConfiguration(**p) for p in parameters],
        distributions,
        {}
    )

    for param in parameters:
        assert isinstance(param, ephys.parameters.NrnParameter)

        if param.name == "distribution_decay":
            assert param.bounds[0] == -0.1
            assert param.bounds[1] == -0.0
