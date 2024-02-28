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

import bluepyopt.ephys as ephys

from bluepyemodel.model import model
from bluepyemodel.model.distribution_configuration import DistributionConfiguration
from bluepyemodel.model.parameter_configuration import ParameterConfiguration


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

        if param.name == "constant.distribution_decay":
            assert param.bounds[0] == -0.1
            assert param.bounds[1] == -0.0
