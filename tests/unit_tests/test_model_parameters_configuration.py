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

import pytest

from bluepyemodel.model.mechanism_configuration import MechanismConfiguration
from bluepyemodel.model.neuron_model_configuration import NeuronModelConfiguration
from bluepyemodel.model.morphology_configuration import MorphologyConfiguration


@pytest.fixture
def configuration():

    available_mechs = [
        MechanismConfiguration(name="test_mechanism", location=None),
        MechanismConfiguration(name="test_mechanism2", location=None),
        MechanismConfiguration(name="test_mechanism3", location=None)
    ]

    available_morphologies = ["C060114A5"]
    config = NeuronModelConfiguration(
        available_mechanisms=available_mechs,
        available_morphologies=available_morphologies
    )

    config.add_parameter(
        parameter_name="test_parameter",
        locations='all',
        value=5,
        mechanism='test_mechanism',
        distribution_name=None,
        stochastic=None,
        auto_mechanism=True,
    )

    return config


@pytest.fixture
def configuration_with_distribution(configuration):

    configuration.add_distribution(
        distribution_name="not_constant",
        function="{value} * 2 * {dist_param1}",
        parameters=["dist_param1"],
    )

    configuration.add_parameter(
        parameter_name="gtest_parameter2",
        locations='soma',
        value=[1., 2.],
        mechanism='test_mechanism2',
        distribution_name="not_constant",
        stochastic=None,
        auto_mechanism=True,
    )

    configuration.add_parameter(
        parameter_name="dist_param1",
        locations='distribution_not_constant',
        value=[3., 10.],
        mechanism=None,
        distribution_name=None,
        stochastic=None,
    )

    return configuration


def test_add_parameter_mechanism(configuration):
    assert len(configuration.parameters) == 1
    assert len(configuration.mechanisms) == 1
    assert configuration.mechanisms[0].location == 'all'
    assert configuration.mechanisms[0].stochastic is False


def test_remove_parameter(configuration):

    configuration.add_parameter(
        parameter_name="test_parameter2",
        locations='axonal',
        value=[1., 2.],
        mechanism='test_mechanism2',
        stochastic=None
    )

    configuration.remove_parameter("test_parameter2", ['somatic'])
    assert len(configuration.parameters) == 2

    configuration.remove_parameter("test_parameter2")
    assert len(configuration.parameters) == 1


def test_remove_mechanisms(configuration):

    configuration.add_parameter(
        parameter_name="test_parameter2",
        locations='somatic',
        value=[1., 2.],
        mechanism='test_mechanism2',
        stochastic=None,
        auto_mechanism=True,
    )

    configuration.add_parameter(
        parameter_name="test_parameter3",
        locations='all',
        value=[1., 2.],
        mechanism='test_mechanism3',
        stochastic=None,
        auto_mechanism=True,
    )

    configuration.remove_mechanism("test_parameter2", "all")
    assert len(configuration.parameters) == 3
    assert len(configuration.mechanisms) == 3
    configuration.remove_mechanism("test_mechanism2")
    assert len(configuration.parameters) == 2
    assert len(configuration.mechanisms) == 2


def test_raise_distribution(configuration):
    with pytest.raises(Exception):
        configuration.add_parameter(
            parameter_name="test_parameter2",
            locations='all',
            value=[1., 2.],
            mechanism='test_mechanism2',
            distribution_name="not_constant",
            stochastic=None
        )


def test_raise_morphology(configuration):
    with pytest.raises(Exception):
        configuration.select_morphology(
            morphology_name="test",
        )


def test_morphology_configuration():
    with pytest.raises(Exception):
        morph_dict = {
            "path": "./C060114A5.asc",
            "format": "swc",
            "name": "C060114A5"
        }
        _ = MorphologyConfiguration(**morph_dict)

    morph_dict = {
        "path": "./C060114A5.asc",
        "format": "asc",
        "name": "C060114A5"
    }
    morph = MorphologyConfiguration(**morph_dict)
    assert morph.format == "asc"


def test_select_morphology(configuration):
    configuration.select_morphology(
        morphology_name="C060114A5",
    )
    morpho_dict = configuration.morphology.as_dict()
    assert configuration.morphology.name == "C060114A5"
    assert morpho_dict["name"] == "C060114A5"


def test_distribution(configuration_with_distribution):
    assert len(configuration_with_distribution.parameters) == 3
    assert len(configuration_with_distribution.mechanisms) == 2
    assert len(configuration_with_distribution.distributions) == 2
    assert len(configuration_with_distribution.distribution_names) == 2
    assert len(configuration_with_distribution.used_distribution_names) == 2
