import pytest

from bluepyemodel.model_configurator.neuron_model_configuration import NeuronModelConfiguration

@pytest.fixture
def configuration():
    config = NeuronModelConfiguration(configuration_name="Test")

    config.add_parameter(
        parameter_name="test_parameter",
        locations='all',
        value=5,
        mechanism='test_mechanism',
        distribution_name=None,
        stochastic=None
    )

    return config


@pytest.fixture
def configuration_with_distribution(configuration):

    configuration.add_distribution(
        distribution_name="not_constant",
        function="{value} * 2 * {dist_param1}",
        parameters=["dist_param1"]
    )

    configuration.add_parameter(
        parameter_name="test_parameter2",
        locations='soma',
        value=[1., 2.],
        mechanism='test_mechanism2',
        distribution_name="not_constant",
        stochastic=None
    )

    configuration.add_parameter(
        parameter_name="dist_param1",
        locations='not_constant',
        value=[3., 10.],
        mechanism=None,
        distribution_name=None,
        stochastic=None
    )

    return configuration


def test_add_parameter_mechanism(configuration):
    assert len(configuration.parameters) == 1
    assert len(configuration.mechanisms) == 1
    assert configuration.mechanisms[0].location == 'all'
    assert configuration.mechanisms[0].stochastic is False


def test_raise_add_mechanism(configuration):
    with pytest.raises(Exception):
        configuration.add_mechanism(
            mechanism_name='test_mechanism',
            locations='all',
            stochastic=True
        )


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


def test_distribution(configuration_with_distribution):
    assert len(configuration_with_distribution.parameters) == 3
    assert len(configuration_with_distribution.mechanisms) == 2
    assert len(configuration_with_distribution.distributions) == 1
    assert len(configuration_with_distribution.distribution_names) == 1
    assert len(configuration_with_distribution.used_distribution_names) == 1


def test_dicts(configuration_with_distribution):

    expected_param_dict = {
        'all': [{'name': 'test_parameter', 'val': 5}],
        'not_constant': [{'name': 'dist_param1', 'val': [3., 10.]}],
        'soma': [{'name': 'test_parameter2', 'val': [1., 2.], 'dist': 'not_constant'}]
    }

    expected_dist_dict = {
        'not_constant': {'fun': '{value} * 2 * {dist_param1}', 'parameters': ['dist_param1']}
    }

    expected_mech_dict = {
        'all': {'mech': ['test_mechanism'], 'stoch': [False]},
        'soma': {'mech': ['test_mechanism2'], 'stoch': [False]}
    }

    expected_mech_names = {'test_mechanism', 'test_mechanism2'}

    param_distr, mechanisms_dict, mechanism_names = configuration_with_distribution.as_legacy_dicts()

    for loc in expected_param_dict:
        for param in expected_param_dict[loc]:
            assert param == param_distr["parameters"][loc][0]
    assert expected_dist_dict['not_constant'] == param_distr['distributions']['not_constant']
    for loc in expected_mech_dict:
        assert expected_mech_dict[loc]['mech'] == mechanisms_dict[loc]['mech']
        assert expected_mech_dict[loc]['stoch'] == mechanisms_dict[loc]['stoch']
    assert expected_mech_names == mechanism_names
