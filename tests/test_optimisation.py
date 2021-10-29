import pytest

from bluepyemodel.optimisation.optimisation import get_checkpoint_path, parse_checkpoint_path


def test_get_checkpoint_path():
    
    emodel = "L5PC"
    seed = 0
    ttype = "t type"
    iteration_tag = "test"

    path = get_checkpoint_path(emodel, seed, ttype=ttype, iteration_tag=iteration_tag)
    assert str(path) == "./checkpoints/emodel=L5PC__seed=0__iteration_tag=test__ttype=t type.pkl"


def test_parse_checkpoint_path():

    metadata = parse_checkpoint_path(
        "./checkpoints/emodel=L5PC__seed=0__iteration_tag=test__ttype=t type.pkl"
    )
    assert metadata == {
        "emodel": "L5PC",
        "seed": "0",
        "ttype": "t type",
        "iteration_tag": "test" 
    }

    metadata = parse_checkpoint_path(
        "./checkpoints/checkpoint__L5PCpyr_ET1_dend__b6f7190__6.pkl"
    )
    assert metadata == {
        "emodel": "L5PCpyr_ET1_dend",
        "seed": "6",
        "ttype": None,
        "iteration_tag": "b6f7190" 
    }
