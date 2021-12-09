import pytest
from bluepyemodel.validation import validation_functions


def test_validate_max_score():

    model = {
        "scores": {"a": 0.0, "b": 4.9, "c": 0.5, "d": 9.9},
        "scores_validation": {"c": 0.5, "d": 9.9},
    }

    validation_result = validation_functions.validate_max_score(
        model, threshold=10.0, validation_protocols_only=True
    )
    assert validation_result

    validation_result = validation_functions.validate_max_score(
        model, threshold=5.0, validation_protocols_only=False
    )
    assert not validation_result


def test_validate_mean_score():

    model = {
        "scores": {"a": 0.0, "b": 4.9, "c": 0.5, "d": 9.9},
        "scores_validation": {"c": 0.5, "d": 9.9},
    }

    validation_result = validation_functions.validate_mean_score(
        model, threshold=3.0, validation_protocols_only=True
    )
    assert not validation_result

    validation_result = validation_functions.validate_mean_score(
        model, threshold=5.0, validation_protocols_only=False
    )
    assert validation_result
