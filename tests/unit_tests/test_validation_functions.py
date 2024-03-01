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

from bluepyemodel.emodel_pipeline.emodel import EModel
from bluepyemodel.validation import validation_functions


def test_validate_max_score():

    model = EModel(
        score={"a": 0.0, "b": 4.9, "c": 0.5, "d": 9.9},
        scoreValidation={"c": 0.5, "d": 9.9}
    )

    validation_result = validation_functions.validate_max_score(
        model, threshold=10.0, validation_protocols_only=True
    )
    assert validation_result

    validation_result = validation_functions.validate_max_score(
        model, threshold=5.0, validation_protocols_only=False
    )
    assert not validation_result


def test_validate_mean_score():

    model = EModel(
        score={"a": 0.0, "b": 4.9, "c": 0.5, "d": 9.9},
        scoreValidation={"c": 0.5, "d": 9.9}
    )

    validation_result = validation_functions.validate_mean_score(
        model, threshold=3.0, validation_protocols_only=True
    )
    assert not validation_result

    validation_result = validation_functions.validate_mean_score(
        model, threshold=5.0, validation_protocols_only=False
    )
    assert validation_result
