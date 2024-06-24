"""
Copyright 2023-2024 Blue Brain Project / EPFL

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

import logging

from bluepyemodel.validation.validation import define_validation_function, validate


def _always_true_validation(model, threshold=5.0, validation_protocols_only=False):
    return True


def test_define_validation_function(db):

    model = {
        "scores": {"a": 0.0, "b": 4.9, "c": 0.5, "d": 9.9},
        "scores_validation": {"c": 0.5, "d": 9.9},
    }

    db.pipeline_settings.validation_function = _always_true_validation

    validation_function = define_validation_function(db)

    validated = bool(
        validation_function(
            model,
            db.pipeline_settings.validation_threshold,
            False,
        )
    )

    assert validated


def test_validation(db):

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    db.get_mechanisms_directory = lambda: None
    emodels = validate(
        access_point=db,
        mapper=map,
        preselect_for_validation=True,
    )

    assert len(emodels) == 1
    assert emodels[0].passed_validation is True
