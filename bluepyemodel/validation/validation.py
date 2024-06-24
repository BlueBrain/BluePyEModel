"""Validation functions."""

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
import pathlib
from importlib.machinery import SourceFileLoader

import numpy

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.tools.utils import are_same_protocol
from bluepyemodel.validation import validation_functions

logger = logging.getLogger(__name__)


def define_validation_function(access_point):
    """Define the validation function based on the validation_function setting. If the settings
    was not specified, validate_max_score is used"""

    validation_function = access_point.pipeline_settings.validation_function

    if validation_function is None or not validation_function:
        logger.warning("Validation function not specified, will use validate_max_score.")
        return validation_functions.validate_max_score

    if isinstance(validation_function, str):
        if validation_function == "max_score":
            validation_function = validation_functions.validate_max_score
        elif validation_function == "mean_score":
            validation_function = validation_functions.validate_mean_score
        else:
            raise ValueError("validation_function must be 'max_score' or 'mean_score'.")

    elif isinstance(validation_function, list) and len(validation_function) == 2:
        # pylint: disable=deprecated-method,no-value-for-parameter
        function_module = SourceFileLoader(
            pathlib.Path(validation_function[0]).stem, validation_function[0]
        ).load_module()
        validation_function = getattr(function_module, validation_function[1])

    if not callable(validation_function):
        raise TypeError("validation_function is not callable nor a list of two strings")

    return validation_function


def compute_scores(model, validation_protocols):
    """Compute the scores of an emodel.

    Args:
        model (EModel): emodel
        validation_protocols (list): list of validation protocols
    """
    model.features = model.evaluator.fitness_calculator.calculate_values(model.responses)
    for key, value in model.features.items():
        if value is not None:
            # turn features from arrays to float to be json serializable
            model.features[key] = float(numpy.nanmean([v for v in value if v is not None]))

    scores = model.evaluator.fitness_calculator.calculate_scores(model.responses)
    for feature_name in scores:
        protocol_name = feature_name.split(".")[0]
        if any(are_same_protocol(p, protocol_name) for p in validation_protocols):
            model.scores_validation[feature_name] = scores[feature_name]
        else:
            model.scores[feature_name] = scores[feature_name]


def validate(access_point, mapper, preselect_for_validation=False):
    """Compute the scores and traces for the optimisation and validation
    protocols and perform validation.

    Args:
        access_point (DataAccessPoint): data access point.
        mapper (map): used to parallelize the evaluation of the
            individual in the population.
        preselect_for_validation (bool): True to not re-run already validated models

    Returns:
        emodels (list): list of emodels.
    """

    cell_evaluator = get_evaluator_from_access_point(
        access_point,
        include_validation_protocols=True,
    )

    emodels = compute_responses(
        access_point,
        cell_evaluator=cell_evaluator,
        map_function=mapper,
        preselect_for_validation=preselect_for_validation,
    )

    if not emodels:
        logger.warning("In validate, no emodels for %s", access_point.emodel_metadata.emodel)
        return []

    validation_function = define_validation_function(access_point)

    logger.info("In validate, %s emodels found to validate.", len(emodels))

    for model in emodels:
        compute_scores(model, access_point.pipeline_settings.validation_protocols)

        # turn bool_ into bool to be json serializable
        model.passed_validation = bool(
            validation_function(
                model,
                access_point.pipeline_settings.validation_threshold,
                False,
            )
        )

        access_point.store_emodel(model)

    return emodels
