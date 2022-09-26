"""Validate function."""

import logging
import pathlib
from importlib.machinery import SourceFileLoader

import numpy

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.validation import validation_functions

logger = logging.getLogger(__name__)


def define_validation_function(access_point):
    """Define the validation function based on the validation_function setting. If the settings
    was not specified, validate_max_score is used"""

    validation_function = access_point.pipeline_settings.validation_function

    if validation_function is None or not validation_function:
        logger.warning("Validation function not specified, will use validate_max_score.")
        validation_function = validation_functions.validate_max_score

    else:

        if isinstance(validation_function, str):
            if validation_function == "max_score":
                validation_function = validation_functions.validate_max_score
            elif validation_function == "mean_score":
                validation_function = validation_functions.validate_mean_score
            else:
                raise Exception("validation_function must be 'max_score' or 'mean_score'.")

        elif isinstance(validation_function, list) and len(validation_function) == 2:
            # pylint: disable=deprecated-method,no-value-for-parameter
            function_module = SourceFileLoader(
                pathlib.Path(validation_function[0]).stem, validation_function[0]
            ).load_module()
            validation_function = getattr(function_module, validation_function[1])

        if not callable(validation_function):
            raise Exception("validation_function is not callable nor a list of two strings")

    return validation_function


def validate(
    access_point,
    mapper,
):
    """Compute the scores and traces for the optimisation and validation
    protocols and perform validation.

    Args:
        access_point (DataAccessPoint): data access point.
        mapper (map): used to parallelize the evaluation of the
            individual in the population.

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
        preselect_for_validation=True,
    )

    if not emodels:
        logger.warning("In compute_scores, no emodels for %s", access_point.emodel_metadata.emodel)
        return []

    validation_function = define_validation_function(access_point)

    logger.info("In validate, %s emodels found to validate.", len(emodels))

    for i, mo in enumerate(emodels):
        # pylint: disable=unnecessary-list-index-lookup

        emodels[i].scores = mo.evaluator.fitness_calculator.calculate_scores(mo.responses)
        # turn features from arrays to float to be json serializable
        emodels[i].features = {}
        values = mo.evaluator.fitness_calculator.calculate_values(mo.responses)
        for key, value in values.items():
            if value is not None:
                emodels[i].features[key] = float(numpy.mean([v for v in value if v]))
            else:
                emodels[i].features[key] = None

        emodels[i].scores_validation = {}
        to_remove = []
        for feature_names in mo.scores:
            for p in access_point.pipeline_settings.validation_protocols:
                if p in feature_names:
                    emodels[i].scores_validation[feature_names] = mo.scores[feature_names]
                    to_remove.append(feature_names)
                    break
        emodels[i].scores = {k: v for k, v in emodels[i].scores.items() if k not in to_remove}

        # turn bool_ into bool to be json serializable
        emodels[i].passed_validation = bool(
            validation_function(
                mo,
                access_point.pipeline_settings.validation_threshold,
                False,
            )
        )

        access_point.store_emodel(emodels[i])

    return emodels
