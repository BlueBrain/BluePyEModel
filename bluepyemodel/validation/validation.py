"""Validate function."""

import logging
import pathlib
from importlib.machinery import SourceFileLoader

import numpy

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_db
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

        if isinstance(validation_function, list) and len(validation_function) == 2:
            # pylint: disable=deprecated-method,no-value-for-parameter
            function_module = SourceFileLoader(
                pathlib.Path(validation_function[0]).stem, validation_function[0]
            ).load_module()
            validation_function = getattr(function_module, validation_function[1])

        elif not callable(validation_function):
            raise Exception("validation_function is not callable nor a list of two strings")

    return validation_function


def validate(
    access_point,
    emodel,
    mapper,
):
    """Compute the scores and traces for the optimisation and validation
    protocols and perform validation.

    Args:
        access_point (DataAccessPoint): data access point.
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        mapper (map): used to parallelize the evaluation of the
            individual in the population.

    Returns:
        emodels (list): list of emodels.
    """

    cell_evaluator = get_evaluator_from_db(
        emodel,
        access_point,
        include_validation_protocols=True,
        additional_protocols={},
    )

    emodels = compute_responses(
        access_point, emodel, cell_evaluator, mapper, preselect_for_validation=True
    )

    if not emodels:
        logger.warning("In compute_scores, no emodels for %s", emodel)
        return []

    validation_function = define_validation_function(access_point)

    access_point.set_emodel(emodel)
    name_validation_protocols = access_point.get_name_validation_protocols()

    logger.info("In validate, %s emodels found to validate.", len(emodels))

    for i, mo in enumerate(emodels):

        emodels[i]["scores"] = mo["evaluator"].fitness_calculator.calculate_scores(mo["responses"])
        # turn features from arrays to float to be json serializable
        emodels[i]["features"] = {}
        values = mo["evaluator"].fitness_calculator.calculate_values(mo["responses"])
        for key, value in values.items():
            if value is not None:
                emodels[i]["features"][key] = float(numpy.mean([v for v in value if v]))
            else:
                emodels[i]["features"][key] = None

        emodels[i]["scores_validation"] = {}
        for feature_names, score in mo["scores"].items():
            for p in name_validation_protocols:
                if p in feature_names:
                    emodels[i]["scores_validation"][feature_names] = score

        # turn bool_ into bool to be json serializable
        emodels[i]["validated"] = bool(
            validation_function(
                mo,
                access_point.pipeline_settings.validation_threshold,
                False,
            )
        )

        access_point.store_emodel(
            scores=emodels[i]["scores"],
            params=emodels[i]["parameters"],
            optimizer_name=emodels[i]["optimizer"],
            seed=emodels[i]["seed"],
            githash=emodels[i]["githash"],
            validated=emodels[i]["validated"],
            scores_validation=emodels[i]["scores_validation"],
            features=emodels[i]["features"],
        )

    return emodels
