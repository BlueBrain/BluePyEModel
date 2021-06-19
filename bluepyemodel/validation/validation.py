"""Validate function."""

import logging

import numpy

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_db
from bluepyemodel.validation import validation_functions

logger = logging.getLogger(__name__)


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

    if emodels:

        validation_function = access_point.pipeline_settings.validation_function
        if validation_function is None:
            logger.warning("Validation function not specified, will use validate_max_score.")
            validation_function = validation_functions.validate_max_score

        access_point.set_emodel(emodel)
        name_validation_protocols = access_point.get_name_validation_protocols()

        logger.info("In validate, %s emodels found to validate.", len(emodels))

        for mo in emodels:

            mo["scores"] = mo["evaluator"].fitness_calculator.calculate_scores(mo["responses"])
            # turn features from arrays to float to be json serializable
            mo["features"] = {}
            values = mo["evaluator"].fitness_calculator.calculate_values(mo["responses"])
            for key, value in values.items():
                if value is not None:
                    mo["features"][key] = float(numpy.mean([v for v in value if v]))
                else:
                    mo["features"][key] = None

            mo["scores_validation"] = {}
            for feature_names, score in mo["scores"].items():
                for p in name_validation_protocols:
                    if p in feature_names:
                        mo["scores_validation"][feature_names] = score

            # turn bool_ into bool to be json serializable
            validated = bool(
                validation_function(
                    mo,
                    access_point.pipeline_settings.validation_threshold,
                    False,
                )
            )

            access_point.store_emodel(
                scores=mo["scores"],
                params=mo["parameters"],
                optimizer_name=mo["optimizer"],
                seed=mo["seed"],
                githash=mo["githash"],
                validated=validated,
                scores_validation=mo["scores_validation"],
                features=mo["features"],
            )

        return emodels

    logger.warning("In compute_scores, no emodels for %s", emodel)
    return []
