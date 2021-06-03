"""Validate function."""

import logging

import numpy

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_db
from bluepyemodel.validation import validation_functions

logger = logging.getLogger(__name__)


def validate(
    emodel_db,
    emodel,
    mapper,
    validation_function=None,
    stochasticity=False,
    additional_protocols=None,
    threshold=5.0,
    validation_protocols_only=False,
):
    """Compute the scores and traces for the optimisation and validation
    protocols and perform validation.

    Args:
        emodel_db (DatabaseAPI): API used to access the database.
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        mapper (map): used to parallelize the evaluation of the
            individual in the population.
        validation_function (str): function used to decide if a model
            passes validation or not. Should rely on emodel['scores'] and
            emodel['scores_validation']. See bluepyemodel/validation for examples.
            Should be a function name in bluepyemodel.validation.validation_functions
        stochasticity (bool): should channels behave stochastically if they can.
        copy_mechanisms (bool): should the mod files be copied in the local
            mechanisms_dir directory.
        compile_mechanisms (bool): should the mod files be compiled.
        mechanisms_dir (str): path of the directory in which the mechanisms
            will be copied and/or compiled. It has to be a subdirectory of
            working_dir.
        additional_protocols (dict): definition of supplementary protocols. See
            examples/optimisation for usage.
        threshold (float): threshold under which the validation function returns True.
        validation_protocols_only (bool): True to only use validation protocols
            during validation.

    Returns:
        emodels (list): list of emodels.
    """
    if additional_protocols is None:
        additional_protocols = {}

    cell_evaluator = get_evaluator_from_db(
        emodel,
        emodel_db,
        stochasticity=stochasticity,
        include_validation_protocols=True,
        additional_protocols=additional_protocols,
    )

    emodels = compute_responses(
        emodel_db,
        emodel,
        cell_evaluator,
        mapper,
        preselect_for_validation=True,
    )

    if emodels:

        if validation_function:
            validation_function = getattr(validation_functions, validation_function)
        else:
            logger.warning("Validation function not  specified, will use validate_max_score.")
            validation_function = validation_functions.validate_max_score

        emodel_db.set_emodel(emodel)
        name_validation_protocols = emodel_db.get_name_validation_protocols()

        logger.info("In validate, %s emodels found to validate.", len(emodels))

        for mo in emodels:

            mo["scores"] = cell_evaluator.fitness_calculator.calculate_scores(mo["responses"])

            values = cell_evaluator.fitness_calculator.calculate_values(mo["responses"])
            # turn features from arrays to float to be json serializable
            mo["features"] = {k: float(numpy.mean(v)) for k, v in values.items()}

            mo["scores_validation"] = {}
            for feature_names, score in mo["scores"].items():
                for p in name_validation_protocols:
                    if p in feature_names:
                        mo["scores_validation"][feature_names] = score

            # turn bool_ into bool to be json serializable
            validated = bool(validation_function(mo, threshold, validation_protocols_only))

            emodel_db.store_emodel(
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

    logger.warning("In compute_scores, no emodel for %s", emodel)
    return []
