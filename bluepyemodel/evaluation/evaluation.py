""" Emodels evaluation functions """
import copy
import logging

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.evaluation.evaluator import create_evaluator
from bluepyemodel.model import model

logger = logging.getLogger(__name__)


def get_responses(to_run):
    """Compute the voltage responses of a set of parameters.

    Args:
        to_run (dict): of the form
            to_run = {"evaluator": CellEvaluator, "parameters": Dict}
    """

    eva = to_run["evaluator"]
    params = to_run["parameters"]

    eva.cell_model.unfreeze(params)

    responses = eva.run_protocols(protocols=eva.fitness_protocols.values(), param_values=params)
    responses["evaluator"] = eva

    return responses


def compute_responses(
    access_point,
    cell_evaluator,
    map_function,
    seeds=None,
    preselect_for_validation=False,
):
    """Compute the responses of the emodel to the optimisation and validation protocols.

    Args:
        access_point (DataAccessPoint): API used to access the data.
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        cell_evaluator (CellEvaluator): evaluator for the cell model/protocols/e-feature set.
        map_function (map): used to parallelize the evaluation of the
            individual in the population.
        seeds (list): if not None, filter emodels to keep only the ones with these seeds.
        preselect_for_validation (bool): if True,
            only select models that have not been through validation yet.
    Returns:
        emodels (list): list of emodels.
    """

    emodels = access_point.get_emodels()

    if seeds:
        emodels = [model for model in emodels if model.seed in seeds]
    if access_point.emodel_metadata.iteration:
        emodels = [
            model
            for model in emodels
            if model.emodel_metadata.iteration == access_point.emodel_metadata.iteration
        ]
    if preselect_for_validation:
        emodels = [model for model in emodels if model.passed_validation is None]

    if emodels:

        logger.info("In compute_responses, %s emodels found to evaluate.", len(emodels))

        to_run = []
        for mo in emodels:
            to_run.append(
                {
                    "evaluator": copy.deepcopy(cell_evaluator),
                    "parameters": mo.parameters,
                }
            )

        responses = list(map_function(get_responses, to_run))

        for mo, r in zip(emodels, responses):
            mo.responses = r
            mo.evaluator = r.pop("evaluator")

    else:
        logger.warning(
            "In compute_responses, no emodel for %s", access_point.emodel_metadata.emodel
        )

    return emodels


def get_evaluator_from_access_point(
    access_point,
    stochasticity=False,
    include_validation_protocols=False,
    timeout=None,
    score_threshold=12.0,
    max_threshold_voltage=-30,
    nseg_frequency=40,
    dt=None,
    strict_holding_bounds=True,
):
    """Create an evaluator for the emodel.

    Args:
        access_point (DataAccessPoint): API used to access the database
        stochasticity (bool): should channels behave stochastically if they can.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        timeout (float): duration (in second) after which the evaluation of a
            protocol will be interrupted.
        score_threshold (float): threshold for score of protocols to stop evaluations
        max_threshold_voltage (float): maximum voltage used as upper
            bound in the threshold current search
        dt (float): if not None, cvode will be disabled and fixed timesteps used.
        strict_holding_bounds (bool): to adaptively enlarge bounds is holding current is outside

    Returns:
        bluepyopt.ephys.evaluators.CellEvaluator
    """

    model_configuration = access_point.get_model_configuration()
    fitness_calculator_configuration = access_point.get_fitness_calculator_configuration()

    cell_model = model.create_cell_model(
        name=access_point.emodel_metadata.emodel,
        model_configuration=model_configuration,
        morph_modifiers=access_point.pipeline_settings.morph_modifiers,
        nseg_frequency=nseg_frequency,
    )

    timeout = timeout or access_point.pipeline_settings.optimisation_timeout
    stochasticity = stochasticity or access_point.pipeline_settings.stochasticity

    if isinstance(access_point, LocalAccessPoint):
        mechanisms_directory = None
    else:
        mechanisms_directory = access_point.get_mechanisms_directory()

    return create_evaluator(
        cell_model=cell_model,
        fitness_calculator_configuration=fitness_calculator_configuration,
        include_validation_protocols=include_validation_protocols,
        stochasticity=stochasticity,
        timeout=timeout,
        efel_settings=access_point.pipeline_settings.efel_settings,
        score_threshold=score_threshold,
        max_threshold_voltage=max_threshold_voltage,
        dt=dt,
        threshold_based_evaluator=access_point.pipeline_settings.threshold_based_evaluator,
        strict_holding_bounds=strict_holding_bounds,
        mechanisms_directory=mechanisms_directory,
        cvode_minstep=access_point.pipeline_settings.cvode_minstep,
    )
