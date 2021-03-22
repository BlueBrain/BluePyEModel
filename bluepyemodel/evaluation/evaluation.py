""" Emodels evaluation functions """
import copy
import logging

from bluepyemodel.evaluation import model
from bluepyemodel.evaluation.evaluator import create_evaluator

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

    return eva.run_protocols(protocols=eva.fitness_protocols.values(), param_values=params)


def compute_responses(
    emodel_db,
    emodel,
    cell_evaluator,
    map_function,
):
    """Compute the responses of the emodel to the optimisation and validation protocols.

    Args:
        emodel_db (DatabaseAPI): API used to access the database.
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        cell_evaluator (CellEvaluator): evaluator for the cell model/protocols/e-feature set.
        map_function (map): used to parallelize the evaluation of the
            individual in the population.
    Returns:
        emodels (list): list of emodels.
    """

    emodels = emodel_db.get_emodels([emodel])

    if emodels:

        logger.info("In compute_responses, %s emodels found to evaluate.", len(emodels))

        to_run = []
        for mo in emodels:
            to_run.append(
                {
                    "evaluator": copy.deepcopy(cell_evaluator),
                    "parameters": mo["parameters"],
                }
            )

        responses = list(map_function(get_responses, to_run))

        for mo, r in zip(emodels, responses):
            mo["responses"] = r

    else:
        logger.warning("In compute_responses, no emodel for %s", emodel)

    return emodels


def get_evaluator_from_db(
    emodel,
    db,
    morphology_modifiers=None,
    stochasticity=False,
    include_validation_protocols=False,
    additional_protocols=None,
    timeout=600,
):
    """Create an evaluator for the emodel.

    Args:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        db (DatabaseAPI): API used to access the database
        githash (str): if provided, the pipeline will work in the directory
            working_dir/run/githash. Needed when continuing work or resuming
            optimisations.
        mechanisms_dir (str): path of the directory in which the mechanisms
            will be copied and/or compiled. It has to be a subdirectory of
            working_dir.
        morphology_modifiers (list): list of python functions that will be
            applied to all the morphologies.
        stochasticity (bool): should channels behave stochastically if they can.
        copy_mechanisms (bool): should the mod files be copied in the local
            mechanisms_dir directory.
        compile_mechanisms (bool): should the mod files be compiled.
        timeout (float): duration (in second) after which the evaluation of a
            protocol will be interrupted.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        additional_protocols (dict): definition of supplementary protocols. See
            examples/optimisation for usage.

    Returns:
        bluepyopt.ephys.evaluators.CellEvaluator
    """

    parameters, mechanisms, _ = db.get_parameters()
    if not (parameters) or not (mechanisms):
        raise Exception("No parameters for emodel %s" % emodel)

    morphologies = db.get_morphologies()
    if not (morphologies):
        raise Exception("No morphologies for emodel %s" % emodel)

    efeatures = db.get_features(include_validation_protocols)
    if not (efeatures):
        raise Exception("No efeatures for emodel %s" % emodel)

    protocols = db.get_protocols(include_validation_protocols)
    if not (protocols):
        raise Exception("No protocols for emodel %s" % emodel)
    if additional_protocols:
        protocols.update(additional_protocols)

    cell_models = model.create_cell_models(
        emodel=emodel,
        morphologies=morphologies,
        mechanisms=mechanisms,
        parameters=parameters,
        morph_modifiers=morphology_modifiers,
    )

    return create_evaluator(
        cell_model=cell_models[0],
        protocols_definition=protocols,
        features_definition=efeatures,
        stochasticity=stochasticity,
        timeout=timeout,
    )
