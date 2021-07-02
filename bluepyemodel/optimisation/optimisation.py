"""Optimisation function"""
import logging
from pathlib import Path

import bluepyopt

from bluepyemodel.emodel_pipeline.utils import read_checkpoint
from bluepyemodel.evaluation.evaluation import get_evaluator_from_db

logger = logging.getLogger(__name__)


def get_checkpoint_path(emodel, seed, githash=""):
    """"""
    return Path("./checkpoints") / f"checkpoint__{emodel}__{githash}__{seed}.pkl"


def setup_optimizer(evaluator, map_function, params, optimizer="IBEA"):
    """Setup the bluepyopt optimiser.

    Args:
        evaluator (CellEvaluator): evaluator used to compute the scores.
        map_function (map): used to parallelize the evaluation of the
            individual in the population.
        params (dict): optimization meta-parameters.
        optimizer (str): name of the optimiser, has to be "IBEA", "SO-CMA" or
            "MO-CMA".

    Returns:
        DEAPOptimisation
    """
    if optimizer == "IBEA":
        return bluepyopt.deapext.optimisations.IBEADEAPOptimisation(
            evaluator=evaluator, map_function=map_function, **params
        )
    if optimizer == "SO-CMA":
        return bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
            evaluator=evaluator,
            map_function=map_function,
            selector_name="single_objective",
            **params,
        )
    if optimizer == "MO-CMA":
        return bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
            evaluator=evaluator,
            map_function=map_function,
            selector_name="multi_objective",
            **params,
        )
    raise Exception("Unknown optimizer: {}".format(optimizer))


def run_optimization(optimizer, checkpoint_path, max_ngen, continue_opt, terminator=None):
    """Run the optimisation.

    Args:
        optimizer (DEAPOptimisation): optimiser used for the run.
        checkpoint_dir (Path): path to which the checkpoint will be saved.
        max_ngen (int): maximum number of generation for which the
            evolutionary strategy will run.
        terminator (multiprocessing.Event): end optimisation when is set.
            Not taken into account if None.

    Returns:
        None
    """
    checkpoint_path.parents[0].mkdir(parents=True, exist_ok=True)

    logger.info("Running optimisation ...")
    pop, hof, log, history = optimizer.run(
        max_ngen=max_ngen,
        cp_filename=str(checkpoint_path),
        continue_cp=continue_opt,
        terminator=terminator,
    )
    logger.info("Running optimisation ... Done.")

    return pop, hof, log, history


def setup_and_run_optimisation(
    access_point,
    emodel,
    seed,
    mapper=None,
    continue_opt=False,
    githash="",
    terminator=None,
):

    cell_evaluator = get_evaluator_from_db(
        emodel=emodel, access_point=access_point, include_validation_protocols=False
    )

    opt_params = access_point.pipeline_settings.optimisation_params
    opt_params["seed"] = seed

    opt = setup_optimizer(
        cell_evaluator,
        mapper,
        params=opt_params,
        optimizer=access_point.pipeline_settings.optimizer,
    )

    checkpoint_path = get_checkpoint_path(emodel, seed, githash)

    run_optimization(
        optimizer=opt,
        checkpoint_path=checkpoint_path,
        max_ngen=access_point.pipeline_settings.max_ngen,
        continue_opt=continue_opt,
        terminator=terminator,
    )


def store_best_model(
    access_point,
    emodel,
    seed,
    checkpoint_path=None,
    githash="",
):
    """Store the best model from an optimization. Reads a checkpoint file generated
        by BluePyOpt and store the best individual of the hall of fame.

    Args:
        access_point (DataAccessPoint): data access point.
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        seed (int): seed used in the optimisation.
            and validation efeatures be added to the evaluator.
        checkpoint_path (str): path to the checkpoint file. If None, checkpoint_dir will be used.
        githash (str): if provided, the pipeline will work in the directory
                working_dir/run/githash. Needed when continuing work or resuming
                optimisations.
    """

    cell_evaluator = get_evaluator_from_db(
        emodel=emodel,
        access_point=access_point,
        include_validation_protocols=False,
    )

    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(emodel, seed, githash)

    run = read_checkpoint(checkpoint_path)

    best_model = run["halloffame"][0]
    feature_names = [obj.name for obj in cell_evaluator.fitness_calculator.objectives]
    param_names = list(cell_evaluator.param_names)

    scores = dict(zip(feature_names, best_model.fitness.values))
    params = dict(zip(param_names, best_model))

    access_point.store_emodel(
        scores=scores,
        params=params,
        optimizer_name=access_point.pipeline_settings.optimizer,
        seed=seed,
        githash=githash,
    )
