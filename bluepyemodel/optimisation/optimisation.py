"""Optimisation function"""
import logging
import os
import pickle
from pathlib import Path

import bluepyopt

from bluepyemodel.emodel_pipeline.emodel import EModel
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.tools.utils import get_checkpoint_path, logger, read_checkpoint

logger = logging.getLogger(__name__)


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
    raise Exception(f"Unknown optimizer: {optimizer}")


def run_optimization(optimizer, checkpoint_path, max_ngen, terminator=None):
    """Run the optimisation.

    Args:
        optimizer (DEAPOptimisation): optimiser used for the run.
        checkpoint_path (str): path to which the checkpoint will be saved.
        max_ngen (int): maximum number of generation for which the
            evolutionary strategy will run.
        terminator (multiprocessing.Event): end optimisation when is set.
            Not taken into account if None.

    Returns:
        None
    """

    Path(checkpoint_path).parents[0].mkdir(parents=True, exist_ok=True)

    if os.path.isfile(checkpoint_path):
        logger.info(
            "Checkopint already exists."
            "Will continue optimisation from last generation in checkpoint"
        )
        continue_opt = True
    else:
        logger.info("No checkpoint found. Will start optimisation from scratch.")
        continue_opt = False

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
    seed,
    mapper=None,
    terminator=None,
):

    cell_evaluator = get_evaluator_from_access_point(
        access_point=access_point, include_validation_protocols=False
    )

    opt_params = access_point.pipeline_settings.optimisation_params
    if "centroids" in opt_params and isinstance(opt_params["centroids"][0], dict):
        opt_params["centroids"][0] = [
            opt_params["centroids"][0][name] for name in list(cell_evaluator.param_names)
        ]
    if opt_params is None and access_point.pipeline_settings.optimizer.endswith("CMA"):
        opt_params = {"offspring_size": 10, "weight_hv": 0.4}

    opt_params["seed"] = seed

    opt = setup_optimizer(
        cell_evaluator,
        mapper,
        params=opt_params,
        optimizer=access_point.pipeline_settings.optimizer,
    )

    checkpoint_path = get_checkpoint_path(access_point.emodel_metadata, seed)

    run_optimization(
        optimizer=opt,
        checkpoint_path=checkpoint_path,
        max_ngen=access_point.pipeline_settings.max_ngen,
        terminator=terminator,
    )


def store_best_model(
    access_point,
    seed=None,
    checkpoint_path=None,
):
    """Store the best model from an optimization. Reads a checkpoint file generated
        by BluePyOpt and store the best individual of the hall of fame.

    Args:
        access_point (DataAccessPoint): data access point.
        checkpoint_path (str): path to the checkpoint file. If None, checkpoint_dir will be used.
    """

    cell_evaluator = get_evaluator_from_access_point(
        access_point=access_point,
        include_validation_protocols=False,
    )

    if checkpoint_path is None:

        if seed is None:
            raise Exception("Please specify either the seed or the checkpoint_path")

        checkpoint_path = get_checkpoint_path(access_point.emodel_metadata, seed=seed)

    run, run_metadata = read_checkpoint(checkpoint_path)

    best_model = run["halloffame"][0]
    feature_names = [obj.name for obj in cell_evaluator.fitness_calculator.objectives]

    if "param_names" in run:
        if run["param_names"] != list(cell_evaluator.param_names):
            raise Exception(
                "The parameter names present in the checkpoint file are different from the"
                f"ones of the evaluator: {run['param_names']} versus "
                f"{list(cell_evaluator.param_names)}"
            )
        params = dict(zip(run["param_names"], best_model))
    else:
        params = dict(zip(list(cell_evaluator.param_names), best_model))

    scores = dict(zip(feature_names, best_model.fitness.values))

    emodel_seed = run_metadata.get("seed", None) if seed is None else seed

    emodel = EModel(
        fitness=sum(list(scores.values())),
        parameter=params,
        score=scores,
        seed=emodel_seed,
        emodel_metadata=access_point.emodel_metadata,
    )

    access_point.store_emodel(emodel)
