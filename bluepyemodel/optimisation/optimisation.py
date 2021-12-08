"""Optimisation function"""
import logging
import os
import pickle
from pathlib import Path

import bluepyopt

from bluepyemodel.emodel_pipeline.utils import logger
from bluepyemodel.emodel_pipeline.utils import run_metadata_as_string
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point

logger = logging.getLogger(__name__)


def get_checkpoint_path(emodel, seed, ttype=None, iteration_tag=None):
    """"""

    filename = run_metadata_as_string(emodel, seed, ttype=ttype, iteration_tag=iteration_tag)

    return f"./checkpoints/{filename}.pkl"


def parse_legacy_checkpoint_path(path):
    """"""

    filename = Path(path).stem.split("__")

    if len(filename) == 4:
        checkpoint_metadata = {
            "emodel": filename[1],
            "seed": filename[3],
            "iteration_tag": filename[2],
            "ttype": None,
        }
    elif len(filename) == 3:
        checkpoint_metadata = {
            "emodel": filename[1],
            "seed": filename[2],
            "iteration_tag": None,
            "ttype": None,
        }

    return checkpoint_metadata


def parse_checkpoint_path(path):
    """"""

    if "emodel" not in path and "checkpoint" in path:
        return parse_legacy_checkpoint_path(path)

    if path.endswith(".tmp"):
        path = path.replace(".tmp", "")

    filename = Path(path).stem.split("__")

    checkpoint_metadata = {}

    for field in ["emodel", "seed", "iteration_tag", "ttype"]:
        search_str = f"{field}="
        checkpoint_metadata[field] = next(
            (e.replace(search_str, "") for e in filename if search_str in e), None
        )

    return checkpoint_metadata


def read_checkpoint(checkpoint_path):
    """Reads a BluePyOpt checkpoint file"""

    p = Path(checkpoint_path)
    p_tmp = p.with_suffix(p.suffix + ".tmp")

    try:
        with open(str(p), "rb") as checkpoint_file:
            run = pickle.load(checkpoint_file)
            run_metadata = parse_checkpoint_path(str(p))
    except EOFError:
        try:
            with open(str(p_tmp), "rb") as checkpoint_tmp_file:
                run = pickle.load(checkpoint_tmp_file)
                run_metadata = parse_checkpoint_path(str(p_tmp))
        except EOFError:
            logger.error(
                "Cannot store model. Checkpoint file %s does not exist or is corrupted.",
                checkpoint_path,
            )

    return run, run_metadata


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

    checkpoint_path = get_checkpoint_path(
        access_point.emodel,
        seed,
        ttype=access_point.ttype,
        iteration_tag=access_point.iteration_tag,
    )

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

        checkpoint_path = get_checkpoint_path(
            access_point.emodel,
            seed=seed,
            ttype=access_point.ttype,
            iteration_tag=access_point.iteration_tag,
        )

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

    access_point.store_emodel(
        scores=scores,
        params=params,
        optimizer_name=access_point.pipeline_settings.optimizer,
        seed=run_metadata.get("seed", None),
    )
