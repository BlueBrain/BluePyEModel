"""Optimisation function"""
import logging
from pathlib import Path

import bluepyopt

from bluepyemodel.emodel_pipeline.utils import read_checkpoint
from bluepyemodel.evaluation.evaluation import get_evaluator_from_db

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


def setup_and_run_optimisation(  # pylint: disable=too-many-arguments
    emodel_db,
    emodel,
    seed,
    stochasticity=False,
    include_validation_protocols=False,
    timeout=None,
    mapper=None,
    opt_params=None,
    optimizer="IBEA",
    max_ngen=1000,
    checkpoint_dir=".",
    checkpoint_path=None,
    continue_opt=False,
    githash="",
    terminator=None,
):

    cell_evaluator = get_evaluator_from_db(
        emodel=emodel,
        db=emodel_db,
        stochasticity=stochasticity,
        include_validation_protocols=include_validation_protocols,
        timeout=timeout,
    )

    if opt_params is None and optimizer.endswith("CMA"):
        opt_params = {"offspring_size": 10, "weight_hv": 0.4}

    opt_params["seed"] = seed
    opt = setup_optimizer(cell_evaluator, mapper, opt_params, optimizer=optimizer)

    if checkpoint_path is None:
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint__{emodel}__{githash}__{seed}.pkl"

    run_optimization(
        optimizer=opt,
        checkpoint_path=checkpoint_path,
        max_ngen=max_ngen,
        continue_opt=continue_opt,
        terminator=terminator,
    )


def store_best_model(
    emodel_db,
    emodel,
    seed,
    stochasticity=False,
    include_validation_protocols=False,
    optimizer="IBEA",
    checkpoint_dir="./checkpoints",
    checkpoint_path=None,
    githash="",
):
    """Store the best model from an optimization. Reads a checkpoint file generated
        by BluePyOpt and store the best individual of the hall of fame.

    Args:
        emodel_db (DatabaseAPI): API used to access the database.
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        seed (int): seed used in the optimisation.
        stochasticity (bool): should channels behave stochastically if they can.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        optimizer (str): algorithm used for optimization, can be "IBEA", "SO-CMA",
            "MO-CMA".
        checkpoint_dir (str): path to the repo where files used as a checkpoint by BluePyOpt are.
        checkpoint_path (str): path to the checkpoint file. If None, checkpoint_dir will be used.
        githash (str): if provided, the pipeline will work in the directory
                working_dir/run/githash. Needed when continuing work or resuming
                optimisations.
    """

    cell_evaluator = get_evaluator_from_db(
        emodel=emodel,
        db=emodel_db,
        stochasticity=stochasticity,
        include_validation_protocols=include_validation_protocols,
    )

    if checkpoint_path is None:
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint__{emodel}__{githash}__{seed}.pkl"

    run = read_checkpoint(checkpoint_path)

    best_model = run["halloffame"][0]
    feature_names = [obj.name for obj in cell_evaluator.fitness_calculator.objectives]
    param_names = list(cell_evaluator.param_names)

    scores = dict(zip(feature_names, best_model.fitness.values))
    params = dict(zip(param_names, best_model))

    emodel_db.store_emodel(
        scores=scores,
        params=params,
        optimizer_name=optimizer,
        seed=seed,
        githash=githash,
    )
