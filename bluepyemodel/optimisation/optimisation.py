"""Optimisation function"""

"""
Copyright 2023, EPFL/Blue Brain Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os
from pathlib import Path

import bluepyopt

from bluepyemodel.emodel_pipeline.emodel import EModel
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.tools.utils import get_checkpoint_path
from bluepyemodel.tools.utils import get_legacy_checkpoint_path
from bluepyemodel.tools.utils import logger
from bluepyemodel.tools.utils import read_checkpoint

logger = logging.getLogger(__name__)


def setup_optimiser(
    evaluator, map_function, params, optimiser="IBEA", use_stagnation_criterion=True
):
    """Setup the bluepyopt optimiser.

    Args:
        evaluator (CellEvaluator): evaluator used to compute the scores.
        map_function (map): used to parallelize the evaluation of the
            individual in the population.
        params (dict): optimisation meta-parameters.
        optimiser (str): name of the optimiser, has to be "IBEA", "SO-CMA" or
            "MO-CMA".
        use_stagnation_criterion (bool): whether to use the stagnation
            stopping criterion on top of the maximum generation criterion for CMA

    Returns:
        DEAPOptimisation
    """
    if optimiser == "IBEA":
        return bluepyopt.deapext.optimisations.IBEADEAPOptimisation(
            evaluator=evaluator, map_function=map_function, **params
        )
    if optimiser == "SO-CMA":
        return bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
            evaluator=evaluator,
            map_function=map_function,
            selector_name="single_objective",
            use_stagnation_criterion=use_stagnation_criterion,
            **params,
        )
    if optimiser == "MO-CMA":
        return bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
            evaluator=evaluator,
            map_function=map_function,
            selector_name="multi_objective",
            use_stagnation_criterion=use_stagnation_criterion,
            **params,
        )
    raise ValueError(f"Unknown optimiser: {optimiser}")


def run_optimisation(
    optimiser, checkpoint_path, max_ngen, terminator=None, optimisation_checkpoint_period=None
):
    """Run the optimisation.

    Args:
        optimiser (DEAPOptimisation): optimiser used for the run.
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
    elif Path(get_legacy_checkpoint_path(checkpoint_path)).is_file():
        checkpoint_path = get_legacy_checkpoint_path(checkpoint_path)
        continue_opt = True
        logger.info(
            "Found a legacy checkpoint path. Will use it instead "
            "and continue optimisation from last generation."
        )
    else:
        logger.info("No checkpoint found. Will start optimisation from scratch.")
        continue_opt = False

    logger.info("Running optimisation ...")
    pop, hof, log, history = optimiser.run(
        max_ngen=max_ngen,
        cp_filename=str(checkpoint_path),
        cp_period=optimisation_checkpoint_period,
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

    use_stagnation_criterion = access_point.pipeline_settings.use_stagnation_criterion
    opt_params = access_point.pipeline_settings.optimisation_params
    if "centroids" in opt_params and isinstance(opt_params["centroids"][0], dict):
        opt_params["centroids"][0] = [
            opt_params["centroids"][0][name] for name in list(cell_evaluator.param_names)
        ]

    opt_params["seed"] = seed

    opt = setup_optimiser(
        cell_evaluator,
        mapper,
        params=opt_params,
        optimiser=access_point.pipeline_settings.optimiser,
        use_stagnation_criterion=use_stagnation_criterion,
    )

    checkpoint_path = get_checkpoint_path(access_point.emodel_metadata, seed)

    optimisation_checkpoint_period = access_point.pipeline_settings.optimisation_checkpoint_period
    run_optimisation(
        optimiser=opt,
        checkpoint_path=checkpoint_path,
        max_ngen=access_point.pipeline_settings.max_ngen,
        terminator=terminator,
        optimisation_checkpoint_period=optimisation_checkpoint_period,
    )


def store_best_model(
    access_point,
    seed=None,
    checkpoint_path=None,
):
    """Store the best model from an optimisation. Reads a checkpoint file generated
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
            raise TypeError("Please specify either the seed or the checkpoint_path")

        checkpoint_path = get_checkpoint_path(access_point.emodel_metadata, seed=seed)

    run, run_metadata = read_checkpoint(checkpoint_path)

    best_model = run["halloffame"][0]
    feature_names = [obj.name for obj in cell_evaluator.fitness_calculator.objectives]

    if "param_names" in run:
        if run["param_names"] != list(cell_evaluator.param_names):
            raise ValueError(
                "The parameter names present in the checkpoint file are different from the "
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
