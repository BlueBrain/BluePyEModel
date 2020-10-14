"""Optimisation function"""

import os
from datetime import datetime
from pathlib import Path
import logging

import bluepyopt

logger = logging.getLogger(__name__)


def ipyparallel_map_function(flag_use_IPYP=None, ipython_profil="IPYTHON_PROFILE"):
    """Get the map function linked to the ipython profile

    Args:
       flag_use_IPYP (str): name of the environement variable used as boolean
           to check if ipyparallel should be used
       ipython_profil (str): name fo the environement variable containing
           the name of the name of the ipython profile

    Returns:
        map
    """
    if flag_use_IPYP and os.getenv(flag_use_IPYP):
        from ipyparallel import Client

        rc = Client(profile=os.getenv(ipython_profil))
        lview = rc.load_balanced_view()

        def mapper(func, it):
            start_time = datetime.now()
            ret = lview.map_sync(func, it)
            logger.debug("Took %s", datetime.now() - start_time)
            return ret

    else:
        mapper = None

    return mapper


def setup_optimizer(evaluator, map_function, params, optimizer="IBEA"):
    """Setup the bluepyopt optimiser.

    Args:
        evaluator (CellEvaluator): evaluator used to compute the scores
        map_function (map): used to parallelize the evaluation of the
            individual in the ppopulation
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
            **params
        )
    if optimizer == "MO-CMA":
        return bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
            evaluator=evaluator,
            map_function=map_function,
            selector_name="multi_objective",
            **params
        )
    raise Exception("Unknown optimizer: {}".format(optimizer))


def run_optimization(optimizer, checkpoint_path, max_ngen):
    """Run the optimisation.

    Args:
        optimizer (DEAPOptimisation): optimiser used for the run.
        checkpoint_path (str): path to which the checkpoint will be saved.
        max_ngen (int): maximum number of generation for which the
            evolutionary strategy will run.

    Returns:
        None
    """
    path = Path(checkpoint_path)
    path.parents[0].mkdir(parents=True, exist_ok=True)

    logger.info("Running optimisation ...")
    pop, hof, log, history = optimizer.run(max_ngen=max_ngen, cp_filename=str(path))
    logger.info("Running optimisation ... Done.")

    return pop, hof, log, history
