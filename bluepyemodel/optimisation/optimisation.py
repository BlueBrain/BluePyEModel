"""Optimisation function"""
import logging
import os
import shutil
from pathlib import Path

import bluepyopt

from bluepyemodel.evaluation import model
from bluepyemodel.evaluation.evaluator import create_evaluator

logger = logging.getLogger(__name__)


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


def run_optimization(optimizer, checkpoint_path, max_ngen, continue_opt):
    """Run the optimisation.

    Args:
        optimizer (DEAPOptimisation): optimiser used for the run.
        checkpoint_dir (Path): path to which the checkpoint will be saved.
        max_ngen (int): maximum number of generation for which the
            evolutionary strategy will run.

    Returns:
        None
    """
    checkpoint_path.parents[0].mkdir(parents=True, exist_ok=True)

    logger.info("Running optimisation ...")
    pop, hof, log, history = optimizer.run(
        max_ngen=max_ngen, cp_filename=str(checkpoint_path), continue_cp=continue_opt
    )
    logger.info("Running optimisation ... Done.")

    return pop, hof, log, history


def copy_mechs(mechanism_paths, out_dir):
    """Copy mod files in the designated directory.

    Args:
        mechanism_paths (list): list of the paths to the mod files that
            have to be copied.
        out_dir (str): path to directory to which the mod files should
            be copied.
    """

    if mechanism_paths:

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for m in mechanism_paths:

            p = Path(m["path"])

            if p.is_file():
                new_p = out_dir / p.name
                shutil.copy(str(p), str(new_p))

            else:
                raise Exception(
                    "Cannot copy the .mod files locally because the "
                    "'mechanism_paths' {} does not exist.".format(p)
                )


def compile_mechs(mechanisms_dir):
    """Compile the mechanisms.

    Args:
        mechanisms_dir (str): path to the directory containing the
            mod files to compile.
    """

    path_mechanisms_dir = Path(mechanisms_dir)

    if path_mechanisms_dir.is_dir():

        if Path("x86_64").is_dir():
            os.popen("rm -rf x86_64").read()
        os.popen("nrnivmodl {}".format(path_mechanisms_dir)).read()

    else:
        raise Exception(
            "Cannot compile the mechanisms because 'mechanisms_dir':"
            " {} does not exist.".format(str(path_mechanisms_dir))
        )


def _get_evaluator_from_db(
    emodel,
    species,
    db,
    mechanisms_dir=None,
    morphology_modifiers=None,
    stochasticity=False,
    copy_mechanisms=False,
    compile_mechanisms=False,
    include_validation_protocols=False,
    additional_protocols=None,
    optimisation_rules=None,
    timeout=600,
    githash=None,
):
    """Create an evaluator for the emodel.

    Args:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
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
        optimisation_rules (list): list of Rules. TO DEPRECATE: should be done
            in the api.

    Returns:
        bluepyopt.ephys.evaluators.CellEvaluator
    """
    if compile_mechanisms and githash:
        raise Exception(
            "Compile mechanisms and the use of githash are not compatible "
            "yet. Please pre-compile the mechanisms and re-run with "
            "compile_mechanisms=False."
        )

    # Get the data
    parameters, mechanisms, mechanism_names = db.get_parameters(emodel, species)
    if not (parameters) or not (mechanisms):
        raise Exception("No parameters for emodel %s" % emodel)

    morphologies = db.get_morphologies(emodel, species)
    if not (morphologies):
        raise Exception("No morphologies for emodel %s" % emodel)

    efeatures = db.get_features(emodel, species, include_validation=include_validation_protocols)
    if not (efeatures):
        raise Exception("No efeatures for emodel %s" % emodel)

    protocols = db.get_protocols(emodel, species, include_validation=include_validation_protocols)
    if not (protocols):
        raise Exception("No protocols for emodel %s" % emodel)
    if additional_protocols:
        protocols.update(additional_protocols)

    if copy_mechanisms:
        mechanism_paths = db.get_mechanism_paths(mechanism_names)
        if not (mechanism_paths):
            raise Exception("No mechanisms paths for emodel %s" % emodel)
        copy_mechs(mechanism_paths, mechanisms_dir)

    if compile_mechanisms:
        compile_mechs(mechanisms_dir)

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
        optimisation_rules=optimisation_rules,
        timeout=timeout,
    )


def setup_and_run_optimisation(  # pylint: disable=too-many-arguments
    emodel_db,
    emodel,
    seed,
    species=None,
    mechanisms_dir=None,
    morphology_modifiers=None,
    stochasticity=False,
    copy_mechanisms=False,
    compile_mechanisms=False,
    include_validation_protocols=False,
    optimisation_rules=None,
    timeout=None,
    mapper=None,
    opt_params=None,
    optimizer="MO-CMA",
    max_ngen=1000,
    checkpoint_dir=None,
    continue_opt=False,
):
    emodel_db.set_seed(emodel, seed, species=species)
    _evaluator = _get_evaluator_from_db(
        emodel=emodel,
        species=species,
        db=emodel_db,
        mechanisms_dir=mechanisms_dir,
        morphology_modifiers=morphology_modifiers,
        stochasticity=stochasticity,
        copy_mechanisms=copy_mechanisms,
        compile_mechanisms=compile_mechanisms,
        include_validation_protocols=include_validation_protocols,
        optimisation_rules=optimisation_rules,
        timeout=timeout,
    )

    opt = setup_optimizer(_evaluator, mapper, opt_params, optimizer=optimizer)

    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{emodel}_{seed}.pkl"

    run_optimization(
        optimizer=opt,
        checkpoint_path=checkpoint_path,
        max_ngen=max_ngen,
        continue_opt=continue_opt,
    )
