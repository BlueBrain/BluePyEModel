"""Module for creation of emodels."""
import logging
from pathlib import Path
import shutil
import os

from bluepyemodel.evaluation import model
from bluepyemodel.evaluation import evaluator
from bluepyemodel.optimisation import optimisation
from bluepyemodel.feature_extraction import extract

logger = logging.getLogger(__name__)


def connect_db(db_api, recipe_path=None, working_dir=None, forge_path=None):
    """Returns a DatabaseAPI object.

    Args:
        db_api (str): name of the api to use, can be 'sql', 'nexus'
            or 'singlecell'.
        recipe_path (str): path to the file containing the recipe. Only
            needed when working with db_api='singlecell'.
        working_dir (str): path of directory containing the parameters,
            features and parameters config files. Only needed when working
            with db_api='singlecell'.

    Returns:
        Database

    """
    if db_api == "sql":
        from bluepyemodel.api.postgreSQL import PostgreSQL_API

        return PostgreSQL_API()

    if db_api == "nexus":
        from bluepyemodel.api.nexus import Nexus_API

        return Nexus_API(forge_path)

    if db_api == "singlecell":
        from bluepyemodel.api.singlecell import Singlecell_API

        return Singlecell_API(recipe_path, working_dir)

    raise Exception(f"Unknown api: {db_api}")


def extract_efeatures(  # pylint: disable=dangerous-default-value
    emodel,
    species,
    db_api,
    file_format="axon",
    threshold_nvalue_save=1,
):
    """

    Args:
        emodel (str): name of the emodel.
        species (str): name of the species (rat, human, mouse).
        db_api (str): name of the api to use, can be 'sql', 'nexus'
            or 'singlecell'.
        file_format (str): format of the trace recordings

    Returns:
        efeatures (dict): mean efeatures and standard deviation
        stimuli (dict): definition of the mean stimuli
        current (dict): threshold and holding current
    """
    db = connect_db(db_api)

    map_function = optimisation.ipyparallel_map_function("USEIPYP")

    cells, protocols, protocols_threshold = db.get_extraction_metadata(emodel, species)
    if not (cells) or not (protocols):
        raise Exception("No extraction metadata for emodel %s" % emodel)

    config_dict = extract.get_config(
        cells,
        protocols,
        file_format=file_format,
        protocols_threshold=protocols_threshold,
        threshold_nvalue_save=threshold_nvalue_save,
    )

    efeatures, stimuli, current = extract.extract_efeatures(
        config_dict, emodel, map_function=map_function
    )

    # Fill database with efeatures and protocols
    db.store_efeatures(emodel, species, efeatures, current)
    db.store_protocols(emodel, species, stimuli)

    db.close()

    return efeatures, stimuli, current


def copy_mechs(mechanism_paths, out_dir):
    """Copyy mechamisms."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for m in mechanism_paths:
        p = Path(m["path"])
        new_p = out_dir / p.name
        shutil.copy(str(p), str(new_p))


def get_evaluator(  # pylint: disable=too-many-arguments
    emodel,
    species,
    db_api,
    working_dir,
    mechanisms_dir,
    stochasticity=False,
    copy_mechanisms=False,
):
    """Create an evaluator for a cell model name and database access.

    Args:
        emodel (str): name of the emodel.
        species (str): name of the species (rat, human, mouse).
        db_api (str): name of the api to use, can be 'sql', 'nexus'
            or 'singlecell'.
        working_dir (str): directory in which the optimisation will take place.
        mechanisms_dir (str): directory to which the mechanisms have to be
            copied (has to be a subdirectory of working_dir).
        stochasticity (bool): should channels behave in a stochastic manner
            if they  can.
        copy_mechanisms (bool): should the mod files be copied in the working
            directory.

    Returns:
        MultiEvaluator
    """
    db = connect_db(db_api)

    # Get the data
    parameters, mechanisms, mechanism_names = db.get_parameters(emodel, species)
    if not (parameters) or not (mechanisms):
        raise Exception("No parameters for emodel %s" % emodel)

    mechanism_paths = db.get_mechanism_paths(mechanism_names)
    if not (mechanism_paths):
        raise Exception("No mechanisms paths for emodel %s" % emodel)

    morphologies = db.get_morphologies(emodel, species)
    if not (morphologies):
        raise Exception("No morphologies for emodel %s" % emodel)

    efeatures = db.get_features(emodel, species)
    if not (efeatures):
        raise Exception("No efeatures for emodel %s" % emodel)

    protocols = db.get_protocols(emodel, species)
    if not (protocols):
        raise Exception("No protocols for emodel %s" % emodel)

    db.close()

    # Copy the mechanisms and compile them
    if copy_mechanisms:
        if mechanism_paths:
            copy_mechs(mechanism_paths, mechanisms_dir)
        else:
            raise Exception(
                "Trying to copy mod files locally but " "'mechanism_paths' is missing."
            )

    os.popen("rm -rf x86_64").read()
    os.popen("nrnivmodl {}".format(mechanisms_dir)).read()

    # Create the cell models
    cell_models = model.create_cell_models(
        emodel=emodel,
        working_dir=Path(working_dir),
        morphologies=morphologies,
        mechanisms=mechanisms,
        parameters=parameters,
    )

    # Create evaluators
    return evaluator.create_evaluators(
        cell_models=cell_models,
        protocols_definition=protocols,
        features_definition=efeatures,
        stochasticity=stochasticity,
    )


def optimize(  # pylint: disable=too-many-arguments
    emodel,
    species,
    db_api,
    working_dir,
    mechanisms_dir,
    max_ngen=1000,
    stochasticity=False,
    copy_mechanisms=True,
    opt_params=None,
    optimizer="MO-CMA",
    checkpoint_path="./checkpoint.pkl",
):
    """
    Run optimisation

    Args:
        emodel (str): name of the emodel.
        species (str): name of the species (rat, human, mouse).
        db_api (str): name of the api to use, can be 'sql', 'nexus'
            or 'singlecell'.
        working_dir (str): directory in which the optimisation will take place.
        mechanisms_dir (str): directory to which the mechanisms have to be
            copied (has to be a subdirectory of working_dir).
        stochasticity (bool): should channels behave in a stochastic manner
            if they  can.
        copy_mechanisms (bool): should the mod files be copied in the working
            directory.
        opt_params (dict): optimisation parameters. Keys have to match the
            optimizer's call.
        optimizer (str): algorithm used for optimization, can be "IBEA",
            "SO-CMA", "MO-CMA"
        checkpoint_path (str): path to the checkpoint file.
    """
    if opt_params is None:
        opt_params = {}

    map_function = optimisation.ipyparallel_map_function("USEIPYP")

    _evaluator = get_evaluator(
        emodel,
        species,
        db_api,
        working_dir,
        mechanisms_dir,
        stochasticity=stochasticity,
        copy_mechanisms=copy_mechanisms,
    )

    opt = optimisation.setup_optimizer(
        _evaluator, map_function, opt_params, optimizer=optimizer
    )

    optimisation.run_optimization(
        optimizer=opt, checkpoint_path=checkpoint_path, max_ngen=max_ngen
    )


def store_optimize(  # pylint: disable=too-many-arguments
    emodel,
    species,
    db_api,
    working_dir,
    mechanisms_dir,
    stochasticity=False,
    opt_params=None,
    optimizer="MO-CMA",
    checkpoint_path="./checkpoint.pkl",
):
    """
    Store the results of an optimization

    Args:
        emodel (str): name of the emodel.
        species (str): name of the species (rat, human, mouse).
        db_api (str): name of the api to use, can be 'sql', 'nexus'
            or 'singlecell'.
        working_dir (str): directory in which the optimisation will take place.
        mechanisms_dir (str): directory to which the mechanisms have to be
            copied (has to be a subdirectory of working_dir).
        stochasticity (bool): should channels behave in a stochastic manner
            if they  can.
        opt_params (dict): optimisation parameters. Keys have to match the
            optimizer's call.
        optimizer (str): algorithm used for optimization, can be "IBEA",
            "SO-CMA", "MO-CMA"
        checkpoint_path (str): path to the checkpoint file.
    """
    if opt_params is None:
        opt_params = {}

    _evaluator = get_evaluator(
        emodel,
        species,
        db_api,
        working_dir,
        mechanisms_dir,
        stochasticity=stochasticity,
        copy_mechanisms=False,
    )
    db = connect_db(db_api)

    feature_names = [
        obj.name for obj in _evaluator.evaluators[0].fitness_calculator.objectives
    ]
    param_names = list(_evaluator.param_names)

    db.store_model(
        emodel=emodel,
        species=species,
        checkpoint_path=checkpoint_path,
        param_names=param_names,
        feature_names=feature_names,
        optimizer_name=optimizer,
        opt_params=opt_params,
        validated=False,
    )

    db.close()
