"""Module for creation of emodels."""
import logging
import shutil
import os
import copy
from pathlib import Path

from bluepyemodel.evaluation import model
from bluepyemodel.evaluation import evaluator
from bluepyemodel.optimisation import optimisation
from bluepyemodel.feature_extraction import extract
from bluepyemodel.emodel_pipeline.utils import read_checkpoint
from bluepyemodel.emodel_pipeline import plotting

logger = logging.getLogger(__name__)


def connect_db(
    db_api,
    recipes_path=None,
    final_path=None,
    working_dir=None,
    project_name=None,
    forge_path=None,
):
    """Returns a DatabaseAPI object.

    Args:
        db_api (str): name of the api to use, can be 'sql', 'nexus'
            or 'singlecell'.
        recipe_path (str): path to the file containing the recipe. Only
            needed when working with db_api='singlecell'.
        working_dir (str): path of directory containing the parameters,
            features and parameters config files. Only needed when working
            with db_api='singlecell'.
        project_name (str): name of the project. Used as prefix to create the tables
            of the postgreSQL database.

    Returns:
        Database

    """
    if db_api == "sql":
        from bluepyemodel.api.postgreSQL import PostgreSQL_API

        return PostgreSQL_API(project_name=project_name)

    if db_api == "nexus":
        from bluepyemodel.api.nexus import Nexus_API

        return Nexus_API(forge_path)

    if db_api == "singlecell":
        from bluepyemodel.api.singlecell import Singlecell_API

        return Singlecell_API(
            working_dir=working_dir, recipes_path=recipes_path, final_path=final_path
        )

    raise Exception(f"Unknown api: {db_api}")


def copy_mechs(mechanism_paths, out_dir):
    """Copy mechamisms."""
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
    """Compile mechanisms"""
    p = Path(mechanisms_dir)

    if p.is_dir():
        if Path("x86_64").is_dir():
            os.popen("rm -rf x86_64").read()
        os.popen("nrnivmodl {}".format(mechanisms_dir)).read()

    else:
        raise Exception(
            "Cannot compile the mechanisms because 'mechanisms_dir'"
            " {} does not exist.".format(p)
        )


def get_responses(to_run):
    """Compute the responses from an evaluator and dict of parameters"""
    eva = to_run["evaluator"]
    params = to_run["parameters"]

    eva.evaluators[0].cell_model.unfreeze(params)

    return eva.evaluators[0].run_protocols(
        protocols=eva.evaluators[0].fitness_protocols.values(), param_values=params
    )


class EModel_pipeline:

    """EModel pipeline"""

    def __init__(
        self,
        emodel,
        species,
        db_api,
        working_dir,
        mechanisms_dir="mechanisms",
        recipes_path=None,
        project_name=None,
        final_path=None,
        forge_path=None,
    ):
        """Init

        Args:
            mechanisms_dir (str): directory to which the mechanisms have to be
                copied (has to be a subdirectory of working_dir).

        """

        self.emodel = emodel
        self.species = species

        if db_api not in ["sql", "nexus", "singlecell"]:
            raise Exception(
                "DB API {} does not exist. Must be 'sql', 'nexus' "
                "or 'singlecell'.".format(db_api)
            )
        self.db_api = db_api

        if recipes_path is None and self.db_api == "singlecell":
            raise Exception(
                "If using DB API 'singlecell', argument recipe_path has to be defined."
            )
        if working_dir is None and self.db_api == "singlecell":
            raise Exception(
                "If using DB API 'singlecell', argument working_dir has to be defined."
            )
        if project_name is None and self.db_api == "sql":
            raise Exception(
                "If using DB API 'sql', argument project_name has to be defined."
            )
        if forge_path is None and self.db_api == "nexus":
            raise Exception(
                "If using DB API 'sqnexusl', argument forge_path has to be defined."
            )

        self.mechanisms_dir = mechanisms_dir
        self.recipes_path = recipes_path
        self.working_dir = working_dir
        self.project_name = project_name
        self.final_path = final_path
        self.forge_path = forge_path

    def connect_db(self):
        """To instantiate the api from which to get and store the data."""
        return connect_db(
            self.db_api,
            recipes_path=self.recipes_path,
            final_path=self.final_path,
            working_dir=self.working_dir,
            project_name=self.project_name,
            forge_path=self.forge_path,
        )

    def get_evaluator(
        self,
        stochasticity=False,
        copy_mechanisms=False,
        compile_mechanisms=False,
        include_validation_protocols=False,
    ):
        """Create an evaluator for the emodel.

        Args:
            stochasticity (bool): should channels behave in a stochastic manner
                if they  can.
            copy_mechanisms (bool): should the mod files be copied in the working
                directory.
            include_validation_protocols (bool): should the validation protocols
                and efeatures be added to the evaluator

        Returns:
            MultiEvaluator
        """
        db = self.connect_db()

        # Get the data
        parameters, mechanisms, mechanism_names = db.get_parameters(
            self.emodel, self.species
        )
        if not (parameters) or not (mechanisms):
            raise Exception("No parameters for emodel %s" % self.emodel)

        morphologies = db.get_morphologies(self.emodel, self.species)
        if not (morphologies):
            raise Exception("No morphologies for emodel %s" % self.emodel)

        efeatures = db.get_features(
            self.emodel, self.species, include_validation_protocols
        )
        if not (efeatures):
            raise Exception("No efeatures for emodel %s" % self.emodel)

        protocols = db.get_protocols(
            self.emodel, self.species, include_validation_protocols
        )
        if not (protocols):
            raise Exception("No protocols for emodel %s" % self.emodel)

        db.close()

        if copy_mechanisms:
            mechanism_paths = db.get_mechanism_paths(mechanism_names)
            if not (mechanism_paths):
                raise Exception("No mechanisms paths for emodel %s" % self.emodel)
            copy_mechs(mechanism_paths, self.mechanisms_dir)

        if compile_mechanisms:
            compile_mechs(self.mechanisms_dir)

        cell_models = model.create_cell_models(
            emodel=self.emodel,
            working_dir=Path(self.working_dir),
            morphologies=morphologies,
            mechanisms=mechanisms,
            parameters=parameters,
        )

        return evaluator.create_evaluators(
            cell_models=cell_models,
            protocols_definition=protocols,
            features_definition=efeatures,
            stochasticity=stochasticity,
        )

    def extract_efeatures(
        self,
        file_format="axon",
        threshold_nvalue_save=1,
    ):
        """

        Args:
            file_format (str): format of the trace recordings
            threshold_nvalue_save (int): lower bounds of values required to return an efeatures

        Returns:
            efeatures (dict): mean efeatures and standard deviation
            stimuli (dict): definition of the mean stimuli
            current (dict): threshold and holding current
        """
        db = self.connect_db()

        map_function = optimisation.ipyparallel_map_function("USEIPYP")

        cells, protocols, protocols_threshold = db.get_extraction_metadata(
            self.emodel, self.species
        )
        if not (cells) or not (protocols):
            raise Exception("No extraction metadata for emodel %s" % self.emodel)

        config_dict = extract.get_config(
            cells,
            protocols,
            file_format=file_format,
            protocols_threshold=protocols_threshold,
            threshold_nvalue_save=threshold_nvalue_save,
        )

        efeatures, stimuli, current = extract.extract_efeatures(
            config_dict, self.emodel, map_function=map_function
        )

        db.store_efeatures(self.emodel, self.species, efeatures, current)
        db.store_protocols(self.emodel, self.species, stimuli)
        db.close()

        return efeatures, stimuli, current

    def optimize(
        self,
        max_ngen=1000,
        stochasticity=False,
        copy_mechanisms=False,
        compile_mechanisms=False,
        opt_params=None,
        optimizer="MO-CMA",
        checkpoint_path="./checkpoint.pkl",
        continue_opt=False,
    ):
        """
        Run optimisation

        Args:
            stochasticity (bool): should channels behave in a stochastic manner
                if they  can.
            copy_mechanisms (bool): should the mod files be copied in the working
                directory.
            compile_mechanisms (bool): should the mod files be compiled.
            opt_params (dict): optimisation parameters. Keys have to match the
                optimizer's call.
            optimizer (str): algorithm used for optimization, can be "IBEA",
                "SO-CMA", "MO-CMA"
            checkpoint_path (str): path to the checkpoint file.
            continue_opt (bool): should the optimization restart from a checkpoint .pkl.
        """
        if opt_params is None:
            opt_params = {}

        map_function = optimisation.ipyparallel_map_function("USEIPYP")

        _evaluator = self.get_evaluator(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
            include_validation_protocols=False,
        )

        p = Path(checkpoint_path)
        p = p.parents[0].mkdir(parents=True, exist_ok=True)
        if continue_opt and not (p.is_file()):
            raise Exception(
                "Continue_opt is True but the path specified in "
                "checkpoint_path does not exist."
            )

        opt = optimisation.setup_optimizer(
            _evaluator, map_function, opt_params, optimizer=optimizer
        )

        return optimisation.run_optimization(
            optimizer=opt,
            checkpoint_path=checkpoint_path,
            max_ngen=max_ngen,
            continue_opt=continue_opt,
        )

    def store_best_model(
        self,
        stochasticity=False,
        copy_mechanisms=False,
        compile_mechanisms=False,
        opt_params=None,
        optimizer="MO-CMA",
        checkpoint_path="./checkpoint.pkl",
    ):
        """
        Store the best model from an optimization

        Args:
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

        _evaluator = self.get_evaluator(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
            include_validation_protocols=False,
        )

        db = self.connect_db()

        run = read_checkpoint(checkpoint_path)

        best_model = run["halloffame"][0]
        feature_names = [
            obj.name for obj in _evaluator.evaluators[0].fitness_calculator.objectives
        ]
        param_names = list(_evaluator.param_names)

        scores = dict(zip(feature_names, best_model.fitness.values))
        params = dict(zip(param_names, best_model))

        db.store_model(
            self.emodel,
            scores,
            params,
            optimizer_name=optimizer,
            seed=opt_params["seed"],
            validated=False,
            species=self.species,
        )

        db.close()

    def validate(
        self,
        validation_function,
        stochasticity=False,
        copy_mechanisms=False,
        compile_mechanisms=False,
    ):
        """Compute the scores and traces for the optimisation and validation
        protocols and perform validation.
        """
        map_function = optimisation.ipyparallel_map_function("USEIPYP")

        _evaluator = self.get_evaluator(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
            include_validation_protocols=True,
        )

        db = self.connect_db()

        emodels = db.get_models(self.emodel, self.species)
        if emodels:

            logger.info(
                "In compute_scores, %s emodels found to evaluate.", len(emodels)
            )

            parameters = [
                mod["parameters"] for mod in emodels if mod["validated"] is False
            ]
            scores = map_function(_evaluator.evaluate_with_dicts, parameters)
            scores = list(scores)

            for mo, s in zip(emodels, scores):

                mo["scores"] = dict(s)
                validated = validation_function(mo)

                db.store_model(
                    emodel=self.emodel,
                    species=self.species,
                    scores=mo["scores"],
                    params=mo["parameters"],
                    optimizer_name=mo["optimizer"],
                    seed=mo["seed"],
                    validated=validated,
                )

        else:
            logger.warning(
                "In compute_scores, no emodel for %s %s", self.emodel, self.species
            )

    def compute_responses(
        self,
        stochasticity=False,
        copy_mechanisms=False,
        compile_mechanisms=False,
    ):
        """Return the responses of the optimisation and validation protocols."""
        # map_function = optimisation.ipyparallel_map_function("USEIPYP")

        _evaluator = self.get_evaluator(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
            include_validation_protocols=True,
        )

        db = self.connect_db()

        emodels = db.get_models(self.emodel, self.species)
        if emodels:

            logger.info(
                "In compute_responses, %s emodels found to evaluate.", len(emodels)
            )

            to_run = []
            for mo in emodels:
                to_run.append(
                    {
                        "evaluator": copy.deepcopy(_evaluator),
                        "parameters": mo["parameters"],
                    }
                )

            responses = list(map(get_responses, to_run))
            # responses = list(map_function(get_responses, to_run))

            for mo, r in zip(emodels, responses):
                mo["responses"] = r

        else:
            logger.warning(
                "In compute_responses, no emodel for %s %s", self.emodel, self.species
            )

        return emodels

    def plot_models(
        self,
        figures_dir="./figures",
        stochasticity=False,
        copy_mechanisms=False,
        compile_mechanisms=False,
    ):
        """Plot the traces and scores for all the models of this emodel."""
        _evaluator = self.get_evaluator(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
            include_validation_protocols=True,
        )

        emodels = self.compute_responses(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
        )

        stimuli = (
            _evaluator.evaluators[0].fitness_protocols["main_protocol"].subprotocols()
        )

        if emodels:
            for mo in emodels:
                plotting.scores(mo, figures_dir)
                plotting.traces(mo, mo["responses"], stimuli, figures_dir)

        else:
            logger.warning(
                "In plot_models, no emodel for %s %s", self.emodel, self.species
            )
