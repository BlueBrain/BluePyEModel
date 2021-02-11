"""Define a class for a pipeline allowing the creation of emodels."""
import copy
import datetime
import logging
import os
import shutil
import tarfile
import time
from pathlib import Path

import bluepyefe.extract
from git import Repo

from bluepyemodel.emodel_pipeline import plotting
from bluepyemodel.emodel_pipeline.utils import read_checkpoint
from bluepyemodel.evaluation import evaluator
from bluepyemodel.evaluation import model
from bluepyemodel.optimisation import optimisation
from bluepyemodel.validation import validation

logger = logging.getLogger(__name__)

mechanisms_dir = "mechanisms"
run_dir = "run"
final_path = "final.json"


def connect_db(
    db_api,
    recipes_path=None,
    working_dir=None,
    project_name=None,
    forge_path=None,
):
    """Returns a DatabaseAPI object.

    Args:
        db_api (str): name of the api to use, can be 'sql', 'nexus'
            or 'singlecell'.
        recipe_path (str): path to the file containing the recipes. Only
            needed when working with db_api='singlecell'.
        working_dir (str): path of the directory containing the parameters,
            features and parameters config files. Only needed when working
            with db_api='singlecell'.
        project_name (str): name of the project. Used as prefix to create the tables
            of the postgreSQL database. Only needed when working with db_api='sql'.

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
            emodel_dir=working_dir, recipes_path=recipes_path, final_path=final_path
        )

    raise Exception(f"Unknown api: {db_api}")


def extract_save_features_protocols(
    emodel_db,
    emodel,
    species=None,
    files_metadata=None,
    targets=None,
    protocols_threshold=None,
    threshold_nvalue_save=1,
    mapper=map,
    name_Rin_protocol=None,
    name_rmp_protocol=None,
    validation_protocols=None,
):

    if files_metadata is None or targets is None or protocols_threshold is None:

        files_metadata, targets, protocols_threshold = emodel_db.get_extraction_metadata(
            emodel, species, threshold_nvalue_save
        )

        if files_metadata is None or targets is None or protocols_threshold is None:
            raise Exception("Could not get the extraction metadata from the api.")

    # extract features
    efeatures, stimuli, current = bluepyefe.extract.extract_efeatures(
        output_directory=emodel,
        files_metadata=files_metadata,
        targets=targets,
        strict_stiminterval=True,
        threshold_nvalue_save=threshold_nvalue_save,
        protocols_threshold=protocols_threshold,
        ap_threshold=-20.0,
        trace_reader=None,
        map_function=mapper,
        write_files=False,
        plot=False,
    )

    # store features & protocols
    emodel_db.store_efeatures(
        emodel,
        species,
        efeatures,
        current,
        name_Rin_protocol,
        name_rmp_protocol,
        validation_protocols,
    )
    emodel_db.store_protocols(emodel, species, stimuli, validation_protocols)

    return efeatures, stimuli, current


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


def compile_mechs(working_dir, githash=None):
    """Compile the mechanisms and copy them in the cwd.

    Args:
        working_dir (str): directory in which to compile and take the mechanisms.
    """

    path_mechanisms_dir = Path(working_dir) / mechanisms_dir

    if path_mechanisms_dir.is_dir():

        if Path("x86_64").is_dir():
            os.popen("rm -rf x86_64").read()

        if githash:
            os.chdir(working_dir)
            os.popen("nrnivmodl mechanisms").read()
            os.chdir("../../")
            os.popen("cp -r {} x86_64".format(str(Path(working_dir) / "x86_64"))).read()
        else:
            os.popen("nrnivmodl mechanisms").read()

    else:
        raise Exception(
            "Cannot compile the mechanisms because 'mechanisms_dir':"
            " {} does not exist.".format(str(path_mechanisms_dir))
        )


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


def update_gitignore():
    """
    Adds the following lines to .gitignore: 'run/', 'checkpoints/', 'figures/',
    'logs/', '.ipython/', '.ipynb_checkpoints/'
    """

    path_gitignore = Path("./.gitignore")

    if not (path_gitignore.is_file()):
        raise Exception("Could not update .gitignore as it does not exist.")

    with open(str(path_gitignore), "r") as f:
        lines = f.readlines()

    lines = " ".join(line for line in lines)

    to_add = [
        run_dir + "/",
        "checkpoints/",
        "figures/",
        "logs/",
        ".ipython/",
        ".ipynb_checkpoints/",
    ]

    with open(str(path_gitignore), "a") as f:
        for a in to_add:
            if a not in lines:
                f.write(f"{a}\n")


def generate_versions():
    """
    Save a list of the versions of the python packages in the current environnement.
    """

    path_versions = Path("./list_versions.log")

    if path_versions.is_file():
        logger.warning("%s already exists and will overwritten.", path_versions)

    os.popen(f"pip list > {str(path_versions)}").read()


def generate_githash():
    """
    Generate a githash and create the associated run directory
    """
    path_run = Path(run_dir)

    if not path_run.is_dir():
        logger.warning("Directory %s does not exist and will be created.", str(path_run))
        path_run.mkdir(parents=True, exist_ok=True)

    while Path("./.git/index.lock").is_file():
        time.sleep(5.0)
        logger.info("emodel_pipeline waiting for ./.git/index.lock.")

    repo = Repo("./")
    changedFiles = [item.a_path for item in repo.index.diff(None)]
    if changedFiles:
        os.popen('git add -A && git commit --allow-empty -a -m "Running pipeline"').read()

    githash = os.popen("git rev-parse --short HEAD").read()[:-1]

    tar_source = path_run / f"{githash}.tar"
    tar_target = path_run / githash

    if not tar_target.is_dir():

        logger.info("New githash directory: %s", githash)

        repo = Repo("./")
        with open(str(tar_source), "wb") as fp:
            repo.archive(fp)
        with tarfile.open(str(tar_source)) as tf:
            tf.extractall(str(tar_target))

        if tar_source.is_file():
            os.popen(f"rm -rf {str(tar_source)}").read()

    else:
        logger.info("Working from existing githash directory: %s", githash)

    return githash


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

    if flag_use_IPYP and os.getenv(flag_use_IPYP) and int(os.getenv(flag_use_IPYP)):
        from ipyparallel import Client

        rc = Client(profile=os.getenv(ipython_profil))
        lview = rc.load_balanced_view()

        def mapper(func, it):
            start_time = datetime.datetime.now()
            ret = lview.map_sync(func, it)
            logger.debug("Took %s", datetime.datetime.now() - start_time)
            return ret

    else:
        mapper = map

    return mapper


class EModel_pipeline:

    """EModel pipeline"""

    def __init__(
        self,
        emodel,
        species,
        db_api,
        recipes_path=None,
        project_name=None,
        forge_path=None,
        use_git=False,
        githash=None,
        morphology_modifiers=None,
    ):
        """Initialize the emodel_pipeline.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            species (str): name of the species.
            db_api (str): name of the API used to access the data, can be "sql",
                "nexus" or "singlecell". "singlecell" expect the configuration to be
                defined in a "config" directory containing recipes as in proj38. "sql"
                expect the configuration to be defined in the PostreSQL table whose
                name can be find ontop of the file bluepyemodel/api/sql.py. "nexus"
                expect the configuration to be defined on Nexus using NexusForge,
                see bluepyemodel/api/nexus.py.
            recipes_path (str): path of the recipes.json, only needed if
                db_api="singlecell".
            project_name (str): name of the project, only needed if db_api="sql".
            forge_path (str): path to the .yml used to connect to Nexus Forge,
                only needed if db_api="nexus".
            use_git (bool): if True, work will be perfomed in a sub-directory:
                working_dir/run/githash. If use_git is True and githash is None,
                a new githash will be generated. The pipeline will expect a git
                repository to exist in working_dir.
            githash (str): if provided, the pipeline will work in the directory
                working_dir/run/githash. Needed when continuing work or resuming
                optimisations.
            morphology_modifiers (list): list of python functions that will be
                applied to all the morphologies.

        """

        self.emodel = emodel
        self.species = species

        if db_api not in ["sql", "nexus", "singlecell"]:
            raise Exception(
                "DB API {} does not exist. Must be 'sql', 'nexus' "
                "or 'singlecell'.".format(db_api)
            )
        self.db_api = db_api

        if project_name is None and self.db_api == "sql":
            raise Exception("If using DB API 'sql', argument project_name has to be defined.")
        if forge_path is None and self.db_api == "nexus":
            raise Exception("If using DB API 'sqnexusl', argument forge_path has to be defined.")

        self.morphology_modifiers = morphology_modifiers
        self.recipes_path = recipes_path
        self.project_name = project_name
        self.forge_path = forge_path

        if use_git:

            is_git = str(os.popen("git rev-parse --is-inside-work-tree").read())[:-1]
            if is_git != "true":
                raise Exception(
                    "use_git is true, but there is no git repository initialized in working_dir."
                )

            if githash is None:

                update_gitignore()
                # generate_version has to be ran before generating the githash as a change
                # in a package version should induce the creation of a new githash
                generate_versions()
                self.githash = generate_githash()

            else:

                self.githash = githash
                logger.info("Working from existing githash directory: %s", self.githash)

        elif githash:
            raise Exception("A githash was provided but use_git is False.")

        else:
            self.githash = None

    @property
    def working_dir(self):
        if self.githash:
            return str(Path("./") / run_dir / self.githash)
        return "./"

    def connect_db(self):
        """
        Instantiate the api from which the pipeline will get and store the data.
        """
        return connect_db(
            self.db_api,
            recipes_path=self.recipes_path,
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
        additional_protocols=None,
        optimisation_rules=None,
        timeout=600,
    ):
        """Create an evaluator for the emodel.

        Args:
            stochasticity (bool): should channels behave stochastically if they can.
            copy_mechanisms (bool): should the mod files be copied in the local
                mechanisms_dir directory.
            compile_mechanisms (bool): should the mod files be compiled.
            include_validation_protocols (bool): should the validation protocols
                and validation efeatures be added to the evaluator.
            additional_protocols (dict): definition of supplementary protocols. See
                examples/optimisation for usage.
            optimisation_rules (list): list of Rules. TO DEPRECATE: should be done
                in the api.
            timeout (float): duration (in second) after which the evaluation of a
                protocol will be interrupted.

        Returns:
            MultiEvaluator

        """
        db = self.connect_db()

        if compile_mechanisms and self.githash:
            raise Exception(
                "Compile mechanisms and the use of githash are not compatible "
                "yet. Please pre-compile the mechanisms and re-run with "
                "compile_mechanisms=False."
            )

        # Get the data
        parameters, mechanisms, mechanism_names = db.get_parameters(self.emodel, self.species)
        if not (parameters) or not (mechanisms):
            raise Exception("No parameters for emodel %s" % self.emodel)

        morphologies = db.get_morphologies(self.emodel, self.species)
        if not (morphologies):
            raise Exception("No morphologies for emodel %s" % self.emodel)

        efeatures = db.get_features(
            self.emodel, self.species, include_validation=include_validation_protocols
        )
        if not (efeatures):
            raise Exception("No efeatures for emodel %s" % self.emodel)

        protocols = db.get_protocols(
            self.emodel, self.species, include_validation=include_validation_protocols
        )
        if not (protocols):
            raise Exception("No protocols for emodel %s" % self.emodel)
        if additional_protocols:
            protocols.update(additional_protocols)
        db.close()

        if copy_mechanisms:
            mechanism_paths = db.get_mechanism_paths(mechanism_names)
            if not (mechanism_paths):
                raise Exception("No mechanisms paths for emodel %s" % self.emodel)
            copy_mechs(mechanism_paths, mechanisms_dir)

        if compile_mechanisms:
            compile_mechs(self.working_dir, self.githash)

        cell_models = model.create_cell_models(
            emodel=self.emodel,
            morphologies=morphologies,
            mechanisms=mechanisms,
            parameters=parameters,
            morph_modifiers=self.morphology_modifiers,
        )

        return evaluator.create_evaluator(
            cell_model=cell_models[0],
            protocols_definition=protocols,
            features_definition=efeatures,
            stochasticity=stochasticity,
            optimisation_rules=optimisation_rules,
            timeout=timeout,
        )

    def extract_efeatures(
        self,
        files_metadata=None,
        targets=None,
        threshold_nvalue_save=1,
        protocols_threshold=None,
        name_Rin_protocol=None,
        name_rmp_protocol=None,
        validation_protocols=None,
    ):
        """Extract efeatures from traces using BluePyEfe 2. Extraction is performed
            as defined in the argument "config_dict". See example/efeatures_extraction
            for usage. The resulting efeatures, protocols and currents will be saved
            in the medium chosen by the api.

        Args:
            files_metadata (dict): define for which cell and protocol each file
                has to be used. Of the form:
                {
                    cell_id: {
                        protocol_name: [{file_metadata1}, {file_metadata1}]
                    }
                }
                A same file path might be present in the file metadata for
                different protocols.
                The entries required in the file_metadata are specific to each
                trace_reader (see bluepyemodel/reader.py to know which one are
                needed for your trace_reader).
            targets (dict): define the efeatures to extract for each protocols
                and the amplitude around which these features should be
                averaged. Of the form:
                {
                    protocol_name: {
                        "amplitudes": [50, 100],
                        "tolerances": [10, 10],
                        "efeatures": ["Spikecount", "AP_amplitude"],
                        "location": "soma"
                    }
                }
                If efeatures must be computed only for a given time interval,
                the beginning and end of this interval can be specified as
                follows (in ms):
                "efeatures": {
                    "Spikecount": [500, 1100],
                    "AP_amplitude": [100, 600],
                }
            threshold_nvalue_save (int): lower bounds of the number of values required
                to save an efeature.
            protocols_threshold (list): names of the protocols that will be
                used to compute the rheobase of the cells. E.g: ['IDthresh'].
            name_Rin_protocol (str): name of the protocol that should be used to compute
                the input resistance. Only used when db_api is 'singlecell'
            name_rmp_protocol (str): name of the protocol that should be used to compute
                the resting membrane potential. Only used when db_api is 'singlecell'.
            validation_protocols (dict): Of the form {"ecodename": [targets]}. Only used
                when db_api is 'singlecell'.

        Returns:
            efeatures (dict): mean and standard deviation of efeatures at targets.
            stimuli (dict): definitions of mean stimuli/protocols.
            current (dict): mean and standard deviation of holding and threshold currents.
        """

        if validation_protocols is None:
            validation_protocols = {}
        if protocols_threshold is None:
            protocols_threshold = []

        db = self.connect_db()

        map_function = ipyparallel_map_function("USEIPYP")

        efeatures, stimuli, current = extract_save_features_protocols(
            emodel=self.emodel,
            emodel_db=db,
            species=self.species,
            files_metadata=files_metadata,
            targets=targets,
            protocols_threshold=protocols_threshold,
            threshold_nvalue_save=threshold_nvalue_save,
            mapper=map_function,
            name_Rin_protocol=name_Rin_protocol,
            name_rmp_protocol=name_rmp_protocol,
            validation_protocols=validation_protocols,
        )

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
        checkpoint_path=None,
        continue_opt=False,
        optimisation_rules=None,
        timeout=600,
    ):
        """Run optimisation.

        Args:
            max_ngen (int): maximum number of generations of the evolutionary process.
            stochasticity (bool): should channels behave stochastically if they can.
            copy_mechanisms (bool): should the mod files be copied in the local
                mechanisms_dir directory.
            compile_mechanisms (bool): should the mod files be compiled.
            opt_params (dict): optimisation parameters. Keys have to match the
                optimizer's call.
            optimizer (str): algorithm used for optimization, can be "IBEA", "SO-CMA",
                "MO-CMA".
            checkpoint_path (str): path to the file used as a checkpoint by BluePyOpt.
            continue_opt (bool): should the optimization restart from a previously
                created checkpoint file.
            timeout (float): duration (in second) after which the evaluation of a
                protocol will be interrupted.

        """
        if opt_params is None:
            opt_params = {}

        if checkpoint_path is None:

            checkpoint_path = f"./checkpoints/emodel:{self.emodel}"

            if self.githash:
                checkpoint_path += f"__githash:{self.githash}"

            if "seed" in opt_params:
                checkpoint_path += f"__seed:{opt_params['seed']}.pkl"
            else:
                checkpoint_path += ".pkl"

        logger.info("Path to the checkpoint file is %s", checkpoint_path)

        map_function = ipyparallel_map_function("USEIPYP")

        _evaluator = self.get_evaluator(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
            include_validation_protocols=False,
            optimisation_rules=optimisation_rules,
            timeout=timeout,
        )

        p = Path(checkpoint_path)
        p.parents[0].mkdir(parents=True, exist_ok=True)
        if continue_opt and not (p.is_file()):
            raise Exception(
                "continue_opt is True but the path specified in checkpoint_path does not exist."
            )

        opt = optimisation.setup_optimizer(
            _evaluator, map_function, opt_params, optimizer=optimizer
        )

        return optimisation.run_optimization(
            optimizer=opt,
            checkpoint_path=Path(checkpoint_path),
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
        """Store the best model from an optimization. Reads a checkpoint file generated
            by BluePyOpt and store the best individual of the hall of fame.

        Args:
            stochasticity (bool): should channels behave stochastically if they can.
            opt_params (dict): optimisation parameters. Keys have to match the
                optimizer's call.
            optimizer (str): algorithm used for optimization, can be "IBEA", "SO-CMA",
                "MO-CMA".
            checkpoint_path (str): path to the BluePyOpt checkpoint file from which
                the best model will be read and stored.
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
        feature_names = [obj.name for obj in _evaluator.fitness_calculator.objectives]
        param_names = list(_evaluator.param_names)

        scores = dict(zip(feature_names, best_model.fitness.values))
        params = dict(zip(param_names, best_model))

        seed = opt_params.get("seed", "")

        db.store_emodel(
            self.emodel,
            scores,
            params,
            optimizer_name=optimizer,
            seed=seed,
            githash=self.githash,
            validated=False,
            species=self.species,
        )

        db.close()

    def compute_responses(
        self,
        stochasticity=False,
        copy_mechanisms=False,
        compile_mechanisms=False,
        additional_protocols=None,
    ):
        """Compute the responses of the emodel to the optimisation and validation protocols.

        Args:
            stochasticity (bool): should channels behave stochastically if they can.
            copy_mechanisms (bool): should the mod files be copied in the local
                mechanisms_dir directory.
            compile_mechanisms (bool): should the mod files be compiled.
            additional_protocols (dict): definition of supplementary protocols. See
                examples/optimisation for usage.

        Returns:
            emodels (list): list of emodels.
        """

        map_function = ipyparallel_map_function("USEIPYP")

        if additional_protocols is None:
            additional_protocols = {}

        _evaluator = self.get_evaluator(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
            include_validation_protocols=True,
            additional_protocols=additional_protocols,
        )

        db = self.connect_db()

        emodels = db.get_emodels([self.emodel], self.species)
        if emodels:

            logger.info("In compute_responses, %s emodels found to evaluate.", len(emodels))

            to_run = []
            for mo in emodels:
                to_run.append(
                    {
                        "evaluator": copy.deepcopy(_evaluator),
                        "parameters": mo["parameters"],
                    }
                )

            responses = list(map_function(get_responses, to_run))

            for mo, r in zip(emodels, responses):
                mo["responses"] = r

        else:
            logger.warning("In compute_responses, no emodel for %s %s", self.emodel, self.species)

        return emodels

    def validate(
        self,
        validation_function=None,
        stochasticity=False,
        copy_mechanisms=False,
        compile_mechanisms=False,
    ):
        """Compute the scores and traces for the optimisation and validation
        protocols and perform validation.

        Args:
            validation_function (python function): function used to decide if a model
                passes validation or not. Should rely on emodel['scores'] and
                emodel['scores_validation']. See bluepyemodel/validation for examples.
            stochasticity (bool): should channels behave stochastically if they can.
            copy_mechanisms (bool): should the mod files be copied in the local
                mechanisms_dir directory.
            compile_mechanisms (bool): should the mod files be compiled.

        Returns:
            emodels (list): list of emodels.
        """
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
            additional_protocols=None,
        )

        db = self.connect_db()

        if emodels:

            if validation_function is None:
                logger.warning("Validation function not  specified, will use max_score.")
                validation_function = validation.max_score

            name_validation_protocols = db.get_name_validation_protocols(self.emodel, self.species)

            logger.info("In validate, %s emodels found to validate.", len(emodels))

            for mo in emodels:

                mo["scores"] = _evaluator.fitness_calculator.calculate_scores(mo["responses"])

                mo["scores_validation"] = {}
                for feature_names, score in mo["scores"].items():
                    for p in name_validation_protocols:
                        if p in feature_names:
                            mo["scores_validation"][feature_names] = score

                validated = validation_function(mo)

                db.store_emodel(
                    emodel=self.emodel,
                    species=self.species,
                    scores=mo["scores"],
                    params=mo["parameters"],
                    optimizer_name=mo["optimizer"],
                    seed=mo["seed"],
                    githash=mo["githash"],
                    validated=validated,
                    scores_validation=mo["scores_validation"],
                )

            return emodels

        logger.warning("In compute_scores, no emodel for %s %s", self.emodel, self.species)
        return []

    def plot_models(
        self,
        figures_dir="./figures",
        stochasticity=False,
        copy_mechanisms=False,
        compile_mechanisms=False,
        additional_protocols=None,
    ):
        """Plot the traces, scores and parameter distributions for all the models
            matching the emodels name.

        Args:
            figures_dir (str): path of the directory in which the figures should be saved.
            stochasticity (bool): should channels behave stochastically if they can.
            copy_mechanisms (bool): should the mod files be copied in the local
                mechanisms_dir directory.
            compile_mechanisms (bool): should the mod files be compiled.
            additional_protocols (dict): definition of supplementary protocols. See
                examples/optimisation for usage.

        Returns:
            emodels (list): list of emodels.
        """
        if additional_protocols is None:
            additional_protocols = {}

        _evaluator = self.get_evaluator(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
            include_validation_protocols=True,
            additional_protocols=additional_protocols,
        )

        emodels = self.compute_responses(
            stochasticity=stochasticity,
            copy_mechanisms=copy_mechanisms,
            compile_mechanisms=compile_mechanisms,
            additional_protocols=additional_protocols,
        )

        stimuli = _evaluator.fitness_protocols["main_protocol"].subprotocols()

        if emodels:

            lbounds = {
                p.name: p.bounds[0]
                for p in _evaluator.cell_model.params.values()
                if p.bounds is not None
            }
            ubounds = {
                p.name: p.bounds[1]
                for p in _evaluator.cell_model.params.values()
                if p.bounds is not None
            }

            plotting.parameters_distribution(
                models=emodels,
                lbounds=lbounds,
                ubounds=ubounds,
                figures_dir=figures_dir,
            )

            for mo in emodels:
                plotting.scores(mo, figures_dir)
                plotting.traces(mo, mo["responses"], stimuli, figures_dir)

            return emodels

        logger.warning("In plot_models, no emodel for %s %s", self.emodel, self.species)
        return []
