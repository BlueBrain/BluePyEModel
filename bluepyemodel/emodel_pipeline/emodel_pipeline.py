"""Define a class for a pipeline allowing the creation of emodels."""
import datetime
import logging
import os
import tarfile
import time
from pathlib import Path

from git import Repo

from bluepyemodel.api import get_db
from bluepyemodel.efeatures_extraction import extract_save_features_protocols
from bluepyemodel.emodel_pipeline import plotting
from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_db
from bluepyemodel.optimisation import optimisation
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.validation.validation import validate

logger = logging.getLogger(__name__)

mechanisms_dir = "mechanisms"
run_dir = "run"
final_path = "final.json"


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
        brain_region,
        db_api,
        recipes_path=None,
        forge_path=None,
        use_git=False,
        githash=None,
        nexus_organisation="demo",
        nexus_projet="emodel_pipeline",
        nexus_enpoint="https://bbp.epfl.ch/nexus/v1",
    ):
        """Initialize the emodel_pipeline.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel under which the
                configuration data are stored.
            species (str): name of the species.
            brain_region (str): name of the brain region.
            db_api (str): name of the API used to access the data, can be "nexus" or "singlecell".
                "singlecell" expect the configuration to be  defined in a "config" directory
                containing recipes as in proj38. "nexus" expect the configuration to be defined
                on Nexus using NexusForge, see bluepyemodel/api/nexus.py.
            recipes_path (str): path of the recipes.json, only needed if db_api="singlecell".
            forge_path (str): path to the .yml used to connect to Nexus Forge, only needed if
                db_api="nexus".
            use_git (bool): if True, work will be perfomed in a sub-directory:
                working_dir/run/githash. If use_git is True and githash is None, a new githash will
                be generated. The pipeline will expect a git repository to exist in working_dir.
            githash (str): if provided, the pipeline will work in the directory
                working_dir/run/githash. Needed when continuing work or resuming optimisations.
        """

        self.emodel = emodel
        self.species = species
        self.brain_region = brain_region

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

        self.db = self.connect_db(
            db_api, recipes_path, nexus_organisation, nexus_projet, nexus_enpoint, forge_path
        )

    @property
    def working_dir(self):
        if self.githash:
            return str(Path("./") / run_dir / self.githash)
        return "./"

    def connect_db(
        self, db_api, recipes_path, nexus_organisation, nexus_projet, nexus_enpoint, forge_path
    ):
        """
        Instantiate the api from which the pipeline will get and store the data.
        """
        return get_db(
            db_api,
            emodel=self.emodel,
            emodel_dir=self.working_dir,
            recipes_path=recipes_path,
            final_path=final_path,
            species=self.species,
            brain_region=self.brain_region,
            organisation=nexus_organisation,
            project=nexus_projet,
            endpoint=nexus_enpoint,
            forge_path=forge_path,
        )

    def get_evaluator(
        self,
        stochasticity=False,
        include_validation_protocols=False,
        additional_protocols=None,
        timeout=600,
    ):
        """Create an evaluator for the emodel.

        Args:
            stochasticity (bool): should channels behave stochastically if they can.
            include_validation_protocols (bool): should the validation protocols
                and validation efeatures be added to the evaluator.
            additional_protocols (dict): definition of supplementary protocols. See
                examples/optimisation for usage.
            timeout (float): duration (in second) after which the evaluation of a protocol will
                be interrupted.

        Returns:
            Evaluator
        """

        return get_evaluator_from_db(
            emodel=self.emodel,
            db=self.db,
            stochasticity=stochasticity,
            include_validation_protocols=include_validation_protocols,
            additional_protocols=additional_protocols,
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
        plot=False,
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

        map_function = ipyparallel_map_function("USEIPYP")

        efeatures, stimuli, current = extract_save_features_protocols(
            emodel=self.emodel,
            emodel_db=self.db,
            files_metadata=files_metadata,
            targets=targets,
            protocols_threshold=protocols_threshold,
            threshold_nvalue_save=threshold_nvalue_save,
            mapper=map_function,
            name_Rin_protocol=name_Rin_protocol,
            name_rmp_protocol=name_rmp_protocol,
            validation_protocols=validation_protocols,
            plot=plot,
        )

        return efeatures, stimuli, current

    def optimize(
        self,
        max_ngen=1000,
        stochasticity=False,
        opt_params=None,
        optimizer="MO-CMA",
        checkpoint_path=None,
        continue_opt=False,
        timeout=600,
    ):
        """Run optimisation.

        Args:
            max_ngen (int): maximum number of generations of the evolutionary process.
            stochasticity (bool): should channels behave stochastically if they can.
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
            include_validation_protocols=False,
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
        checkpoint_path,
        stochasticity=False,
        opt_params=None,
        optimizer="MO-CMA",
    ):
        """Store the best model from an optimization. Reads a checkpoint file generated
            by BluePyOpt and store the best individual of the hall of fame.

        Args:
            stochasticity (bool): should channels behave stochastically if they can.
            opt_params (dict): optimisation parameters. Keys have to match the optimizer's call.
            optimizer (str): algorithm used for optimization, can be "IBEA", "SO-CMA", "MO-CMA".
        """

        if opt_params is None:
            opt_params = {}

        store_best_model(
            emodel_db=self.db,
            emodel=self.emodel,
            seed=opt_params.get("seed", ""),
            stochasticity=stochasticity,
            include_validation_protocols=False,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            githash=self.githash,
        )

    def compute_responses(
        self,
        cell_evaluator=None,
        map_function=None,
        stochasticity=False,
        additional_protocols=None,
    ):
        """Wrapper around compute_responses.

        Compute the responses of the emodel to the optimisation and validation protocols.

        Args:
            cell_evaluator (CellEvaluator): evaluator for the cell model/protocols/e-feature set.
            map_function (map): used to parallelize the evaluation of the
                individual in the population.
            stochasticity (bool): should channels behave stochastically if they can.
            additional_protocols (dict): definition of supplementary protocols. See
                examples/optimisation for usage.
        Returns:
            emodels (list): list of emodels.
        """

        if cell_evaluator is None:

            if additional_protocols is None:
                additional_protocols = {}

            cell_evaluator = self.get_evaluator(
                stochasticity=stochasticity,
                include_validation_protocols=True,
                additional_protocols=additional_protocols,
            )

        if map_function is None:
            map_function = ipyparallel_map_function("USEIPYP")

        return compute_responses(
            self.db,
            self.emodel,
            cell_evaluator,
            map_function,
        )

    def validate(
        self,
        validation_function=None,
        stochasticity=False,
    ):
        """Compute the scores and traces for the optimisation and validation
        protocols and perform validation.

        Args:
            validation_function (python function): function used to decide if a model
                passes validation or not. Should rely on emodel['scores'] and
                emodel['scores_validation']. See bluepyemodel/validation for examples.
            stochasticity (bool): should channels behave stochastically if they can.

        Returns:
            emodels (list): list of emodels.
        """

        mapper = ipyparallel_map_function("USEIPYP")

        return validate(
            emodel_db=self.db,
            emodel=self.emodel,
            mapper=mapper,
            validation_function=validation_function,
            stochasticity=stochasticity,
            additional_protocols=None,
            threshold=5.0,
            validation_protocols_only=False,
        )

    def plot_models(
        self,
        figures_dir="./figures",
        stochasticity=False,
        additional_protocols=None,
        map_function=None,
        seeds=None,
        plot_distributions=True,
        plot_scores=True,
        plot_traces=True,
        only_validated=False,
    ):
        """Plot the traces, scores and parameter distributions for all the models
            matching the emodels name.

        Args:
            figures_dir (str): path of the directory in which the figures should be saved.
            stochasticity (bool): should channels behave stochastically if they can.
            additional_protocols (dict): definition of supplementary protocols. See
                examples/optimisation for usage.
            map_function (map): used to parallelize the evaluation of the
                individual in the population.
            seeds (list): if not None, filter emodels to keep only the ones with these seeds.
            plot_distributions (bool): True to plot the parameters distributions
            plot_scores (bool): True to plot the scores
            plot_traces (bool): True to plot the traces
            only_validated (bool): True to only plot validated models

        Returns:
            emodels (list): list of emodels.
        """

        if map_function is None:
            map_function = ipyparallel_map_function("USEIPYP")

        return plotting.plot_models(
            self.db,
            self.emodel,
            mapper=map_function,
            seeds=seeds,
            figures_dir=figures_dir,
            stochasticity=stochasticity,
            additional_protocols=additional_protocols,
            plot_distributions=plot_distributions,
            plot_scores=plot_scores,
            plot_traces=plot_traces,
            only_validated=only_validated,
        )
