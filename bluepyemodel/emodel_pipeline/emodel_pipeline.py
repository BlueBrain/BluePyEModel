"""Allows to execute the steps of the e-model building pipeline using python or CLI"""

import glob
import logging
import pathlib

from bluepyemodel.access_point import get_access_point
from bluepyemodel.efeatures_extraction.efeatures_extraction import extract_save_features_protocols
from bluepyemodel.emodel_pipeline import plotting
from bluepyemodel.model.model_configuration import configure_model
from bluepyemodel.optimisation import setup_and_run_optimisation
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.tools.multiprocessing import ipyparallel_map_function
from bluepyemodel.validation.validation import validate

logger = logging.getLogger()


class EModel_pipeline:

    """EModel pipeline"""

    def __init__(
        self,
        emodel,
        data_access_point,
        etype=None,
        ttype=None,
        mtype=None,
        species=None,
        brain_region=None,
        iteration_tag=None,
        recipes_path=None,
        forge_path=None,
        nexus_organisation=None,
        nexus_project=None,
        nexus_endpoint="staging",
        use_ipyparallel=None,
    ):
        """Initialize the emodel_pipeline.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel under which the
                configuration data are stored.
            species (str): name of the species.
            brain_region (str): name of the brain region.
            data_access_point (str): name of the access_point used to access the data,
                can be "nexus" or "local".
                "local" expect the configuration to be  defined in a "config" directory
                containing recipes as in proj38. "nexus" expect the configuration to be defined
                on Nexus using NexusForge, see bluepyemodel/api/nexus.py.
            recipes_path (str): path of the recipes.json, only needed if the data access point is
                "local".
            forge_path (str): path to the .yml used to connect to Nexus Forge, only needed if
                db_api="nexus".
            nexus_organisation (str): name of the Nexus organisation in which the project is
                located.
            nexus_project (str): name of the Nexus project to which the forge will connect to
                retrieve the data
            nexus_endpoint (str): Nexus endpoint ("prod" or "staging")
            ttype (str): name of the t-type. Required if using the gene expression or IC selector.
            iteration_tag (str): tag associated to the current run. If used with the local access
                point,the pipeline will work in the directory working_dir/run/iteration.
                If used with the Nexus access point, it will be used to tag the resources
                generated during the run.
            use_ipyparallel (bool): should the parallelization map be base on ipyparallel.
        """

        self.emodel = emodel
        self.etype = etype
        self.ttype = ttype
        self.mtype = mtype
        self.species = species
        self.brain_region = brain_region
        self.iteration_tag = iteration_tag

        if use_ipyparallel:
            self.mapper = ipyparallel_map_function()
        else:
            self.mapper = map

        self.access_point = self.init_access_point(
            data_access_point,
            recipes_path,
            forge_path,
            nexus_organisation,
            nexus_endpoint,
            nexus_project,
        )

    def init_access_point(
        self,
        data_access_point,
        recipes_path,
        forge_path,
        nexus_organisation,
        nexus_endpoint,
        nexus_project,
    ):
        """Instantiate a data access point, either to Nexus or GPFS"""

        endpoint = None
        if nexus_endpoint == "prod":
            endpoint = "https://bbp.epfl.ch/nexus/v1"
        elif nexus_endpoint == "staging":
            endpoint = "https://staging.nexus.ocp.bbp.epfl.ch/v1"

        emodel_dir = "./"
        if self.iteration_tag and data_access_point == "local":
            emodel_dir = str(pathlib.Path("./") / "run" / self.iteration_tag)

        return get_access_point(
            emodel=self.emodel,
            etype=self.etype,
            ttype=self.ttype,
            mtype=self.mtype,
            species=self.species,
            brain_region=self.brain_region,
            iteration_tag=self.iteration_tag,
            access_point=data_access_point,
            emodel_dir=emodel_dir,
            recipes_path=recipes_path,
            final_path="final.json",
            organisation=nexus_organisation,
            project=nexus_project,
            endpoint=endpoint,
            forge_path=forge_path,
        )

    def configure_model(
        self, morphology_name, morphology_path=None, morphology_format=None, use_gene_data=False
    ):
        """"""

        return configure_model(
            self.access_point,
            morphology_name=morphology_name,
            morphology_path=morphology_path,
            morphology_format=morphology_format,
            use_gene_data=use_gene_data,
        )

    def extract_efeatures(self):
        """"""

        return extract_save_features_protocols(access_point=self.access_point, mapper=self.mapper)

    def optimize(self, seed=1):
        """"""

        setup_and_run_optimisation(
            self.access_point,
            seed=seed,
            mapper=self.mapper,
            terminator=None,
        )

    def store_optimisation_results(self, seed=None):
        """"""

        for chkp_path in glob.glob("./checkpoints/*.pkl"):

            if self.emodel not in chkp_path:
                continue

            if (
                self.access_point.emodel_metadata.iteration
                and self.access_point.emodel_metadata.iteration not in chkp_path
            ):
                continue

            if seed and str(seed) not in chkp_path:
                continue

            store_best_model(
                access_point=self.access_point,
                seed=seed,
                checkpoint_path=chkp_path,
            )

    def validation(self):
        """"""

        validate(
            access_point=self.access_point,
            mapper=self.mapper,
        )

    def plot(self, only_validated=False):

        for chkp_path in glob.glob("./checkpoints/*.pkl"):

            if self.emodel not in chkp_path:
                continue

            if (
                self.access_point.emodel_metadata.iteration
                and self.access_point.emodel_metadata.iteration not in chkp_path
            ):
                continue

            plotting.optimization(
                checkpoint_path=chkp_path,
                figures_dir=pathlib.Path("./figures") / self.emodel / "optimisation",
            )

        return plotting.plot_models(
            access_point=self.access_point,
            mapper=self.mapper,
            seeds=None,
            figures_dir=pathlib.Path("./figures") / self.emodel,
            plot_distributions=True,
            plot_scores=True,
            plot_traces=True,
            only_validated=only_validated,
        )


def sanitize_gitignore():
    """In order to avoid git issue when archiving the current working directory,
    adds the following lines to .gitignore: 'run/', 'checkpoints/', 'figures/',
    'logs/', '.ipython/', '.ipynb_checkpoints/'"""

    path_gitignore = pathlib.Path("./.gitignore")

    if not (path_gitignore.is_file()):
        raise Exception("Could not update .gitignore as it does not exist.")

    with open(str(path_gitignore), "r") as f:
        lines = f.readlines()

    lines = " ".join(line for line in lines)

    to_add = ["run/", "checkpoints/", "figures/", "logs/", ".ipython/", ".ipynb_checkpoints/"]

    with open(str(path_gitignore), "a") as f:
        for a in to_add:
            if a not in lines:
                f.write(f"{a}\n")
