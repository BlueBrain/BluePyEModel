"""Allows to execute the steps of the e-model building pipeline using python or CLI"""

import glob
import logging
import pathlib

from bluepyemodel.access_point import get_access_point
from bluepyemodel.efeatures_extraction.efeatures_extraction import extract_save_features_protocols
from bluepyemodel.emodel_pipeline import plotting
from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata
from bluepyemodel.model.model_configuration import configure_model
from bluepyemodel.optimisation import setup_and_run_optimisation
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.tools.multiprocessing import get_mapper
from bluepyemodel.tools.multiprocessing import ipyparallel_map_function
from bluepyemodel.tools.utils import get_checkpoint_path
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
        morph_class=None,
        synapse_class=None,
        layer=None,
        recipes_path=None,
        forge_path=None,
        nexus_organisation=None,
        nexus_project=None,
        nexus_endpoint="staging",
        use_ipyparallel=None,
        use_multiprocessing=None,
    ):
        """Initialize the emodel_pipeline.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel under which the
                configuration data are stored.
            data_access_point (str): name of the access_point used to access the data,
                can be "nexus" or "local".
                "local" expect the configuration to be  defined in a "config" directory
                containing recipes as in proj38. "nexus" expect the configuration to be defined
                on Nexus using NexusForge, see bluepyemodel/api/nexus.py.
            etype (str): name of the etype.
            ttype (str): name of the t-type. Required if using the gene expression or IC selector.
            mtype (str): name of the mtype.
            species (str): name of the species.
            brain_region (str): name of the brain region.
            iteration_tag (str): tag associated to the current run. If used with the local access
                point,the pipeline will work in the directory working_dir/run/iteration.
                If used with the Nexus access point, it will be used to tag the resources
                generated during the run.
            morph_class (str): name of the morphology class, has to be "PYR", "INT".
            synapse_class (str): name of the synapse class, has to be "EXC", "INH".
            layer (str): layer of the model.
            recipes_path (str): path of the recipes.json, only needed if the data access point is
                "local".
            forge_path (str): path to the .yml used to connect to Nexus Forge, only needed if
                db_api="nexus".
            nexus_organisation (str): name of the Nexus organisation in which the project is
                located.
            nexus_project (str): name of the Nexus project to which the forge will connect to
                retrieve the data
            nexus_endpoint (str): Nexus endpoint ("prod" or "staging")
            use_ipyparallel (bool): should the parallelization map be base on ipyparallel.
            use_multiprocessing (bool): should the parallelization map be based on multiprocessing.
        """

        # pylint: disable=too-many-arguments

        if use_ipyparallel and use_multiprocessing:
            raise ValueError(
                "use_ipyparallel and use_multiprocessing cannot be both True at the same time. "
                "Please choose one."
            )
        if use_ipyparallel:
            self.mapper = ipyparallel_map_function()
        elif use_multiprocessing:
            self.mapper = get_mapper(backend="multiprocessing")
        else:
            self.mapper = map

        endpoint = None
        if nexus_endpoint == "prod":
            endpoint = "https://bbp.epfl.ch/nexus/v1"
        elif nexus_endpoint == "staging":
            endpoint = "https://staging.nexus.ocp.bbp.epfl.ch/v1"

        self.access_point = get_access_point(
            emodel=emodel,
            etype=etype,
            ttype=ttype,
            mtype=mtype,
            species=species,
            brain_region=brain_region,
            iteration_tag=iteration_tag,
            morph_class=morph_class,
            synapse_class=synapse_class,
            layer=layer,
            access_point=data_access_point,
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

    def optimise(self, seed=1):
        """"""

        setup_and_run_optimisation(
            self.access_point,
            seed=seed,
            mapper=self.mapper,
            terminator=None,
        )

    def store_optimisation_results(self, seed=None):
        """"""

        checkpoint_path = get_checkpoint_path(self.access_point.emodel_metadata, seed=1)

        for chkp_path in glob.glob(checkpoint_path.replace("seed=1", "*")):

            if seed is not None and str(seed) not in chkp_path:
                continue

            tmp_seed = seed
            if tmp_seed is None:
                tmp_path = pathlib.Path(chkp_path).stem
                tmp_seed = next(
                    int(e.replace("seed=", "")) for e in tmp_path.split("__") if "seed=" in e
                )

            store_best_model(
                access_point=self.access_point,
                seed=tmp_seed,
                checkpoint_path=chkp_path,
            )

    def validation(self):
        """"""

        validate(
            access_point=self.access_point,
            mapper=self.mapper,
        )

    def plot(self, only_validated=False, load_from_local=False):
        for chkp_path in glob.glob("./checkpoints/*.pkl"):
            if self.access_point.emodel_metadata.emodel not in chkp_path:
                continue

            if (
                self.access_point.emodel_metadata.iteration
                and self.access_point.emodel_metadata.iteration not in chkp_path
            ):
                continue

            stem = str(pathlib.Path(chkp_path).stem)
            seed = int(stem.rsplit("seed=", maxsplit=1)[-1])

            plotting.optimisation(
                optimiser=self.access_point.pipeline_settings.optimiser,
                emodel=self.access_point.emodel_metadata.emodel,
                iteration=self.access_point.emodel_metadata.iteration,
                seed=seed,
                checkpoint_path=chkp_path,
                figures_dir=pathlib.Path("./figures")
                / self.access_point.emodel_metadata.emodel
                / "optimisation",
            )

        return plotting.plot_models(
            access_point=self.access_point,
            mapper=self.mapper,
            seeds=None,
            figures_dir=pathlib.Path("./figures") / self.access_point.emodel_metadata.emodel,
            plot_distributions=True,
            plot_scores=True,
            plot_traces=True,
            plot_currentscape=self.access_point.pipeline_settings.plot_currentscape,
            only_validated=only_validated,
            load_from_local=load_from_local,
        )

    def export_emodels(self, only_validated=False, seeds=None):
        export_emodels_sonata(
            self.access_point, only_validated, seeds=seeds, map_function=self.mapper
        )

    def summarize(self):
        print(self.access_point)


def sanitize_gitignore():
    """In order to avoid git issue when archiving the current working directory,
    adds the following lines to .gitignore: 'run/', 'checkpoints/', 'figures/',
    'logs/', '.ipython/', '.ipynb_checkpoints/'"""

    path_gitignore = pathlib.Path("./.gitignore")

    if not (path_gitignore.is_file()):
        raise FileNotFoundError("Could not update .gitignore as it does not exist.")

    with open(str(path_gitignore), "r") as f:
        lines = f.readlines()

    lines = " ".join(line for line in lines)

    to_add = ["run/", "checkpoints/", "figures/", "logs/", ".ipython/", ".ipynb_checkpoints/"]

    with open(str(path_gitignore), "a") as f:
        for a in to_add:
            if a not in lines:
                f.write(f"{a}\n")
