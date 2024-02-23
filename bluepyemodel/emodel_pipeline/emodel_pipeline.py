"""EModel_pipeline class."""

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

import glob
import logging
import pathlib

from bluepyemodel.access_point import get_access_point
from bluepyemodel.efeatures_extraction.efeatures_extraction import extract_save_features_protocols
from bluepyemodel.emodel_pipeline import plotting
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata
from bluepyemodel.model.model_configuration import configure_model
from bluepyemodel.optimisation import setup_and_run_optimisation
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.tools.multiprocessing import get_mapper
from bluepyemodel.tools.utils import get_checkpoint_path
from bluepyemodel.tools.utils import get_legacy_checkpoint_path
from bluepyemodel.validation.validation import validate

logger = logging.getLogger()


class EModel_pipeline:
    """The EModel_pipeline class is there to allow the execution of the steps
    of the e-model building pipeline using python (as opposed to the Luigi workflow).

    For an example of how to use to present class, see the example emodel_pipeline_local_python or
    the README.rst file."""

    def __init__(
        self,
        emodel,
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
        use_ipyparallel=None,
        use_multiprocessing=None,
        data_access_point=None,
    ):
        """Initializes the EModel_pipeline.

        Args:
            emodel (str): name of the emodel. Can be arbitrary but has to match the name of the
                emodel in the recipes.json configuration file.
            etype (str): name of the e-type of the e-model. Used as an identifier for the e-model.
            ttype (str): name of the t-type of the e-model. Used as an identifier for the e-model.
                This argument is required when using the gene expression or IC selector.
            mtype (str): name of the m-type of the e-model. Used as an identifier for the e-model.
            species (str): name of the species of the e-model. Used as an identifier for the
                e-model.
            brain_region (str): name of the brain region of the e-model. Used as an identifier for
                the e-model.
            iteration_tag (str): tag associated to the current run. Used as an identifier for the
                e-model.
                If used with an access point of type "local", the current pipeline will execute
                the model building steps in the subdirectory of ``./run/{iteration_tag}/`` expected
                to contain a copy of the configuration files, mechanisms, morphologies needed for
                model building. This subdirectory can be created, for example using the following
                shell script (see also the example emodel_pipeline_local_python):

                .. code-block:: shell

                    git add -A && git commit --allow-empty -a -m "Running optimization"
                    export iteration_tag=$(git rev-parse --short HEAD)
                    git archive --format=tar --prefix=${iteration_tag}/ HEAD
                    | (cd ./run/ && tar xf -)

                In this case, the current, the iteration_tag can then be passed during the
                instantiation of the EModel_pipeline.
            morph_class (str): name of the morphology class, has to be "PYR", "INT". To be
                depracted.
            synapse_class (str): name of the synapse class of the e-model, has to be "EXC", "INH".
                Not used at the moment.
            layer (str): layer of the e-model. To be depracted.
            recipes_path (str): path of the recipes.json configuration file.This configuration
                file is the main file required when using the access point of type "local". It
                is expected to be a json file containing a dictionary whose keys are the names
                of the e-models that will be built. The values associated to these keys are
                the recipes used to build these e-models. See the example recipes.json file in
                the example emodel_pipeline_local_python for more details.
            use_ipyparallel (bool): should the parallelization map used for the different steps of
                the e-model building pipeline be based on ipyparallel.
            use_multiprocessing (bool): should the parallelization map used for the different steps
                of the e-model building pipeline be based on multiprocessing.
            data_access_point (str): Used for legacy purposes only
        """

        # pylint: disable=too-many-arguments

        if use_ipyparallel and use_multiprocessing:
            raise ValueError(
                "use_ipyparallel and use_multiprocessing cannot be both True at the same time. "
                "Please choose one."
            )
        if use_ipyparallel:
            self.mapper = get_mapper(backend="ipyparallel")
        elif use_multiprocessing:
            self.mapper = get_mapper(backend="multiprocessing")
        else:
            self.mapper = map

        endpoint = None

        if data_access_point is not None and data_access_point != "local":
            raise ValueError(
                "Attempted to set a legacy variable. "
                "This variable should not be modified in new code."
            )

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
            access_point="local",
            recipes_path=recipes_path,
            final_path="final.json",
            endpoint=endpoint,
        )

    def configure_model(
        self, morphology_name, morphology_path=None, morphology_format=None, use_gene_data=False
    ):
        """To be deprecated"""

        return configure_model(
            self.access_point,
            morphology_name=morphology_name,
            morphology_path=morphology_path,
            morphology_format=morphology_format,
            use_gene_data=use_gene_data,
        )

    def extract_efeatures(self):
        """Extract the e-features related to the current e-model."""

        return extract_save_features_protocols(access_point=self.access_point, mapper=self.mapper)

    def optimise(self, seed=1):
        """Optimise the e-model.

        Args:
            seed (int): seed used for the random number generator used in the optimisation.
        """

        setup_and_run_optimisation(
            self.access_point,
            seed=seed,
            mapper=self.mapper,
            terminator=None,
        )

    def store_optimisation_results(self, seed=None):
        """Store the results of the optimisation. That is, reads the pickles file containing
        checkpoint of the optimisations and store the best model for each seed. When using the
        "local" access point, the best models are stored in a local json file called "final.json"
        When using the "nexus" access point, the best models are stored in Nexus resources of type
        EModel.

        Args:
            seed (int): specifies which seed to store. If None, the best models for all seeds
                are stored.
        """

        if seed is not None:
            store_best_model(access_point=self.access_point, seed=seed)

        else:
            checkpoint_path = get_checkpoint_path(self.access_point.emodel_metadata, seed=1)
            checkpoint_list = glob.glob(checkpoint_path.replace("seed=1", "*"))
            if not checkpoint_list:
                checkpoint_list = glob.glob(
                    get_legacy_checkpoint_path(checkpoint_path).replace("seed=1", "*")
                )

            for chkp_path in checkpoint_list:
                file_name = pathlib.Path(chkp_path).stem
                tmp_seed = next(
                    int(e.replace("seed=", "")) for e in file_name.split("__") if "seed=" in e
                )

                store_best_model(
                    access_point=self.access_point,
                    seed=tmp_seed,
                    checkpoint_path=chkp_path,
                )

    def validation(self):
        """Run a validation on the stored e-models. To work, some protocols have to be
        marked as for validation only. If no protocol is marked as such, the validation will
        simply check if the scores are all below a given threshold."""

        validate(
            access_point=self.access_point,
            mapper=self.mapper,
        )

    def plot(self, only_validated=False, load_from_local=False):
        """Plot the results of the optimisation in a subfolder called "figures".

        Args:
            only_validated (bool): if True, only the e-models that have successfully passed
                validation will be plotted.
            load_from_local (bool): if True, loads responses of the e-models from local files
                instead of recomputing them. Responses are automatically saved locally when
                plotting currentscapes.
        """

        cell_evaluator = get_evaluator_from_access_point(
            self.access_point,
            include_validation_protocols=True,
            record_ions_and_currents=self.access_point.pipeline_settings.plot_currentscape,
        )

        chkp_paths = glob.glob("./checkpoints/**/*.pkl", recursive=True)
        if not chkp_paths:
            raise ValueError("The checkpoints directory is empty, or there are no .pkl files.")

        # Filter the checkpoints to plot
        checkpoint_paths = []
        for chkp_path in chkp_paths:
            if self.access_point.emodel_metadata.emodel not in chkp_path:
                continue
            if (
                self.access_point.emodel_metadata.iteration
                and self.access_point.emodel_metadata.iteration not in chkp_path
            ):
                continue
            checkpoint_paths.append(chkp_path)

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

        if self.access_point.pipeline_settings.plot_parameter_evolution:
            plotting.evolution_parameters_density(
                evaluator=cell_evaluator,
                checkpoint_paths=checkpoint_paths,
                figures_dir=pathlib.Path("./figures")
                / self.access_point.emodel_metadata.emodel
                / "parameter_evolution",
            )

        return plotting.plot_models(
            access_point=self.access_point,
            mapper=self.mapper,
            seeds=None,
            figures_dir=pathlib.Path("./figures") / self.access_point.emodel_metadata.emodel,
            plot_distributions=True,
            plot_scores=True,
            plot_traces=True,
            plot_thumbnail=True,
            plot_currentscape=self.access_point.pipeline_settings.plot_currentscape,
            plot_bAP_EPSP=self.access_point.pipeline_settings.plot_bAP_EPSP,
            plot_dendritic_ISI_CV=True,
            plot_dendritic_rheobase=True,
            only_validated=only_validated,
            save_recordings=self.access_point.pipeline_settings.save_recordings,
            load_from_local=load_from_local,
            cell_evaluator=cell_evaluator,
        )

    def export_emodels(self, only_validated=False, seeds=None):
        """Export the e-models in the SONATA format. The results of the export are stored
        in a subfolder called export_emodels_sonata.

        Args:
            only_validated (bool): if True, only the e-models that have successfully passed
                validation will be exported.
            seeds (list): list of the seeds of the e-models to export. If None, all the e-models
                are exported.
        """

        export_emodels_sonata(
            self.access_point, only_validated, seeds=seeds, map_function=self.mapper
        )

    def summarize(self):
        """Prints a summary of the state of the current e-model building procedure"""

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
