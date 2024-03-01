"""DataAccessPoint class."""

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
import pickle
from enum import Enum
from itertools import chain

import efel
import numpy
from bluepyopt.deapext.algorithms import _check_stopping_criteria
from bluepyopt.deapext.stoppingCriteria import MaxNGen

from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.tools.utils import get_checkpoint_path
from bluepyemodel.tools.utils import get_legacy_checkpoint_path
from bluepyemodel.tools.utils import read_checkpoint

# pylint: disable=no-member,unused-argument,assignment-from-no-return,no-value-for-parameter

logger = logging.getLogger(__name__)


class OptimisationState(Enum):
    """
    the possible states of an optimisation process.

    Attributes:
        COMPLETED (str): The optimisation process has completed successfully.
        IN_PROGRESS (str): The optimisation process is currently in progress.
        EMPTY (str): The optimisation process has not yet been started.
    """

    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    EMPTY = "empty"


class DataAccessPoint:
    """Abstract data access point class. This class is not meant to be used directly.
    Instead, it is used through the classes LocalAccessPoint and NexusAccessPoint.
    """

    def __init__(
        self,
        emodel=None,
        etype=None,
        ttype=None,
        mtype=None,
        species=None,
        brain_region=None,
        iteration_tag=None,
        synapse_class=None,
    ):
        """Init"""

        self.emodel_metadata = EModelMetadata(
            emodel,
            etype,
            ttype,
            mtype,
            species,
            brain_region,
            iteration_tag,
            synapse_class,
        )

    def set_emodel(self, emodel):
        """Setter for the name of the emodel."""
        self.emodel_metadata.emodel = emodel

    def get_pipeline_settings(self):
        """ """
        return EModelPipelineSettings()

    def store_efeatures(
        self,
        efeatures,
        current,
    ):
        """Save the efeatures and currents obtained from BluePyEfe"""

    def store_protocols(self, stimuli):
        """Save the protocols obtained from BluePyEfe"""

    def store_emodel(
        self,
        scores,
        params,
        optimiser_name,
        seed,
        validated=None,
        scores_validation=None,
    ):
        """Save a model obtained from BluePyOpt"""

    def get_emodel(self):
        """Get dict with parameter of single emodel (including seed if any)"""

    def get_emodels(self, emodels):
        """Get the list of emodels dictionaries."""

    def store_targets_configuration(self):
        """Store the configuration of the targets (targets and ephys files used)"""

    def get_targets_configuration(self):
        """Get the configuration of the targets (targets and ephys files used)"""

    def store_model_configuration(self):
        """Store the configuration of a model, including parameters, mechanisms and distributions"""

    def get_available_mechanisms(self):
        """Get all the available mechanisms"""

    def get_available_efeatures(self, cleaned=True):
        """Returns a curated list of available eFEL features"""

        efel_features = efel.getFeatureNames()

        if not cleaned:
            return efel_features

        features = []
        for f in efel_features:
            if f.endswith("indices"):
                continue
            if f in ["voltage", "time", "current"]:
                continue
            features.append(f)

        features.remove("AP_begin_time")
        features.remove("peak_time")

        return features

    def get_available_traces(self, filter_species=False, filter_brain_region=False):
        """Get the list of available Traces for the current species from Nexus"""
        return None

    def get_distributions(self):
        """Get the list of available distributions"""

    def store_distribution(self, distribution):
        """Store a channel distribution as a resource of type EModelChannelDistribution"""

    def get_model_configuration(self):
        """Get the configuration of the model, including parameters, mechanisms and distributions"""

    def store_fitness_calculator_configuration(self, configuration):
        """Store a fitness calculator configuration"""

    def get_fitness_calculator_configuration(self, record_ions_and_currents=False):
        """Get the configuration of the fitness calculator (efeatures and protocols)"""

    def get_mechanisms_directory(self):
        """Return the path to the directory containing the mechanisms for the current emodel"""

    def download_mechanisms(self):
        """Download the mod files when not already downloaded"""

    def get_emodel_names(self):
        """Get the list of all the names of emodels

        Returns:
            dict: keys are emodel names with seed, values are names without seed.
        """

    def store_hocs(
        self,
        only_validated=False,
        only_best=True,
        seeds=None,
        map_function=map,
        new_emodel_name=None,
        description=None,
        output_base_dir="export_emodels_hoc",
    ):
        """Store the hoc files"""

    def store_emodels_hoc(
        self,
        only_validated=False,
        only_best=True,
        seeds=None,
        map_function=map,
        new_emodel_name=None,
        description=None,
    ):
        """Store hoc file produced by export_hoc"""

    def store_emodels_sonata(
        self,
        only_validated=False,
        only_best=True,
        seeds=None,
        map_function=map,
        new_emodel_name=None,
        description=None,
    ):
        """Store hoc file produced by export_sonata"""

    def optimisation_state(self, seed=None, continue_opt=False):
        """Return the state of the optimisation.

        Args:
            seed (int): seed used in the optimisation.
            continue_opt (bool): whether to continue optimisation or not
                when the optimisation is not complete

        Raises:
            ValueError if optimiser in pipeline settings in neither
                "SO-CMA", "MO-CMA" or "IBEA"

        Returns:
            bool: True if completed, False if in progress or empty
        """

        checkpoint_path = get_checkpoint_path(self.emodel_metadata, seed=seed)

        # no file -> target not complete
        if not pathlib.Path(checkpoint_path).is_file():
            checkpoint_path = get_legacy_checkpoint_path(checkpoint_path)
            if not pathlib.Path(checkpoint_path).is_file():
                return OptimisationState.EMPTY

        # there is a file & continue opt is False -> target considered complete
        if not continue_opt:
            return OptimisationState.COMPLETED

        # there is a file & we want to continue optimisation -> check if optimisation if finished
        optimiser = self.pipeline_settings.optimiser
        ngen = self.pipeline_settings.max_ngen

        with open(str(checkpoint_path), "rb") as checkpoint_file:
            cp = pickle.load(checkpoint_file, encoding="latin1")

        # CMA
        if optimiser in ["SO-CMA", "MO-CMA"]:
            gen = cp["generation"]
            CMA_es = cp["CMA_es"]
            CMA_es.check_termination(gen)
            # no termination met -> still active -> target not complete
            if CMA_es.active:
                return OptimisationState.IN_PROGRESS
            return OptimisationState.COMPLETED

        # IBEA
        if optimiser == "IBEA":
            gen = cp["generation"]
            stopping_criteria = [MaxNGen(ngen)]
            stopping_params = {"gen": gen}
            run_complete = _check_stopping_criteria(stopping_criteria, stopping_params)
            if run_complete:
                return OptimisationState.COMPLETED
            return OptimisationState.IN_PROGRESS

        raise ValueError(f"Unknown optimiser: {optimiser}")

    def get_ion_currents_concentrations(self):
        """Get all ion currents and ion concentrations.

        Returns:
            tuple containing:

            (list of str): current (ion and non-specific) names for all available mechanisms
            (list of str): ionic concentration names for all available mechanisms
        """
        # pylint: disable=assignment-from-no-return
        mechs = self.get_available_mechanisms()
        if mechs is None:
            return None, None
        ion_currents = list(chain.from_iterable([mech.get_current() for mech in mechs]))
        ionic_concentrations = list(
            chain.from_iterable([mech.ionic_concentrations for mech in mechs])
        )
        # append i_pas which is present by default
        ion_currents.append("i_pas")
        return ion_currents, ionic_concentrations

    def __str__(self):
        str_ = "#############################################################\n"
        str_ += "################## SUMMARY: EMODEL CREATION #################\n"
        str_ += "#############################################################\n\n"

        str_ += "GENERAL INFORMATION\n"
        str_ += f"  Current working directory: {pathlib.Path.cwd()}\n"
        str_ += f"  Type of access point: {type(self).__name__}\n\n"

        str_ += "EMODEL METADATA\n"
        for k, v in vars(self.emodel_metadata).items():
            if v is not None:
                str_ += f"  {k}: {v}\n"
        str_ += "\n"

        str_ += "CONFIGURATION STATUS\n"
        str_ += f"  Has pipeline settings: {self.has_pipeline_settings()}\n"
        str_ += f"  Has targets configuration: {self.has_targets_configuration()}\n"
        str_ += (
            f"  Has a fitness calculator configuration: "
            f"{self.has_fitness_calculator_configuration()}\n"
        )
        str_ += f"  Has a model configuration: {self.has_model_configuration()}\n\n"

        if pathlib.Path("./checkpoints/").is_dir():
            checkpoints = glob.glob("./checkpoints/**/*.pkl", recursive=True)
            template_path = self.emodel_metadata.as_string()
            checkpoints = [c for c in checkpoints if template_path in c]
            str_ += "OPTIMISATION STATUS\n"
            str_ += f"  Number of checkpoints: {len(checkpoints)}\n"
            for c in checkpoints:
                run, run_metadata = read_checkpoint(c)
                str_ += f"    Seed {run_metadata['seed']};"
                str_ += f" Last generation: {run['logbook'].select('gen')[-1]};"
                str_ += f" Best fitness: {sum(run['halloffame'][0].fitness.values)}\n"
            str_ += "\n"

        str_ += "EMODELS BUILDING STATUS\n"
        emodels = self.get_emodels()
        if emodels:
            n_emodels = len(emodels)
            n_emodels_validated = len([e for e in emodels if e.passed_validation])
            idx_best = numpy.argmin([e.fitness for e in emodels])
            str_ += f"  Emodels stored: {n_emodels}\n"
            str_ += (
                f"  Number of validated emodel: {n_emodels_validated} "
                f"({n_emodels_validated / n_emodels} %)\n"
            )
            str_ += (
                f"  Best emodel: seed {emodels[idx_best].seed} with fitness"
                f" {emodels[idx_best].fitness}\n\n"
            )
        else:
            str_ += "  No emodels\n\n"

        return str_
