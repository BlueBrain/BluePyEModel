"""Access point using Nexus Forge"""

"""
Copyright 2024 Blue Brain Project / EPFL

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

import copy
import logging
import os
import pathlib
import subprocess
import time
from itertools import chain

import pandas
from kgforge.core import Resource

from bluepyemodel.access_point.access_point import DataAccessPoint
from bluepyemodel.access_point.forge_access_point import NEXUS_PROJECTS_TRACES
from bluepyemodel.access_point.forge_access_point import AccessPointException
from bluepyemodel.access_point.forge_access_point import NexusForgeAccessPoint
from bluepyemodel.access_point.forge_access_point import check_resource
from bluepyemodel.access_point.forge_access_point import filter_mechanisms_with_brain_region
from bluepyemodel.access_point.forge_access_point import get_available_traces
from bluepyemodel.access_point.forge_access_point import get_brain_region_notation
from bluepyemodel.access_point.forge_access_point import get_curated_morphology
from bluepyemodel.access_point.forge_access_point import get_nexus_brain_region
from bluepyemodel.access_point.forge_access_point import ontology_forge_access_point
from bluepyemodel.efeatures_extraction.trace_file import TraceFile
from bluepyemodel.emodel_pipeline.emodel_script import EModelScript
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.emodel_pipeline.emodel_workflow import EModelWorkflow
from bluepyemodel.export_emodel.utils import copy_hocs_to_new_output_path
from bluepyemodel.export_emodel.utils import get_hoc_file_path
from bluepyemodel.export_emodel.utils import get_output_path
from bluepyemodel.export_emodel.utils import select_emodels
from bluepyemodel.model.mechanism_configuration import MechanismConfiguration
from bluepyemodel.tools.mechanisms import NEURON_BUILTIN_MECHANISMS
from bluepyemodel.tools.mechanisms import discriminate_by_temp

# pylint: disable=too-many-arguments,unused-argument

logger = logging.getLogger("__main__")


class NexusAccessPoint(DataAccessPoint):
    """API to retrieve, push and format data from and to the Knowledge Graph"""

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
        project="emodel_pipeline",
        organisation="demo",
        endpoint="https://bbp.epfl.ch/nexus/v1",
        forge_path=None,
        forge_ontology_path=None,
        access_token=None,
        sleep_time=10,
    ):
        """Init

        Args:
            emodel (str): name of the emodel
            etype (str): name of the electric type.
            ttype (str): name of the transcriptomic type.
                Required if using the gene expression or IC selector.
            mtype (str): name of the morphology type.
            species (str): name of the species.
            brain_region (str): name of the brain location.
            iteration_tag (str): tag associated to the current run.
            synapse_class (str): synapse class (neurotransmitter).
            project (str): name of the Nexus project.
            organisation (str): name of the Nexus organization to which the project belong.
            endpoint (str): Nexus endpoint.
            forge_path (str): path to a .yml used as configuration by nexus-forge.
            forge_ontology_path (str): path to a .yml used for the ontology in nexus-forge.
            access_token (str): Nexus connection token.
            sleep_time (int): time to wait between two Nexus requests (in case of slow indexing).
        """

        super().__init__(
            emodel,
            etype,
            ttype,
            mtype,
            species,
            brain_region,
            iteration_tag,
            synapse_class,
        )

        self.access_point = NexusForgeAccessPoint(
            project=project,
            organisation=organisation,
            endpoint=endpoint,
            forge_path=forge_path,
            access_token=access_token,
        )

        if forge_ontology_path is None:
            self.forge_ontology_path = forge_path
        else:
            self.forge_ontology_path = forge_ontology_path

        # This trick is used to have nexus type descriptions on one side and basic
        # strings on the other
        self.emodel_metadata_ontology = copy.deepcopy(self.emodel_metadata)
        self.build_ontology_based_metadata()
        self.emodel_metadata.allen_notation = get_brain_region_notation(
            self.emodel_metadata.brain_region,
            self.access_point.access_token,
            self.forge_ontology_path,
            self.access_point.endpoint,
        )

        self.pipeline_settings = self.get_pipeline_settings(strict=False)

        directory_name = self.emodel_metadata.as_string()
        (pathlib.Path("./nexus_temp/") / directory_name).mkdir(parents=True, exist_ok=True)

        self.sleep_time = sleep_time

    @property
    def download_directory(self):
        return pathlib.Path("./nexus_temp") / str(self.emodel_metadata.iteration)

    def check_mettypes(self):
        """Check that etype, mtype and ttype are present on nexus"""
        ontology_access_point = ontology_forge_access_point(
            self.access_point.access_token, self.forge_ontology_path, self.access_point.endpoint
        )

        logger.info("Checking if etype %s is present on nexus...", self.emodel_metadata.etype)
        check_resource(
            self.emodel_metadata.etype,
            "etype",
            access_point=ontology_access_point,
            access_token=self.access_point.access_token,
            forge_path=self.forge_ontology_path,
            endpoint=self.access_point.endpoint,
        )
        logger.info("Etype checked")

        if self.emodel_metadata.mtype is not None:
            logger.info(
                "Checking if mtype %s is present on nexus...",
                self.emodel_metadata.mtype,
            )
            check_resource(
                self.emodel_metadata.mtype,
                "mtype",
                access_point=ontology_access_point,
                access_token=self.access_point.access_token,
                forge_path=self.forge_ontology_path,
                endpoint=self.access_point.endpoint,
            )
            logger.info("Mtype checked")
        else:
            logger.info("Mtype is None, its presence on Nexus is not being checked.")

        if self.emodel_metadata.ttype is not None:
            logger.info(
                "Checking if ttype %s is present on nexus...",
                self.emodel_metadata.ttype,
            )
            check_resource(
                self.emodel_metadata.ttype,
                "ttype",
                access_point=ontology_access_point,
                access_token=self.access_point.access_token,
                forge_path=self.forge_ontology_path,
                endpoint=self.access_point.endpoint,
            )
            logger.info("Ttype checked")
        else:
            logger.info("Ttype is None, its presence on Nexus is not being checked.")

    def get_pipeline_settings(self, strict=True):
        if strict:
            return self.access_point.nexus_to_object(
                type_="EModelPipelineSettings",
                metadata=self.emodel_metadata_ontology.filters_for_resource(),
                download_directory=self.download_directory,
                legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
            )

        try:
            return self.access_point.nexus_to_object(
                type_="EModelPipelineSettings",
                metadata=self.emodel_metadata_ontology.filters_for_resource(),
                download_directory=self.download_directory,
                legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
            )
        except AccessPointException:
            return EModelPipelineSettings()

    def store_pipeline_settings(self, pipeline_settings):
        """Save an EModelPipelineSettings on Nexus"""

        self.access_point.object_to_nexus(
            pipeline_settings,
            self.emodel_metadata_ontology.for_resource(),
            self.emodel_metadata.as_string(),
            self.emodel_metadata_ontology.filters_for_resource_legacy(),
            replace=True,
        )

    def build_ontology_based_metadata(self):
        """Get the ontology related to the metadata"""

        self.emodel_metadata_ontology.species = self.get_nexus_subject(self.emodel_metadata.species)
        self.emodel_metadata_ontology.brain_region = get_nexus_brain_region(
            self.emodel_metadata.brain_region,
            self.access_point.access_token,
            self.forge_ontology_path,
            self.access_point.endpoint,
        )

    def get_nexus_subject(self, species):
        """
        Get the ontology of a species based on the species name.

        Args:
            species (str): The common name or scientific name of the species.
            Can be None, in which case the function will return None.

        Returns:
            dict: The ontology data for the specified species.

        Raises:
            ValueError: If the species is not recognized.
        """

        if species is None:
            return None

        species = species.lower()
        if species in ("human", "homo sapiens"):
            subject = {
                "type": "Subject",
                "species": {
                    "id": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
                    "label": "Homo sapiens",
                },
            }

        elif species in ("rat", "rattus norvegicus"):
            subject = {
                "type": "Subject",
                "species": {
                    "id": "http://purl.obolibrary.org/obo/NCBITaxon_10116",
                    "label": "Rattus norvegicus",
                },
            }

        elif species in ("mouse", "mus musculus"):
            subject = {
                "type": "Subject",
                "species": {
                    "id": "http://purl.obolibrary.org/obo/NCBITaxon_10090",
                    "label": "Mus musculus",
                },
            }
        else:
            raise ValueError(f"Unknown species {species}.")

        return subject

    def store_object(self, object_, seed=None, description=None, currents=None):
        """Store a BPEM object on Nexus"""

        metadata_dict = self.emodel_metadata_ontology.for_resource()
        if seed is not None:
            metadata_dict["seed"] = seed
        if description is not None:
            metadata_dict["description"] = description

        self.access_point.object_to_nexus(
            object_,
            metadata_dict,
            self.emodel_metadata.as_string(),
            self.emodel_metadata_ontology.filters_for_resource_legacy(),
            replace=True,
            currents=currents,
        )

    def get_targets_configuration(self):
        """Get the configuration of the targets (targets and ephys files used)"""

        configuration = self.access_point.nexus_to_object(
            type_="ExtractionTargetsConfiguration",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            download_directory=self.download_directory,
            legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
        )

        configuration.available_traces = self.get_available_traces()
        configuration.available_efeatures = self.get_available_efeatures()

        if not configuration.files:
            logger.debug(
                "Empty list of files in the TargetsConfiguration, filling "
                "it using what is available on Nexus for the present etype."
            )
            filtered_traces = [
                trace
                for trace in configuration.available_traces
                if trace.etype == self.emodel_metadata.etype
            ]
            if not filtered_traces:
                raise AccessPointException(
                    "Could not find any trace with etype {self.emodel_metadata.etype}. "
                    "Please specify files in your ExtractionTargetsConfiguration."
                )
            configuration.files = filtered_traces

        for file in configuration.files:
            file.filepath = self.download_trace(
                id_=file.id, id_legacy=file.resource_id, name=file.filename
            )

        return configuration

    def store_targets_configuration(self, configuration):
        """Store the configuration of the targets (targets and ephys files used)"""

        # Search for all Traces on Nexus and add their Nexus ids to the configuration
        traces = get_available_traces(
            access_token=self.access_point.access_token,
            forge_path=self.access_point.forge_path,
        )

        available_traces_names = [trace.name for trace in traces]

        for file in configuration.files:
            if file.cell_name in available_traces_names:
                file.id = traces[available_traces_names.index(file.cell_name)].id
            else:
                logger.warning("Trace %s not found.", file.cell_name)

        self.store_object(configuration)

    def get_fitness_calculator_configuration(self, record_ions_and_currents=False):
        """Get the configuration of the fitness calculator (efeatures and protocols)"""

        configuration = self.access_point.nexus_to_object(
            type_="FitnessCalculatorConfiguration",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            download_directory=self.download_directory,
            legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
        )

        # contains ion currents and ionic concentrations to be recorded
        ion_variables = None
        if record_ions_and_currents:
            ion_currents, ionic_concentrations = self.get_ion_currents_concentrations()
            if ion_currents is not None and ionic_concentrations is not None:
                ion_variables = list(chain.from_iterable((ion_currents, ionic_concentrations)))

        for prot in configuration.protocols:
            prot.recordings, prot.recordings_from_config = prot.init_recordings(
                prot.recordings_from_config, ion_variables
            )

        if configuration.name_rmp_protocol is None:
            configuration.name_rmp_protocol = self.pipeline_settings.name_rmp_protocol
        if configuration.name_rin_protocol is None:
            configuration.name_rin_protocol = self.pipeline_settings.name_Rin_protocol
        if configuration.validation_protocols is None or configuration.validation_protocols == []:
            configuration.validation_protocols = self.pipeline_settings.validation_protocols
        if configuration.stochasticity is None:
            configuration.stochasticity = self.pipeline_settings.stochasticity

        return configuration

    def store_fitness_calculator_configuration(self, configuration):
        """Store a fitness calculator configuration as a resource of type
        FitnessCalculatorConfiguration"""
        workflow, nexus_id = self.get_emodel_workflow()

        if workflow is None:
            raise AccessPointException(
                "No EModelWorkflow available to which the EModels can be linked"
            )

        configuration.workflow_id = nexus_id
        self.store_object(configuration)
        # wait for the object to be uploaded and fetchable
        time.sleep(self.sleep_time)

        # fetch just uploaded FCC resource to get its id and give it to emodel workflow
        type_ = "FitnessCalculatorConfiguration"
        filters = {"type": type_}
        filters.update(self.emodel_metadata_ontology.filters_for_resource())
        filters_legacy = {"type": type_}
        filters_legacy.update(self.emodel_metadata_ontology.filters_for_resource_legacy())
        resource = self.access_point.fetch_one(filters, filters_legacy)
        fitness_id = resource.id

        workflow.fitness_configuration_id = fitness_id
        self.store_or_update_emodel_workflow(workflow)

    def get_model_configuration(self, skip_get_available_morph=True):
        """Get the configuration of the model, including parameters, mechanisms and distributions

        Args:
            skip_get_available_morphs (bool): set to True to skip getting the available
                morphologies and setting them to configuration.
                available_morphologies are only used in
                bluepyemodel.model.model_configuration.configure_model, so we assume
                they have already been checked for configuration present on nexus.
        """

        configuration = self.access_point.nexus_to_object(
            type_="EModelConfiguration",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            download_directory=self.download_directory,
            legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
        )

        morph_path = self.download_morphology(
            configuration.morphology.name,
            configuration.morphology.format,
            configuration.morphology.id,
        )
        any_downloaded = False
        if self.pipeline_settings.use_ProbAMPANMDA_EMS:
            any_downloaded = self.download_ProbAMPANMDA_EMS()
        self.download_mechanisms(configuration.mechanisms, any_downloaded)

        configuration.morphology.path = morph_path
        logger.debug("Using morphology: %s", configuration.morphology.path)
        configuration.available_mechanisms = self.get_available_mechanisms()
        if not skip_get_available_morph:
            configuration.available_morphologies = self.get_available_morphologies()

        return configuration

    def fetch_and_filter_mechanism(self, mechanism, ontology_access_point):
        """Find a mech resource based on its brain region, temperature, species and ljp correction

        Args:
            mechanism (MechanismConfiguration): the mechanism to find on nexus
            ontology_access_point (NexusForgeAccessPoint): access point
                where to find the brain regions

        Returns the mechanism as a nexus resource
        """
        default_temperatures = [34, 35, 37]
        default_ljp = True

        resources = self.access_point.fetch(
            {"type": "SubCellularModelScript", "name": mechanism.name}
        )
        if resources is None:
            raise AccessPointException(f"SubCellularModelScript {mechanism.name} not found")

        # brain region filtering
        br_visited = set()
        filtered_resources, br_visited = filter_mechanisms_with_brain_region(
            ontology_access_point.forge,
            resources,
            self.emodel_metadata_ontology.brain_region["brainRegion"]["label"],
            br_visited,
        )
        br_visited_to_str = ", ".join(br_visited)
        error_msg = f"brain region in ({br_visited_to_str})"
        if filtered_resources is not None:
            resources = filtered_resources

        # temperature filtering
        if mechanism.temperature is not None:
            error_msg += f"temperature = {mechanism.temperature} "
            filtered_resources = [
                r
                for r in resources
                if hasattr(r, "temperature")
                and getattr(r.temperature, "value", r.temperature) == mechanism.temperature
            ]
            if len(filtered_resources) > 0:
                resources = filtered_resources

        # species filtering
        error_msg += f"species = {self.emodel_metadata_ontology.species['species']['label']} "
        filtered_resources = [
            r
            for r in resources
            if hasattr(r, "subject")
            and r.subject.species.label == self.emodel_metadata_ontology.species["species"]["label"]
        ]
        if len(filtered_resources) > 0:
            resources = filtered_resources

        # ljp correction filtering
        if mechanism.ljp_corrected is not None:
            error_msg += f"ljp correction = {mechanism.ljp_corrected} "
            filtered_resources = [
                r
                for r in resources
                if hasattr(r, "isLjpCorrected") and r.isLjpCorrected == mechanism.ljp_corrected
            ]
            if len(filtered_resources) > 0:
                resources = filtered_resources

        if len(resources) == 0:
            raise AccessPointException(
                f"SubCellularModelScript {mechanism.name} not found with {error_msg}"
            )

        # use default values
        if len(resources) > 1:
            logger.warning(
                "More than one resource fetched for mechanism %s",
                mechanism.name,
            )
        if len(resources) > 1 and mechanism.temperature is None:
            resources = discriminate_by_temp(resources, default_temperatures)

        if len(resources) > 1 and mechanism.ljp_corrected is None:
            tmp_resources = [
                r
                for r in resources
                if hasattr(r, "isLjpCorrected") and r.isLjpCorrected is default_ljp
            ]
            if len(tmp_resources) > 0 and len(tmp_resources) < len(resources):
                logger.warning(
                    "Discriminating resources based on ljp correction. "
                    "Keeping only resource with ljp correction."
                )
                resources = tmp_resources

        if len(resources) > 1:
            logger.warning(
                "Could not reduce the number of resources fetched down to one. "
                "Keeping the 1st resource of the list."
            )

        return resources[0]

    def store_model_configuration(self, configuration, path=None):
        """Store a model configuration as a resource of type EModelConfiguration"""

        # Search for all Morphologies on Nexus and add their Nexus ids to the configuration
        morphologies = self.access_point.fetch({"type": "NeuronMorphology"})
        if not morphologies:
            raise AccessPointException(
                "Cannot find morphologies on Nexus. Please make sure that "
                "morphologies can be reached from the current Nexus session."
            )

        available_morphologies_names = [morphology.name for morphology in morphologies]

        if configuration.morphology.name in available_morphologies_names:
            configuration.morphology.id = morphologies[
                available_morphologies_names.index(configuration.morphology.name)
            ].id
        else:
            logger.warning("Morphology %s not found.", configuration.morphology.name)

        # set id to mechanisms by filtering with brain region, temperature, species, ljp correction
        ontology_access_point = ontology_forge_access_point(
            self.access_point.access_token, self.forge_ontology_path, self.access_point.endpoint
        )
        for mechanism in configuration.mechanisms:
            if mechanism.name in NEURON_BUILTIN_MECHANISMS:
                continue
            mech_resource = self.fetch_and_filter_mechanism(mechanism, ontology_access_point)
            mechanism.id = mech_resource.id

        if self.pipeline_settings.use_ProbAMPANMDA_EMS:
            ProbAMPANMDA_EMS_id = self.get_ProbAMPANMDA_EMS_resource().id
            configuration.extra_mech_ids = [(ProbAMPANMDA_EMS_id, "SynapsePhysiologyModel")]

        self.store_object(configuration)

    def get_distributions(self):
        """Get the list of available distributions"""

        return self.access_point.nexus_to_objects(
            type_="EModelChannelDistribution",
            metadata={},
            download_directory=self.download_directory,
        )[0]

    def store_distribution(self, distribution):
        """Store a channel distribution as a resource of type EModelChannelDistribution"""

        self.store_object(distribution)

    def create_emodel_workflow(self, state="not launched"):
        """Create an EModelWorkflow instance filled with the appropriate configuration"""

        try:
            targets_configuration_id = self.access_point.get_nexus_id(
                type_="ExtractionTargetsConfiguration",
                metadata=self.emodel_metadata_ontology.filters_for_resource(),
                legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
            )
        except AccessPointException:
            targets_configuration_id = None

        pipeline_settings_id = self.access_point.get_nexus_id(
            type_="EModelPipelineSettings",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
        )
        emodel_configuration_id = self.access_point.get_nexus_id(
            type_="EModelConfiguration",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
        )

        try:
            fitness_configuration_id = self.access_point.get_nexus_id(
                type_="FitnessCalculatorConfiguration",
                metadata=self.emodel_metadata_ontology.filters_for_resource(),
                legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
            )
        except AccessPointException:
            fitness_configuration_id = None

        return EModelWorkflow(
            targets_configuration_id,
            pipeline_settings_id,
            emodel_configuration_id,
            fitness_configuration_id=fitness_configuration_id,
            state=state,
        )

    def get_emodel_workflow(self):
        """Get the emodel workflow, containing configuration data and workflow status

        Returns None if the emodel workflow is not present on nexus."""

        emodel_workflow, ids = self.access_point.nexus_to_objects(
            type_="EModelWorkflow",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            download_directory=self.download_directory,
            legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
        )

        if emodel_workflow:
            return emodel_workflow[0], ids[0]

        return None, None

    def check_emodel_workflow_configurations(self, emodel_workflow):
        """Return True if the emodel workflow's configurations are on nexus, and False otherwise"""

        for id_ in emodel_workflow.get_configuration_ids():
            if id_ is not None and self.access_point.retrieve(id_) is None:
                return False

        return True

    def store_or_update_emodel_workflow(self, emodel_workflow):
        """If emodel workflow is not on nexus, store it. If it is, fetch it and update its state"""
        type_ = "EModelWorkflow"

        filters = {"type": type_}
        filters.update(self.emodel_metadata_ontology.filters_for_resource())
        filters_legacy = {"type": type_}
        filters_legacy.update(self.emodel_metadata_ontology.filters_for_resource_legacy())
        resources = self.access_point.fetch_legacy_compatible(filters, filters_legacy)

        # not present on nexus yet -> store it
        if resources is None:
            self.access_point.object_to_nexus(
                emodel_workflow,
                self.emodel_metadata_ontology.for_resource(),
                self.emodel_metadata.as_string(),
                self.emodel_metadata_ontology.filters_for_resource_legacy(),
                replace=False,
            )
        # if present on nexus -> update its state
        else:
            resource = resources[0]
            resource.state = emodel_workflow.state
            ids_dict = emodel_workflow.get_related_nexus_ids()
            if "generates" in ids_dict:
                resource.generates = ids_dict["generates"]
            if "hasPart" in ids_dict:
                resource.hasPart = ids_dict["hasPart"]

            # in case some data has been updated, e.g. fitness_configuration_id
            updated_resource = self.access_point.update_distribution(
                resource, self.emodel_metadata.as_string(), emodel_workflow
            )

            self.access_point.forge.update(updated_resource)

    def get_emodel(self, seed=None):
        """Fetch an emodel"""

        metadata = self.emodel_metadata_ontology.filters_for_resource()
        legacy_metadata = self.emodel_metadata_ontology.filters_for_resource_legacy()

        if seed is not None:
            metadata["seed"] = int(seed)
            legacy_metadata["seed"] = int(seed)

        emodel = self.access_point.nexus_to_object(
            type_="EModel",
            metadata=metadata,
            download_directory=self.download_directory,
            legacy_metadata=legacy_metadata,
        )
        emodel.emodel_metadata = copy.deepcopy(self.emodel_metadata)

        return emodel

    def store_emodel(self, emodel, description=None):
        """Store an EModel on Nexus"""

        workflow, nexus_id = self.get_emodel_workflow()

        if workflow is None:
            raise AccessPointException(
                "No EModelWorkflow available to which the EModels can be linked"
            )

        emodel.workflow_id = nexus_id
        self.store_object(emodel, seed=emodel.seed, description=description)
        # wait for the object to be uploaded and fetchable
        time.sleep(self.sleep_time)

        # fetch just uploaded emodel resource to get its id and give it to emodel workflow
        type_ = "EModel"
        filters = {"type": type_, "seed": emodel.seed}
        filters.update(self.emodel_metadata_ontology.filters_for_resource())
        filters_legacy = {"type": type_, "seed": emodel.seed}
        filters_legacy.update(self.emodel_metadata_ontology.filters_for_resource_legacy())
        resource = self.access_point.fetch_one(filters, filters_legacy)
        model_id = resource.id

        workflow.add_emodel_id(model_id)
        self.store_or_update_emodel_workflow(workflow)

    def get_emodels(self, emodels=None):
        """Get all the emodels"""

        emodels, _ = self.access_point.nexus_to_objects(
            type_="EModel",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            download_directory=self.download_directory,
            legacy_metadata=self.emodel_metadata_ontology.filters_for_resource_legacy(),
        )

        for em in emodels:
            em.emodel_metadata = copy.deepcopy(self.emodel_metadata)

        return emodels

    def has_best_model(self, seed):
        """Check if the best model has been stored."""

        try:
            self.get_emodel(seed=seed)
            return True
        except AccessPointException:
            return False

    def is_checked_by_validation(self, seed):
        """Check if the emodel with a given seed has been checked by Validation task.

        Reminder: the logic of validation is as follows:
            if None: did not go through validation
            if False: failed validation
            if True: passed validation
        """

        try:
            emodel = self.get_emodel(seed=seed)
        except AccessPointException:
            return False

        if emodel.passed_validation is True or emodel.passed_validation is False:
            return True

        return False

    def is_validated(self):
        """Check if enough models have been validated.

        Reminder: the logic of validation is as follows:
            if None: did not go through validation
            if False: failed validation
            if True: passed validation
        """

        try:
            emodels = self.get_emodels()
        except TypeError:
            return False

        n_validated = len([em for em in emodels if em.passed_validation])

        return n_validated >= self.pipeline_settings.n_model

    def has_pipeline_settings(self):
        """Returns True if pipeline settings are present on Nexus"""

        try:
            _ = self.get_pipeline_settings(strict=True)
            return True
        except AccessPointException:
            return False

    def has_fitness_calculator_configuration(self):
        """Check if the fitness calculator configuration exists"""

        try:
            _ = self.get_fitness_calculator_configuration()
            return True
        except AccessPointException:
            return False

    def has_targets_configuration(self):
        """Check if the target configuration exists"""

        try:
            _ = self.get_targets_configuration()
            return True
        except AccessPointException:
            return False

    def has_model_configuration(self):
        """Check if the model configuration exists"""

        try:
            _ = self.get_model_configuration()
            return True
        except AccessPointException:
            return False

    def get_ProbAMPANMDA_EMS_resource(self):
        """Get the ProbAMPANMDA_EMS resource from nexus."""
        resources = self.access_point.fetch(
            {"type": "SynapsePhysiologyModel", "name": "ProbAMPANMDA_EMS"}
        )
        if resources is None:
            raise AccessPointException("SynapsePhysiologyModel ProbAMPANMDA_EMS not found")

        if len(resources) > 1:
            logger.warning(
                "Could not reduce the number of resources fetched down to one. "
                "Keeping the 1st resource of the list."
            )
        return resources[0]

    def download_ProbAMPANMDA_EMS(self):
        """Download the ProbAMPANMDA_EMS mod file.

        Returns True if the mod file has been downloaded, returns False otherwise.
        """
        mechanisms_directory = self.get_mechanisms_directory()
        resource = self.get_ProbAMPANMDA_EMS_resource()

        mod_file_name = "ProbAMPANMDA_EMS.mod"
        if os.path.isfile(str(mechanisms_directory / mod_file_name)):
            return False

        filepath = self.access_point.download(
            resource.id, str(mechanisms_directory), content_type="application/neuron-mod"
        )[0]

        # Rename the file in case it's different from the name of the resource
        filepath = pathlib.Path(filepath)
        if filepath.stem != "ProbAMPANMDA_EMS":
            filepath.rename(pathlib.Path(filepath.parent / mod_file_name))

        return True

    def download_mechanisms(self, mechanisms, any_downloaded=False):
        """Download the mod files if not already downloaded"""
        # pylint: disable=protected-access

        mechanisms_directory = self.get_mechanisms_directory()
        ontology_access_point = ontology_forge_access_point(
            self.access_point.access_token, self.forge_ontology_path, self.access_point.endpoint
        )

        for mechanism in mechanisms:
            if mechanism.name in NEURON_BUILTIN_MECHANISMS:
                continue

            resource = None
            if mechanism.id is not None:
                resource = self.access_point.forge.retrieve(mechanism.id)
                if resource is not None and resource._store_metadata["_deprecated"]:
                    logger.info(
                        "Nexus resource for mechanism %s is deprecated. "
                        "Looking for a new resource...",
                        mechanism.name,
                    )
                    resource = None

            # if could not find by id, try with filtering
            if resource is None:
                resource = self.fetch_and_filter_mechanism(mechanism, ontology_access_point)

            mod_file_name = f"{mechanism.name}.mod"
            if os.path.isfile(str(mechanisms_directory / mod_file_name)):
                continue

            filepath = self.access_point.download(resource.id, str(mechanisms_directory))[0]
            any_downloaded = True
            # Rename the file in case it's different from the name of the resource
            filepath = pathlib.Path(filepath)
            if filepath.stem != mechanism:
                filepath.rename(pathlib.Path(filepath.parent / mod_file_name))

        if any_downloaded:
            previous_dir = os.getcwd()
            os.chdir(pathlib.Path(mechanisms_directory.parent))
            subprocess.run("nrnivmodl mechanisms", shell=True, check=True)
            os.chdir(previous_dir)

    def download_morphology(self, name=None, format_=None, id_=None):
        """Download a morphology by its id if provided. If no id is given,
        the function attempts to download the morphology by its name,
        provided it has not already been downloaded

        Args:
            name (str): name of the morphology resource
            format_ (str): Optional. Can be 'asc', 'swc', or 'h5'.
                Must be available in the resource.
            id_ (str): id of the nexus resource

        Raises:
            TypeError if id_ and name are not given
            AccessPointException if resource could not be retrieved
            FileNotFoundError if downloaded morphology could not be find locally
        """

        if id_ is None and name is None:
            raise TypeError("In download_morphology, at least name or id_ must be given.")

        if id_ is None:
            species_label = self.emodel_metadata_ontology.species["species"]["label"]
            resources = self.access_point.fetch(
                {
                    "type": "NeuronMorphology",
                    "name": name,
                    "subject": {"species": {"label": species_label}},
                }
            )
            if resources is None:
                raise AccessPointException(f"Could not get resource for morphology {name}")
            if len(resources) == 1:
                resource = resources[0]
            elif len(resources) >= 2:
                resource = get_curated_morphology(resources)
                if resource is None:
                    raise AccessPointException(f"Could not get resource for morphology {name}")

            res_id = resource.id
        else:
            res_id = id_

        filepath = pathlib.Path(self.access_point.download(res_id, self.download_directory)[0])

        # Some morphologies have .h5 attached and we don't want that:
        if format_:
            suffix = "." + format_
            filepath = filepath.with_suffix(suffix)
            # special case example: format_ is 'asc', but morph has '.ASC' format
            if not filepath.is_file() and filepath.with_suffix(suffix.upper()).is_file():
                filepath = filepath.with_suffix(suffix.upper())
        elif filepath.suffix == ".h5":
            for suffix in [".swc", ".asc", ".ASC"]:
                if filepath.with_suffix(suffix).is_file():
                    filepath = filepath.with_suffix(suffix)
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find morphology {filepath.stem}"
                    f"at path {filepath.parent} with allowed suffix '.asc', '.swc' or '.ASC'."
                )

        return str(filepath)

    def download_trace(self, id_=None, id_legacy=None, name=None):
        """Does not actually download the Trace since traces are already stored on Nexus"""

        for proj_traces in NEXUS_PROJECTS_TRACES:
            access_point = NexusForgeAccessPoint(
                project=proj_traces["project"],
                organisation=proj_traces["organisation"],
                endpoint="https://bbp.epfl.ch/nexus/v1",
                forge_path=self.access_point.forge_path,
                access_token=self.access_point.access_token,
                cross_bucket=True,
            )

            if id_:
                resource = access_point.retrieve(id_)
            elif id_legacy:
                resource = access_point.retrieve(id_legacy)
            elif name:
                resource = access_point.fetch_one(
                    {
                        "type": "Trace",
                        "name": name,
                        "distribution": {"encodingFormat": "application/nwb"},
                    },
                    strict=False,
                )
            else:
                raise TypeError("At least id_ or name should be informed.")

            if resource:
                return access_point.resource_location(resource, self.download_directory)[0]

        raise ValueError(f"No matching resource for {id_} {name}")

    def get_mechanisms_directory(self):
        """Return the path to the directory containing the mechanisms for the current emodel"""

        mechanisms_directory = self.download_directory / "mechanisms"

        return mechanisms_directory.resolve()

    def load_channel_gene_expression(self, name):
        """Retrieve a channel gene expression resource and read its content"""

        dataset = self.access_point.fetch_one(filters={"type": "RNASequencing", "name": name})

        filepath = self.access_point.resource_location(dataset, self.download_directory)[0]

        df = pandas.read_csv(filepath, index_col=["me-type", "t-type", "modality"])

        return df, filepath

    def load_ic_map(self):
        """Get the ion channel/genes map from Nexus"""

        resource_ic_map = self.access_point.fetch_one(
            {"type": "IonChannelMapping", "name": "icmapping"}
        )

        return self.access_point.download(resource_ic_map.id, self.download_directory)[0]

    def get_t_types(self, table_name):
        """Get the list of t-types available for the present emodel"""

        df, _ = self.load_channel_gene_expression(table_name)
        # replace non-alphanumeric characters with underscores in t-types from RNASeq data
        df["me-type"] = df["me-type"].str.replace(r"\W", "_")
        return (
            df.loc[self.emodel_metadata.emodel].index.get_level_values("t-type").unique().tolist()
        )

    def get_available_morphologies(self):
        """Get the list of names of the available morphologies"""

        resources = self.access_point.fetch({"type": "NeuronMorphology"})

        if resources:
            return {r.name for r in resources}

        logger.warning("Did not find any available morphologies.")
        return set()

    def get_available_mechanisms(self, filters=None):
        """Get all the available mechanisms.
        Optional filters can be applied to refine the search."""

        filter = {"type": "SubCellularModelScript"}
        if filters:
            filter.update(filters)
        # do not look in other projects for these resources,
        # because resources have 'full' metadata only in a few projects.
        resources = self.access_point.fetch(filter, cross_bucket=False)
        if resources is None:
            logger.warning("No SubCellularModelScript mechanisms available")
            return None

        available_mechanisms = []
        for r in resources:
            logger.debug("fetching %s mechanism from nexus.", r.name)
            version = r.modelId if hasattr(r, "modelId") else None
            temperature = (
                getattr(r.temperature, "value", r.temperature)
                if hasattr(r, "temperature")
                else None
            )
            ljp_corrected = r.isLjpCorrected if hasattr(r, "isLjpCorrected") else None
            stochastic = r.stochastic if hasattr(r, "stochastic") else None

            parameters = {}
            ion_currents = []
            non_specific_currents = []
            ionic_concentrations = []
            if hasattr(r, "exposesParameter"):
                exposes_parameters = r.exposesParameter
                if not isinstance(exposes_parameters, list):
                    exposes_parameters = [exposes_parameters]
                for ep in exposes_parameters:
                    if ep.type == "ConductanceDensity":
                        lower_limit = ep.lowerLimit if hasattr(ep, "lowerLimit") else None
                        upper_limit = ep.upperLimit if hasattr(ep, "upperLimit") else None
                        # resource name is the mech SUFFIX
                        parameters[f"{ep.name}_{r.name}"] = [lower_limit, upper_limit]
                    elif ep.type == "CurrentDensity":
                        if not hasattr(ep, "ionName"):
                            logger.warning(
                                "Will not add %s current, "
                                "because 'ionName' field was not found in %s.",
                                ep.name,
                                r.name,
                            )
                        elif ep.ionName == "non-specific":
                            non_specific_currents.append(ep.name)
                        else:
                            ion_currents.append(ep.name)
                            ionic_concentrations.append(f"{ep.ionName}i")
                    elif ep.type == "IonConcentration":
                        ionic_concentrations.append(ep.name)

            # remove duplicates
            ionic_concentrations = list(set(ionic_concentrations))

            mech = MechanismConfiguration(
                r.name,
                location=None,
                stochastic=stochastic,
                version=version,
                temperature=temperature,
                ljp_corrected=ljp_corrected,
                parameters=parameters,
                ion_currents=ion_currents,
                ionic_concentrations=ionic_concentrations,
                id=r.id,
            )

            available_mechanisms.append(mech)

        return available_mechanisms

    def get_available_traces(self, filter_species=True, filter_brain_region=False):
        """Get the list of available Traces for the current species from Nexus"""

        species = None
        if filter_species:
            species = self.emodel_metadata_ontology.species
        brain_region = None
        if filter_brain_region:
            brain_region = self.emodel_metadata_ontology.brain_region

        resource_traces = get_available_traces(
            species=species,
            brain_region=brain_region,
            access_token=self.access_point.access_token,
            forge_path=self.access_point.forge_path,
        )

        traces = []
        if resource_traces is None:
            return traces

        for r in resource_traces:
            ecodes = None
            if hasattr(r, "stimulus"):
                # is stimulus a list
                stimuli = r.stimulus if isinstance(r.stimulus, list) else [r.stimulus]
                ecodes = {
                    stim.stimulusType.label: {}
                    for stim in stimuli
                    if hasattr(stim.stimulusType, "label")
                }
            species = None
            if hasattr(r, "subject") and hasattr(r.subject, "species"):
                species = r.subject.species

            brain_region = None
            if hasattr(r, "brainLocation"):
                brain_region = r.brainLocation

            etype = None
            if hasattr(r, "annotation"):
                if isinstance(r.annotation, Resource):
                    if "e-type" in r.annotation.name.lower():
                        etype = r.annotation.hasBody.label
                else:
                    for annotation in r.annotation:
                        if "e-type" in annotation.name.lower():
                            etype = annotation.hasBody.label

            traces.append(
                TraceFile(
                    r.name,
                    filename=None,
                    filepath=None,
                    resource_id=r.id,
                    ecodes=ecodes,
                    other_metadata=None,
                    species=species,
                    brain_region=brain_region,
                    etype=etype,
                    id=r.id,
                )
            )
        return traces

    def store_morphology(self, morphology_name, morphology_path, mtype=None, reconstructed=True):
        payload = {
            "type": [
                "NeuronMorphology",
                "Dataset",
                (
                    "ReconstructedNeuronMorphology"
                    if reconstructed
                    else "PlaceholderNeuronMorphology"
                ),
            ],
            "name": pathlib.Path(morphology_path).stem,
            "objectOfStudy": {
                "@id": "http://bbp.epfl.ch/neurosciencegraph/taxonomies/objectsofstudy/singlecells",
                "label": "Single Cell",
            },
        }

        if mtype:
            payload["annotation"] = (
                {
                    "@type": ["Annotation", "nsg:MTypeAnnotation"],
                    "hasBody": {
                        "@id": "nsg:InhibitoryNeuron",
                        "@type": ["Mtype", "AnnotationBody"],
                        "label": mtype,
                        "prefLabel": mtype,
                    },
                    "name": "M-type Annotation",
                },
            )

        self.access_point.register(
            resource_description=payload,
            distributions=[morphology_path],
        )

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
        """Store hoc files on nexus.

        Args:
            export_function (function): can be export_emodels_hoc or export_emodels_sonata
        """
        workflow, workflow_id = self.get_emodel_workflow()
        if workflow is None:
            raise AccessPointException(
                "No EModelWorkflow available to which the EModelScripts can be linked"
            )

        emodels = self.get_emodels()
        emodels = select_emodels(
            self.emodel_metadata.emodel,
            emodels,
            only_validated=only_validated,
            only_best=only_best,
            seeds=seeds,
            iteration=self.emodel_metadata.iteration,
        )

        if not emodels:
            logger.warning(
                "No emodels selected in store_hocs. No hoc file will be registered on nexus."
            )

        hold_key = "SearchHoldingCurrent.soma.v.bpo_holding_current"
        thres_key = "SearchThresholdCurrent.soma.v.bpo_threshold_current"

        # maybe use map here?
        for emodel in emodels:
            if new_emodel_name is not None:
                emodel.emodel_metadata.emodel = new_emodel_name

            # in case hocs have been created with the local output path
            copy_hocs_to_new_output_path(emodel, output_base_dir)

            output_path = get_output_path(
                emodel, output_base_dir=output_base_dir, use_allen_notation=True
            )

            hoc_file_path = pathlib.Path(get_hoc_file_path(output_path)).resolve()
            if not hoc_file_path.is_file():
                logger.warning(
                    "Could not find the hoc file for %s. "
                    "Will not register EModelScript on nexus.",
                    emodel.emodel_metadata.emodel,
                )
                continue

            currents = None
            if emodel.features is not None:
                if hold_key in emodel.features and thres_key in emodel.features:
                    currents = {
                        "holding": emodel.features[hold_key],
                        "threshold": emodel.features[thres_key],
                    }

            emodelscript = EModelScript(str(hoc_file_path), emodel.seed, workflow_id)
            self.store_object(
                emodelscript,
                seed=emodel.seed,
                description=description,
                currents=currents,
            )
            # wait for the object to be uploaded and fetchable
            time.sleep(self.sleep_time)

            # fetch just uploaded emodelscript resource to get its id
            type_ = "EModelScript"
            filters = {"type": type_, "seed": emodel.seed}
            filters.update(self.emodel_metadata_ontology.filters_for_resource())
            filters_legacy = {"type": type_, "seed": emodel.seed}
            filters_legacy.update(self.emodel_metadata_ontology.filters_for_resource_legacy())
            resource = self.access_point.fetch_one(filters, filters_legacy, strict=True)
            modelscript_id = resource.id
            workflow.add_emodel_script_id(modelscript_id)

        time.sleep(self.sleep_time)
        self.store_or_update_emodel_workflow(workflow)

    def store_emodels_hoc(
        self,
        only_validated=False,
        only_best=True,
        seeds=None,
        map_function=map,
        new_emodel_name=None,
        description=None,
    ):
        self.store_hocs(
            only_validated,
            only_best,
            seeds,
            map_function,
            new_emodel_name,
            description,
            "export_emodels_hoc",
        )

    def store_emodels_sonata(
        self,
        only_validated=False,
        only_best=True,
        seeds=None,
        map_function=map,
        new_emodel_name=None,
        description=None,
    ):
        self.store_hocs(
            only_validated,
            only_best,
            seeds,
            map_function,
            new_emodel_name,
            description,
            "export_emodels_sonata",
        )

    def get_hoc(self, seed=None):
        """Get the EModelScript resource"""
        metadata = self.emodel_metadata_ontology.filters_for_resource()
        legacy_metadata = self.emodel_metadata_ontology.filters_for_resource_legacy()
        if seed is not None:
            metadata["seed"] = int(seed)
            legacy_metadata["seed"] = int(seed)

        hoc = self.access_point.nexus_to_object(
            type_="EModelScript",
            metadata=metadata,
            download_directory=self.download_directory,
            legacy_metadata=legacy_metadata,
        )
        return hoc

    def sonata_exists(self, seed):
        """Returns True if the sonata hoc file has been exported"""
        try:
            _ = self.get_hoc(seed)
            return True
        except AccessPointException:
            return False
