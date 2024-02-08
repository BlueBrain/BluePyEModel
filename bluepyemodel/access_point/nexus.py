"""Access point using Nexus Forge"""

import copy
import logging
import os
import pathlib
import subprocess
import time

import pandas
from kgforge.core import Resource
from bluepyemodel.access_point.access_point import DataAccessPoint
from bluepyemodel.access_point.forge_access_point import NEXUS_PROJECTS_TRACES
from bluepyemodel.access_point.forge_access_point import AccessPointException
from bluepyemodel.access_point.forge_access_point import NexusForgeAccessPoint
from bluepyemodel.access_point.forge_access_point import check_resource
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
        )

        self.pipeline_settings = self.get_pipeline_settings(strict=False)

        directory_name = self.emodel_metadata.as_string()
        (pathlib.Path("./nexus_temp/") / directory_name).mkdir(parents=True, exist_ok=True)

        self.sleep_time = sleep_time

    def check_mettypes(self):
        """Check that etype, mtype and ttype are present on nexus"""
        ontology_access_point = ontology_forge_access_point(
            self.access_point.access_token, self.forge_ontology_path
        )

        logger.info("Checking if etype %s is present on nexus...", self.emodel_metadata.etype)
        check_resource(
            self.emodel_metadata.etype,
            "etype",
            access_point=ontology_access_point,
            access_token=self.access_point.access_token,
            forge_path=self.forge_ontology_path,
        )
        logger.info("Etype checked")

        if self.emodel_metadata.mtype is not None:
            logger.info("Checking if mtype %s is present on nexus...", self.emodel_metadata.mtype)
            check_resource(
                self.emodel_metadata.mtype,
                "mtype",
                access_point=ontology_access_point,
                access_token=self.access_point.access_token,
                forge_path=self.forge_ontology_path,
            )
            logger.info("Mtype checked")
        else:
            logger.info("Mtype is None, its presence on Nexus is not being checked.")

        if self.emodel_metadata.ttype is not None:
            logger.info("Checking if ttype %s is present on nexus...", self.emodel_metadata.ttype)
            check_resource(
                self.emodel_metadata.ttype,
                "ttype",
                access_point=ontology_access_point,
                access_token=self.access_point.access_token,
                forge_path=self.forge_ontology_path,
            )
            logger.info("Ttype checked")
        else:
            logger.info("Ttype is None, its presence on Nexus is not being checked.")

    def get_pipeline_settings(self, strict=True):
        if strict:
            return self.access_point.nexus_to_object(
                type_="EModelPipelineSettings",
                metadata=self.emodel_metadata_ontology.filters_for_resource(),
                metadata_str=self.emodel_metadata.as_string(),
            )

        try:
            return self.access_point.nexus_to_object(
                type_="EModelPipelineSettings",
                metadata=self.emodel_metadata_ontology.filters_for_resource(),
                metadata_str=self.emodel_metadata.as_string(),
            )
        except AccessPointException:
            return EModelPipelineSettings()

    def store_pipeline_settings(self, pipeline_settings):
        """Save an EModelPipelineSettings on Nexus"""

        self.access_point.object_to_nexus(
            pipeline_settings,
            self.emodel_metadata_ontology.for_resource(),
            self.emodel_metadata.as_string(),
            replace=True,
        )

    def build_ontology_based_metadata(self):
        """Get the ontology related to the metadata"""

        self.emodel_metadata_ontology.species = self.get_nexus_subject(self.emodel_metadata.species)
        self.emodel_metadata_ontology.brain_region = get_nexus_brain_region(
            self.emodel_metadata.brain_region,
            self.access_point.access_token,
            self.forge_ontology_path,
        )

    def get_nexus_subject(self, species):
        """Get the ontology of a species based on the species name."""

        if species == "human":
            subject = {
                "type": "Subject",
                "species": {
                    "id": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
                    "label": "Homo sapiens",
                },
            }

        elif species == "rat":
            subject = {
                "type": "Subject",
                "species": {
                    "id": "http://purl.obolibrary.org/obo/NCBITaxon_10116",
                    "label": "Rattus norvegicus",
                },
            }

        elif species == "mouse":
            subject = {
                "type": "Subject",
                "species": {
                    "id": "http://purl.obolibrary.org/obo/NCBITaxon_10090",
                    "label": "Mus musculus",
                },
            }
        elif species is None:
            subject = None
        else:
            raise ValueError(f"Unknown species {species}.")

        return subject

    def store_object(self, object_, seed=None, description=None):
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
            replace=True,
            seed=seed,
        )

    def get_targets_configuration(self):
        """Get the configuration of the targets (targets and ephys files used)"""

        configuration = self.access_point.nexus_to_object(
            type_="ExtractionTargetsConfiguration",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            metadata_str=self.emodel_metadata.as_string(),
        )

        configuration.available_traces = self.get_available_traces()
        configuration.available_efeatures = self.get_available_efeatures()

        if not configuration.files:
            logger.debug(
                "Empty list of files in the TargetsConfiguration, filling"
                "it using what is available on Nexus for the present etype."
            )
            configuration.files = configuration.available_traces

        for file in configuration.files:
            file.filepath = self.download_trace(id_=file.resource_id, name=file.filename)

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

        self.store_object(configuration)

    def get_fitness_calculator_configuration(self, record_ions_and_currents=False):
        """Get the configuration of the fitness calculator (efeatures and protocols)"""

        configuration = self.access_point.nexus_to_object(
            type_="FitnessCalculatorConfiguration",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            metadata_str=self.emodel_metadata.as_string(),
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
        resource = self.access_point.fetch_one(filters)
        fitness_id = resource.id

        workflow.fitness_configuration_id = fitness_id
        self.store_or_update_emodel_workflow(workflow)

    def get_model_configuration(self):
        """Get the configuration of the model, including parameters, mechanisms and distributions"""

        configuration = self.access_point.nexus_to_object(
            type_="EModelConfiguration",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            metadata_str=self.emodel_metadata.as_string(),
        )

        morph_path = self.download_morphology(
            configuration.morphology.name, configuration.morphology.format
        )
        self.download_mechanisms(configuration.mechanisms)

        configuration.morphology.path = morph_path
        configuration.available_mechanisms = self.get_available_mechanisms()
        configuration.available_morphologies = self.get_available_morphologies()

        return configuration

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

        # Search for all Mechanisms on Nexus and add their Nexus ids to the configuration
        mechanisms = self.get_available_mechanisms()
        if not mechanisms:
            raise AccessPointException(
                "Cannot find mechanisms on Nexus. Please make sure that "
                "mechanisms can be reached from the current Nexus session."
            )

        available_mechanisms_names = [mechanism.name for mechanism in mechanisms]

        for mechanism in configuration.mechanisms:
            if mechanism.name in available_mechanisms_names:
                mechanism.id = mechanisms[available_mechanisms_names.index(mechanism.name)].id

        self.store_object(configuration)

    def get_distributions(self):
        """Get the list of available distributions"""

        return self.access_point.nexus_to_objects(
            type_="EModelChannelDistribution",
            metadata={},
            metadata_str=self.emodel_metadata.as_string(),
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
            )
        except AccessPointException:
            targets_configuration_id = None

        pipeline_settings_id = self.access_point.get_nexus_id(
            type_="EModelPipelineSettings",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
        )
        emodel_configuration_id = self.access_point.get_nexus_id(
            type_="EModelConfiguration",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
        )

        try:
            fitness_configuration_id = self.access_point.get_nexus_id(
                type_="FitnessCalculatorConfiguration",
                metadata=self.emodel_metadata_ontology.filters_for_resource(),
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
            metadata_str=self.emodel_metadata.as_string(),
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
        resources = self.access_point.fetch(filters)

        # not present on nexus yet -> store it
        if resources is None:
            self.access_point.object_to_nexus(
                emodel_workflow,
                self.emodel_metadata_ontology.for_resource(),
                self.emodel_metadata.as_string(),
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

        if seed is not None:
            metadata["seed"] = int(seed)

        emodel = self.access_point.nexus_to_object(
            type_="EModel",
            metadata=metadata,
            metadata_str=self.emodel_metadata.as_string(),
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
        resource = self.access_point.fetch_one(filters)
        model_id = resource.id

        workflow.add_emodel_id(model_id)
        self.store_or_update_emodel_workflow(workflow)

    def get_emodels(self, emodels=None):
        """Get all the emodels"""

        emodels, _ = self.access_point.nexus_to_objects(
            type_="EModel",
            metadata=self.emodel_metadata_ontology.filters_for_resource(),
            metadata_str=self.emodel_metadata.as_string(),
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

    def download_mechanisms(self, mechanisms):
        """Download the mod files if not already downloaded"""
        default_temperatures = [34, 35, 37]
        default_ljp = True

        mechanisms_directory = self.get_mechanisms_directory()

        any_downloaded = False
        for mechanism in mechanisms:
            if mechanism.name in NEURON_BUILTIN_MECHANISMS:
                continue

            resources = self.access_point.fetch(
                {"type": "SubCellularModelScript", "name": mechanism.name}
            )
            if resources is None:
                raise AccessPointException(f"SubCellularModelScript {mechanism.name} not found")

            error_msg = ""
            if mechanism.version is not None:
                error_msg += f"version = {mechanism.version}"
                resources = [r for r in resources if r.modelId == mechanism.version]
            if mechanism.temperature is not None:
                error_msg += f"temperature = {mechanism.temperature}"
                resources = [r for r in resources if r.temperature.value == mechanism.temperature]
            if mechanism.ljp_corrected is not None:
                error_msg += f"ljp correction = {mechanism.ljp_corrected}"
                resources = [r for r in resources if r.isLjpCorrected == mechanism.ljp_corrected]

            if len(resources) == 0:
                raise AccessPointException(
                    f"SubCellularModelScript {mechanism.name} not found with {error_msg}"
                )

            # use default values
            if len(resources) > 1:
                logger.warning("More than one resource fetched for mechanism %s", mechanism.name)
            if len(resources) > 1 and mechanism.temperature is None:
                resources = discriminate_by_temp(resources, default_temperatures)

            if len(resources) > 1 and mechanism.ljp_corrected is None:
                tmp_resources = [r for r in resources if r.isLjpCorrected is default_ljp]
                if len(tmp_resources) > 0 and len(tmp_resources) < len(resources):
                    logger.warning(
                        "Discriminating resources based on ljp correction. "
                        "Keeping only resource with ljp correction."
                    )
                    resources = tmp_resources

            # use latest version
            if len(resources) > 1 and all(hasattr(r, "modelId") for r in resources):
                logger.warning(
                    "Discriminating resources based on version. Keeping only the latest version."
                )
                resource = sorted(resources, key=lambda x: x.modelId)[-1]
            else:
                if len(resources) > 1:
                    logger.warning(
                        "Could not reduce the number of resources fetched down to one. "
                        "Keeping the 1st resource of the list."
                    )
                resource = resources[0]

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

    def download_morphology(self, name, format_=None):
        """Download a morphology by name if not already downloaded"""

        # Temporary fix for the duplicated morphology issue
        resources = self.access_point.fetch({"type": "NeuronMorphology", "name": name})
        if resources is None:
            raise AccessPointException(f"Could not get resource for morphology {name}")
        if len(resources) == 1:
            resource = resources[0]
        elif len(resources) == 2:
            resource = get_curated_morphology(resources)
            if resource is None:
                raise AccessPointException(f"Could not get resource for morphology {name}")
        else:
            raise AccessPointException(f"More than 2 morphologies with name: {name}")

        directory_name = self.emodel_metadata.as_string()
        filepath = pathlib.Path(
            self.access_point.download(resource.id, pathlib.Path("./nexus_temp/") / directory_name)[
                0
            ]
        )

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

    def download_trace(self, id_=None, name=None):
        """Does not actually download the Trace since traces are already stored on Nexus"""

        for proj_traces in NEXUS_PROJECTS_TRACES:
            access_point = NexusForgeAccessPoint(
                project=proj_traces["project"],
                organisation=proj_traces["organisation"],
                endpoint="https://bbp.epfl.ch/nexus/v1",
                forge_path=self.access_point.forge_path,
                access_token=self.access_point.access_token,
                cross_bucket=False,
            )

            if id_:
                resource = access_point.retrieve(id_)
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
                metadata_str = self.emodel_metadata.as_string()
                return access_point.resource_location(resource, metadata_str=metadata_str)[0]

        raise ValueError(f"No matching resource for {id_} {name}")

    def get_mechanisms_directory(self):
        """Return the path to the directory containing the mechanisms for the current emodel"""

        directory_name = self.emodel_metadata.as_string()

        mechanisms_directory = pathlib.Path("./nexus_temp/") / directory_name / "mechanisms"

        return mechanisms_directory.resolve()

    def load_channel_gene_expression(self, name):
        """Retrieve a channel gene expression resource and read its content"""

        dataset = self.access_point.fetch_one(filters={"type": "RNASequencing", "name": name})

        metadata_str = self.emodel_metadata.as_string()
        filepath = self.access_point.resource_location(dataset, metadata_str=metadata_str)[0]

        df = pandas.read_csv(filepath, index_col=["me-type", "t-type", "modality"])

        return df, filepath

    def load_ic_map(self):
        """Get the ion channel/genes map from Nexus"""

        resource_ic_map = self.access_point.fetch_one(
            {"type": "IonChannelMapping", "name": "icmapping"}
        )

        return self.access_point.download(
            resource_ic_map.id, metadata_str=self.emodel_metadata.as_string()
        )[0]

    def get_t_types(self, table_name):
        """Get the list of t-types available for the present emodel"""

        df, _ = self.load_channel_gene_expression(table_name)
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

    def get_available_mechanisms(self):
        """Get all the available mechanisms"""

        resources = self.access_point.fetch({"type": "SubCellularModelScript"})

        if resources is None:
            logger.warning("No SubCellularModelScript mechanisms available")
            return None

        available_mechanisms = []
        for r in resources:
            version = r.modelId if hasattr(r, "modelId") else None
            temperature = (
                getattr(r.temperature, "value", r.temperature)
                if hasattr(r, "temperature")
                else None
            )
            ljp_corrected = r.isLjpCorrected if hasattr(r, "isLjpCorrected") else None
            stochastic = r.stochastic if hasattr(r, "stochastic") else None

            parameters = {}
            if hasattr(r, "exposesParameter"):
                exposes_parameters = r.exposesParameter
                if not isinstance(exposes_parameters, list):
                    exposes_parameters = [exposes_parameters]
                for ep in exposes_parameters:
                    if ep.type == "ConductanceDensity":
                        lower_limit = ep.lowerLimit if hasattr(ep, "lowerLimit") else None
                        upper_limit = ep.upperLimit if hasattr(ep, "upperLimit") else None
                        if hasattr(r, "mod"):
                            parameters[f"{ep.name}_{r.mod.suffix}"] = [lower_limit, upper_limit]
                        else:
                            parameters[f"{ep.name}_{r.nmodlParameters.suffix}"] = [
                                lower_limit,
                                upper_limit,
                            ]

            ion_currents = []
            ionic_concentrations = []
            # technically, also adds non-specific currents to ion_currents list,
            # because they are not distinguished in nexus for now, but
            # the code should work nevertheless
            ions = []
            if hasattr(r, "mod") and hasattr(r.mod, "ion"):
                ions = r.mod.ion if isinstance(r.mod.ion, list) else [r.mod.ion]
            elif hasattr(r, "ion"):
                ions = r.ion if isinstance(r.ion, list) else [r.ion]

            for ion in ions:
                ion_name = ion.label.lower()
                ion_currents.append(f"i{ion_name}")
                ionic_concentrations.append(f"{ion_name}i")

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
                ecodes = {
                    stim.stimulusType.label: {}
                    for stim in r.stimulus
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

            if etype is not None and etype == self.emodel_metadata.etype:
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
                "ReconstructedNeuronMorphology" if reconstructed else "SynthesizedNeuronMorphology",
            ],
            "name": pathlib.Path(morphology_path).stem,
            "objectOfStudy": {
                "@id": "http://bbp.epfl.ch/neurosciencegraph/taxonomies/objectsofstudy/singlecells",
                "@type": "ObjectOfStudy",
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

            emodelscript = EModelScript(str(hoc_file_path), emodel.seed, workflow_id)
            self.store_object(emodelscript, seed=emodel.seed, description=description)
            # wait for the object to be uploaded and fetchable
            time.sleep(self.sleep_time)

            # fetch just uploaded emodelscript resource to get its id
            type_ = "EModelScript"
            filters = {"type": type_, "seed": emodel.seed}
            filters.update(self.emodel_metadata_ontology.filters_for_resource())
            resource = self.access_point.fetch_one(filters, strict=True)
            modelscript_id = resource.id
            workflow.add_emodel_script_id(modelscript_id)

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

    def get_hoc(self, seed):
        """Get the EModelScript resource"""
        metadata = self.emodel_metadata_ontology.filters_for_resource()
        if seed is not None:
            metadata["seed"] = int(seed)

        hoc = self.access_point.nexus_to_object(
            type_="EModelScript",
            metadata=metadata,
            metadata_str=self.emodel_metadata.as_string(),
        )
        return hoc

    def sonata_exists(self, seed):
        """Returns True if the sonata hoc file has been exported"""
        try:
            _ = self.get_hoc(seed)
            return True
        except AccessPointException:
            return False
