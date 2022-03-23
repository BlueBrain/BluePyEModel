"""Access point using Nexus Forge"""
import copy
import logging
import os
import pathlib

import pandas

from bluepyemodel.access_point.access_point import DataAccessPoint
from bluepyemodel.access_point.forge_access_point import AccessPointException
from bluepyemodel.access_point.forge_access_point import NexusForgeAccessPoint
from bluepyemodel.efeatures_extraction.trace_file import TraceFile
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings

# pylint: disable=too-many-arguments,unused-argument,too-many-locals
from bluepyemodel.model.mechanism_configuration import MechanismConfiguration

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
        project="emodel_pipeline",
        organisation="demo",
        endpoint="https://bbp.epfl.ch/nexus/v1",
        forge_path=None,
        access_token=None,
    ):
        """Init

        Args:
            emodel (str): name of the emodel
            etype (str): name of the electric type.
            ttype (str): name of the transcriptomic type.
            mtype (str): name of the morphology type.
            species (str): name of the species.
            brain_region (str): name of the brain location.
            iteration_tag (str): tag associated to the current run.
            project (str): name of the Nexus project.
            organisation (str): name of the Nexus organization to which the project belong.
            endpoint (str): Nexus endpoint.
            forge_path (str): path to a .yml used as configuration by nexus-forge.
            ttype (str): name of the t-type. Required if using the gene expression or IC selector.
            access_token (str): Nexus connection token.
        """

        super().__init__(emodel, etype, ttype, mtype, species, brain_region, iteration_tag)

        self.access_point = NexusForgeAccessPoint(
            project=project,
            organisation=organisation,
            endpoint=endpoint,
            forge_path=forge_path,
            access_token=access_token,
        )

        # This trick is used to have nexus type descriptions on one side and basic
        # strings on the other
        self.emodel_metadata_ontology = copy.deepcopy(self.emodel_metadata)
        self.build_ontology_based_metadata()

        self.pipeline_settings = self.get_pipeline_settings(strict=False)

    def get_pipeline_settings(self, strict=True):

        if strict:
            return self.access_point.nexus_to_object(
                type_="EModelPipelineSettings",
                metadata=self.emodel_metadata_ontology.for_resource(),
            )

        try:
            return self.access_point.nexus_to_object(
                type_="EModelPipelineSettings",
                metadata=self.emodel_metadata_ontology.for_resource(),
            )
        except AccessPointException:
            return None

    def store_pipeline_settings(
        self,
        extraction_threshold_value_save=1,
        efel_settings=None,
        stochasticity=False,
        morph_modifiers=None,
        threshold_based_evaluator=True,
        optimizer="IBEA",
        optimisation_params=None,
        optimisation_timeout=600.0,
        threshold_efeature_std=0.05,
        max_ngen=100,
        validation_threshold=5.0,
        optimization_batch_size=5,
        max_n_batch=3,
        n_model=3,
        name_gene_map=None,
        plot_extraction=True,
        plot_optimisation=True,
        compile_mechanisms=False,
        name_Rin_protocol=None,
        name_rmp_protocol=None,
        validation_protocols=None,
    ):
        """Creates an EModelPipelineSettings resource.

        Args:
            extraction_threshold_value_save (int): name of the mechanism.
            efel_settings (dict): efel settings in the form {setting_name: setting_value}.
                If settings are also informed in the targets per efeature, the latter
                will have priority.
            stochasticity (bool): should channels behave stochastically if they can.
            morph_modifiers (list): List of morphology modifiers. Each modifier has to be
                informed by the path the file containing the modifier and the name of the
                function. E.g: morph_modifiers = [["path_to_module", "name_of_function"], ...].
            threshold_based_evaluator (bool): if the evaluator is threshold-based. All
                protocol's amplitude and holding current will be rescaled by the ones of the
                models. If True, name_Rin_protocol and name_rmp_protocol have to be informed.
            optimizer (str): algorithm used for optimization, can be "IBEA", "SO-CMA",
                "MO-CMA" (use cma option in pip install for CMA optimizers).
            optimisation_params (dict): optimisation parameters. Keys have to match the
                optimizer's call. E.g., for optimizer MO-CMA:
                {"offspring_size": 10, "weight_hv": 0.4}
            optimisation_timeout (float): duration (in second) after which the evaluation
                of a protocol will be interrupted.
            max_ngen (int): maximum number of generations of the evolutionary process of the
                optimization.
            validation_threshold (float): score threshold under which the emodel passes
                validation.
            optimization_batch_size (int): number of optimisation seeds to run in parallel.
            max_n_batch (int): maximum number of optimisation batches.
            n_model (int): minimum number of models to pass validation
                to consider the EModel building task done.
            plot_extraction (bool): should the efeatures and experimental traces be plotted.
            plot_optimisation (bool): should the EModel scores and traces be plotted.
            validation_protocols (dict): names and targets of the protocol that will be used for
                validation only. This settings has to be set before efeature extraction if you
                wish to run validation.
        """

        if efel_settings is None:
            efel_settings = {"interp_step": 0.025, "strict_stiminterval": True}

        if optimisation_params is None:
            optimisation_params = {"offspring_size": 100}

        pipeline_settings = EModelPipelineSettings(
            extraction_threshold_value_save=extraction_threshold_value_save,
            efel_settings=efel_settings,
            stochasticity=stochasticity,
            threshold_based_evaluator=threshold_based_evaluator,
            optimizer=optimizer,
            optimisation_params=optimisation_params,
            optimisation_timeout=optimisation_timeout,
            max_ngen=max_ngen,
            validation_threshold=validation_threshold,
            optimisation_batch_size=optimization_batch_size,
            max_n_batch=max_n_batch,
            n_model=n_model,
            name_gene_map=name_gene_map,
            plot_extraction=plot_extraction,
            plot_optimisation=plot_optimisation,
            threshold_efeature_std=threshold_efeature_std,
            morph_modifiers=morph_modifiers,
            compile_mechanisms=compile_mechanisms,
            name_Rin_protocol=name_Rin_protocol,
            name_rmp_protocol=name_rmp_protocol,
            validation_protocols=validation_protocols,
        )

        self.access_point.object_to_nexus(
            pipeline_settings, self.emodel_metadata_ontology.for_resource(), replace=True
        )

    def build_ontology_based_metadata(self):
        """Get the ontology related to the metadata"""

        self.emodel_metadata_ontology.species = self.get_nexus_subject(self.emodel_metadata.species)
        self.emodel_metadata_ontology.brain_region = self.get_nexus_brain_region(
            self.emodel_metadata.brain_region
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
                    "id": "http://purl.obolibrary.org/obo/NCBITaxon_7370",
                    "label": "Musca domestica",
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

        else:
            raise Exception(f"Unknown species {species}.")

        return subject

    def get_nexus_brain_region(self, brain_region):
        """Get the ontology of the brain location."""

        # TODO:
        # if not self.access_token:
        #    self.access_token = get_access_token()

        # forge_CCFv3 = connect_forge(bucket, endpoint, self.access_token)
        # forge.resolve(
        #    "CA1",
        #    scope="brainRegion",
        #    strategy=ResolvingStrategy.EXACT_MATCH
        # )

        return {
            "type": "BrainLocation",
            "brainRegion": {
                # "id": "http://purl.obolibrary.org/obo/UBERON_0003881",
                "label": brain_region
            },
        }

    def store_object(self, object_, metadata=None):
        """Store a BPEM object on Nexus"""

        if metadata is None:
            metadata = self.emodel_metadata_ontology.for_resource()
        self.access_point.object_to_nexus(object_, metadata, replace=True)

    def get_targets_configuration(self):
        """Get the configuration of the targets (targets and ephys files used)"""

        configuration = self.access_point.nexus_to_object(
            type_="ExtractionTargetsConfiguration",
            metadata=self.emodel_metadata_ontology.for_resource(),
        )

        configuration.available_traces = self.get_available_traces()
        configuration.available_efeatures = self.get_available_efeatures()

        for file in configuration.files:
            file.filepath = self.download_trace(id_=file.resource_id, name=file.filename)

        return configuration

    def store_targets_configuration(self, configuration):
        """Store the configuration of the targets (targets and ephys files used)"""

        self.store_object(configuration)

    def get_fitness_calculator_configuration(self, record_ions_and_currents=False):
        """Get the configuration of the fitness calculator (efeatures and protocols)"""

        configuration = self.access_point.nexus_to_object(
            type_="FitnessCalculatorConfiguration",
            metadata=self.emodel_metadata_ontology.for_resource(),
        )

        configuration.name_rmp_protocol = self.pipeline_settings.name_rmp_protocol
        configuration.name_rin_protocol = self.pipeline_settings.name_Rin_protocol
        configuration.threshold_efeature_std = self.pipeline_settings.threshold_efeature_std
        configuration.validation_protocols = self.pipeline_settings.validation_protocols

        return configuration

    def store_fitness_calculator_configuration(self, configuration):
        """Store a fitness calculator configuration as a resource of type
        FitnessCalculatorConfiguration"""

        self.store_object(configuration)

    def get_model_configuration(self):
        """Get the configuration of the model, including parameters, mechanisms and distributions"""

        configuration = self.access_point.nexus_to_object(
            type_="EModelConfiguration", metadata=self.emodel_metadata_ontology.for_resource()
        )

        morph_path = self.download_morphology(configuration.morphology.name)
        self.download_mechanisms(configuration.mechanisms)

        configuration.morphology.path = morph_path
        configuration.available_mechanisms = self.get_available_mechanisms()
        configuration.available_morphologies = self.get_available_morphologies()

        return configuration

    def store_model_configuration(self, configuration, path=None):
        """Store a model configuration as a resource of type EModelConfiguration"""

        self.store_object(configuration)

    def get_distributions(self):
        """Get the list of available distributions"""

        return self.access_point.nexus_to_objects(type_="EModelChannelDistribution", metadata={})

    def store_distribution(self, distribution):
        """Store a channel distribution as a resource of type EModelChannelDistribution"""

        self.store_object(distribution, metadata={})

    def get_emodel(self, seed=None):
        """Fetch an emodel"""

        metadata = self.emodel_metadata_ontology.for_resource()

        if seed:
            metadata["seed"] = int(seed)

        emodel = self.access_point.nexus_to_object(type_="EModel", metadata=metadata)
        emodel.emodel_metadata = copy.deepcopy(self.emodel_metadata)

        return emodel

    def store_emodel(self, emodel):
        """Store an EModel on Nexus"""

        metadata = self.emodel_metadata_ontology.for_resource()
        metadata["seed"] = emodel.seed

        self.store_object(emodel, metadata)

    def get_emodels(self, emodels=None):
        """Get all the emodels"""

        emodels = self.access_point.nexus_to_objects(
            type_="EModel", metadata=self.emodel_metadata_ontology.for_resource()
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

        if emodel.passed_validation:
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

    def has_fitness_calculator_configuration(self):
        """Check if the fitness calculator configuration exists"""

        try:
            _ = self.get_fitness_calculator_configuration()
            return True
        except AccessPointException:
            return False

    def download_mechanisms(self, mechanisms):
        """Download the mod files if not already downloaded"""

        mechanisms_directory = self.get_mechanisms_directory()

        for mechanism in mechanisms:

            if mechanism.name == "pas":
                continue

            if mechanism.version is not None:
                resource = self.access_point.fetch_one(
                    {
                        "type": "SubCellularModelScript",
                        "name": mechanism.name,
                        "version": mechanism.version,
                    },
                )

            else:
                resources = self.access_point.fetch(
                    {"type": "SubCellularModelScript", "name": mechanism.name}
                )

                # If version not specified, we take the most recent one:
                if len(resources) > 1 and all(hasattr(r, "version") for r in resources):
                    resource = sorted(resources, key=lambda x: x.version)[-1]
                else:
                    resource = resources[0]

            mod_file_name = f"{mechanism.name}.mod"
            if os.path.isfile(str(mechanisms_directory / mod_file_name)):
                continue

            filepath = self.access_point.download(resource.id, str(mechanisms_directory))

            # Rename the file in case it's different from the name of the resource
            filepath = pathlib.Path(filepath)
            if filepath.stem != mechanism:
                filepath.rename(pathlib.Path(filepath.parent / mod_file_name))

    def download_morphology(self, name):
        """Download a morphology by name if not already downloaded"""

        resource = self.access_point.fetch_one({"type": "NeuronMorphology", "name": name})

        return self.access_point.download(resource.id, "./nexus_temp/")

    def download_trace(self, id_=None, name=None):
        """Does not actually download the Trace since traces are already stored on Nexus"""

        # TODO: actually download the Trace if trace is not available in local

        if id_:
            resource = self.access_point.retrieve(id_)
        elif name:
            resource = self.access_point.fetch_one(
                {
                    "type": "Trace",
                    "name": name,
                    "distribution": {"encodingFormat": "application/nwb"},
                },
            )
        else:
            raise Exception("At least id_ or name should be informed.")

        if not resource:
            raise Exception(f"No matching resource for {id_} {name}")

        return self.access_point.resource_location(resource)[0]

    def get_mechanisms_directory(self):
        """Return the path to the directory containing the mechanisms for the current emodel"""

        directory_name = self.emodel_metadata.as_string()

        mechanisms_directory = pathlib.Path("./nexus_temp/") / directory_name / "mechanisms"

        return mechanisms_directory.resolve()

    def load_channel_gene_expression(self, name):
        """Retrieve a channel gene expression resource and read its content"""

        dataset = self.access_point.fetch_one(filters={"type": "RNASequencing", "name": name})

        filepath = self.access_point.resource_location(dataset)[0]

        df = pandas.read_csv(filepath, index_col=["me-type", "t-type", "modality"])

        return df, filepath

    def load_ic_map(self):
        """Get the ion channel/genes map from Nexus"""

        resource_ic_map = self.access_point.fetch_one(
            {"type": "IonChannelMapping", "name": "icmapping"}
        )

        return self.access_point.download(resource_ic_map.id)

    def get_t_types(self, table_name):
        """Get the list of t-types available for the present emodel"""

        df, _ = self.load_channel_gene_expression(table_name)
        return df.loc[self.emodel].index.get_level_values("t-type").unique().tolist()

    def get_available_morphologies(self):
        """Get the list of names of the available morphologies"""

        resources = self.access_point.fetch({"type": "NeuronMorphology"})

        return {r.name for r in resources}

    def get_available_mechanisms(self):
        """Get all the available mechanisms"""

        resources = self.access_point.fetch({"type": "SubCellularModelScript"})

        available_mechanisms = []
        for r in resources:

            version = r.modelid if hasattr(r, "modelid") else None
            stochastic = r.stochastic if hasattr(r, "stochastic") else None

            parameters = {}
            if hasattr(r, "exposesParameters"):

                exposes_parameters = r.exposesParameters
                if not isinstance(exposes_parameters, list):
                    exposes_parameters = [exposes_parameters]

                for ep in exposes_parameters:
                    if ep.type == "NmodlRangeVariable":
                        lower_limit = ep.lowerLimit if hasattr(ep, "lowerLimit") else None
                        upper_limit = ep.upperLimit if hasattr(ep, "upperLimit") else None
                        parameters[f"{ep.name}_{r.mod.suffix}"] = [lower_limit, upper_limit]

            ion_currents = []
            if hasattr(r.mod, "write"):
                ions_ = r.mod.write
                if isinstance(ions_, str):
                    if ions_[0] == "i":
                        ion_currents = [ions_]
                elif isinstance(ions_, list):
                    ion_currents = [ion for ion in ions_ if ion[0] == "i"]

            mech = MechanismConfiguration(
                r.name,
                location=None,
                stochastic=stochastic,
                version=version,
                parameters=parameters,
                ion_currents=ion_currents,
            )

            available_mechanisms.append(mech)

        return available_mechanisms

    def get_available_traces(self, filter_species=False, filter_brain_region=False):
        """Get the list of available Traces for the current species from Nexus"""

        filters = {"type": "Trace", "distribution": {"encodingFormat": "application/nwb"}}

        if filter_species:
            filters["subject"] = self.emodel_metadata_ontology.species
        if filter_brain_region:
            filters["brainLocation"] = self.emodel_metadata_ontology.brain_region

        resource_traces = self.access_point.fetch(filters)

        traces = []

        if resource_traces is None:
            return traces

        for r in resource_traces:

            ecodes = None
            if hasattr(r, "image"):
                ecodes = []
                for stimulus in r.image:
                    ecode = stimulus.stimulusType.id.split("/")[-1]
                    ecodes.append(ecode)

            species = None
            if hasattr(r, "subject") and hasattr(r.subject, "species"):
                species = r.subject.species

            brain_region = None
            if hasattr(r, "brainLocation"):
                brain_region = r.brainLocation

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
                    etype=None,  # Todo: update when etype will be available
                )
            )

        return traces
