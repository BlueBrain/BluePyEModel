"""Access point using Nexus Forge"""

import json
import logging
import os
import pathlib
from collections import OrderedDict

import numpy
import pandas
from kgforge.specializations.resources import Dataset

from bluepyemodel.access_point.access_point import DataAccessPoint
from bluepyemodel.access_point.forge_access_point import NexusForgeAccessPoint
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings

# pylint: disable=simplifiable-if-expression,too-many-arguments

logger = logging.getLogger("__main__")


BPEM_NEXUS_SCHEMA = [
    "ElectrophysiologyFeatureOptimisationNeuronMorphology",
    "ElectrophysiologyFeatureExtractionTrace",
    "ElectrophysiologyFeatureExtractionTarget",
    "ElectrophysiologyFeatureOptimisationTarget",
    "ElectrophysiologyFeatureValidationTarget",
    "ElectrophysiologyFeatureOptimisationParameter",
    "ElectrophysiologyFeatureOptimisationChannelDistribution",
    "SubCellularModel",
    "ElectrophysiologyFeature",
    "ElectrophysiologyFeatureExtractionProtocol",
    "EModel",
    "EModelPipelineSettings"
]


class NexusAccessPointException(Exception):
    """For Exceptions related to the NexusAccessPoint"""


def yesno(question):
    """Ask a Yes/No question"""

    prompt = f"{question} ? (y/n): "
    ans = input(prompt).strip().lower()

    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yesno(question)

    if ans == "y":
        return True

    return False


def format_dict_for_resource(d):
    """Translates a dictionary to a list of the format used by resources"""

    out = []

    if d is None:
        return out

    for k, v in d.items():

        if numpy.isnan(v):
            v = None

        out.append({"name": k, "value": v, "unitCode": ""})

    return out


class NexusAccessPoint(DataAccessPoint):
    """API to retrieve, push and format data from and to the Knowledge Graph"""

    def __init__(
        self,
        emodel,
        species,
        brain_region=None,
        project="emodel_pipeline",
        organisation="demo",
        endpoint="https://bbp.epfl.ch/nexus/v1",
        forge_path=None,
        ttype=None,
        iteration_tag=None,
    ):
        """Init

        Args:
            emodel (str): name of the emodel
            species (str): name of the species.
            brain_region (str): name of the brain location (e.g: "CA1").
            project (str): name of the Nexus project.
            organisation (str): name of the Nexus organization to which the project belong.
            endpoint (str): Nexus endpoint.
            forge_path (str): path to a .yml used as configuration by nexus-forge.
            ttype (str): name of the t-type. Required if using the gene expression or IC selector.
            iteration_tag (str): tag associated to the current run. Used to tag the
                Resources generated during the different run.
        """

        super().__init__(emodel)

        self.species = species
        self.brain_region = self.get_brain_region(brain_region)
        self.ttype = ttype

        self.access_point = NexusForgeAccessPoint(
            project=project,
            organisation=organisation,
            endpoint=endpoint,
            forge_path=forge_path,
            iteration_tag=iteration_tag,
        )

        self.pipeline_settings = self.load_pipeline_settings(strict=False)

    def get_subject(self, for_search=False):
        """Get the ontology of a species based n the species name. The id is not used
        during search as if it is specified the search fail (will be fixed soon)."""

        if self.species == "human":
            subject = {"type": "Subject", "species": {"label": "Homo sapiens"}}
            if not for_search:
                subject["species"]["id"] = "http://purl.obolibrary.org/obo/NCBITaxon_9606"

        elif self.species == "rat":
            subject = {"type": "Subject", "species": {"label": "Musca domestica"}}
            if not for_search:
                subject["species"]["id"] = "http://purl.obolibrary.org /obo/NCBITaxon_7370"

        elif self.species == "mouse":
            subject = {"type": "Subject", "species": {"label": "Mus musculus"}}
            if not for_search:
                subject["species"]["id"] = "http://purl.obolibrary.org/obo/NCBITaxon_10090"

        else:
            raise NexusAccessPointException("Unknown species %s." % self.species)

        return subject

    def get_brain_region(self, brain_region):
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

    def fetch_emodel(self, seed=None, githash=None, use_version=True):
        """Fetch an emodel"""

        filters = {
            "type": "EModel",
            "eModel": self.emodel,
            "subject": self.get_subject(for_search=True),
            "brainLocation": self.brain_region,
        }

        if seed:
            filters["seed"] = seed

        if githash:
            filters["githash"] = githash

        resources = self.access_point.fetch(filters, use_version=use_version)

        return resources

    def deprecate_project(self, use_version=True):
        """Deprecate all resources used or produced by BluePyModel. Use with extreme caution."""

        if not yesno("Confirm deprecation of all BluePyEmodel resources in Nexus project"):
            return

        for type_ in BPEM_NEXUS_SCHEMA:
            filters = {"type": type_}
            self.access_point.deprecate(filters, use_version=use_version)

    def load_pipeline_settings(self, strict=True):
        """ """

        settings = {}

        resource = self.access_point.fetch_one(
            filters={
                "type": "EModelPipelineSettings",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            },
            strict=strict,
        )

        if not resource:
            return settings

        resource_dict = self.access_point.forge.as_json(resource)

        # Removes the unwanted entries of the Resource such as the metadata
        for setting in [
            "extraction_threshold_value_save",
            "plot_extraction",
            "efel_settings",
            "stochasticity",
            "morph_modifiers",
            "optimizer",
            "optimisation_params",
            "optimisation_timeout",
            "threshold_efeature_std",
            "max_ngen",
            "validation_threshold",
            "validation_function",
            "plot_optimisation",
            "n_model",
            "optimisation_batch_size",
            "max_n_batch",
            "path_extract_config",
            "name_Rin_protocol",
            "name_rmp_protocol",
            "validation_protocols",
            "additional_protocols",
            "compile_mechanisms",
        ]:
            if setting in resource_dict:
                settings[setting] = resource_dict[setting]

        return EModelPipelineSettings(**settings)

    def store_channel_gene_expression(
        self, table_path, name="Mouse_met_types_ion_channel_expression"
    ):
        """Create a channel gene expression resource containing the gene expression table.

        Args:
            table_path (str): path to the gene expression table
        """

        if pathlib.Path(table_path).suffix != ".csv":
            raise NexusAccessPointException("Was expecting a .csv file")

        dataset = Dataset(
            self.access_point.forge,
            type=["Entity", "Dataset", "RNASequencing"],
            name=name,
            brainLocation=self.brain_region,
            description="Output from IC_selector module",
        )
        dataset.add_distribution(table_path)

        self.access_point.forge.register(dataset)

    def load_channel_gene_expression(self, name):
        """Retrieve a channel gene expression resource and read its content"""

        dataset = self.access_point.fetch_one(
            filters={"type": "RNASequencing", "name": name}, use_version=False
        )

        filepath = self.access_point.resource_location(dataset)[0]

        df = pandas.read_csv(filepath, index_col=["me-type", "t-type", "modality"])

        return df, filepath

    def get_t_types(self, table_name):
        """Get the list of t-types available for the present emodel"""

        df, _ = self.load_channel_gene_expression(table_name)
        return df.loc[self.emodel].index.get_level_values("t-type").unique().tolist()

    def get_channel_gene_expression(self, table_name):
        """Get the channel gene expression and gene distribution for a given emodel and t-type"""

        df, _ = self.load_channel_gene_expression(table_name)

        # TODO: improve for soma, axon, dendrites (data[1], data[2], data[3]),
        # right now it uses only the somatic distribution
        data = df.loc[self.emodel, self.ttype].to_dict(orient="records")
        presence = data[0]
        distrib = data[1]

        gene_expression = {}
        for gene in presence:
            if int(presence[gene]):
                gene_expression[gene] = distrib[gene]

        return gene_expression

    def get_mechanism_from_gene(self, path_mapping, gene, channel_version=None):
        """Returns Nexus resource for a SubCellularModel file corresponding to a given
        gene name.

        Args:
            path_mapping (str): path to the gene expression table
            gene (str): name of the gene or protein for which to retrieve the channel name
            version (str): version number of the mod files, if None, returns the
                highest version number
        """
        with open(path_mapping, "r") as mapping_file:
            mapping = json.load(mapping_file)

        lower_gene = gene.lower()

        if lower_gene in mapping["genes"]:
            mechanism_name = mapping["genes"][lower_gene]["protein"]

            filters = {"type": "SubCellularModel", "name": mechanism_name}

            search_version = channel_version
            if search_version is None:
                if "versions" in mapping["genes"][lower_gene] and len(
                    mapping["genes"][lower_gene]["versions"]
                ):
                    search_version = sorted(mapping["genes"][lower_gene]["versions"])[-1]

            if search_version:
                filters["version"] = search_version

            return self.access_point.fetch_one(filters, use_version=False)

        return None

    def store_morphology(
        self,
        name=None,
        id_=None,
        seclist_names=None,
        secarray_names=None,
        section_index=None,
    ):
        """Creates an ElectrophysiologyFeatureOptimisationNeuronMorphology resource based on a
        NeuronMorphology.

        Args:
            name (str): name of the morphology.
            id_ (str): nexus id of the NeuronMorphology.
            seclist_names (list): Names of the lists of sections (ex: 'somatic')
            secarray_names (list): names of the sections (ex: 'soma')
            section_index (int): index to a specific section, used for non-somatic recordings.
        """

        if id_:
            resource = self.access_point.retrieve(id_)
        elif name:
            resource = self.access_point.fetch_one(
                {"type": "NeuronMorphology", "name": name}, use_version=False
            )
        else:
            raise NexusAccessPointException("At least id_ or name should be informed.")

        if not resource:
            raise NexusAccessPointException("No matching resource for morphology %s %s" % id_, name)

        if not name:
            name = resource.name

        self.access_point.register(
            {
                "type": [
                    "Entity",
                    "ElectrophysiologyFeatureOptimisationNeuronMorphology",
                ],
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "brainLocation": self.brain_region,
                "name": name,
                "morphology": {"id": resource.id},
                "sectionListNames": seclist_names,
                "sectionArrayNames": secarray_names,
                "sectionIndex": section_index,
            },
            {
                "type": "ElectrophysiologyFeatureOptimisationNeuronMorphology",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
                "name": name,
            },
        )

    def store_trace_metadata(
        self,
        name=None,
        id_=None,
        ecode=None,
        recording_metadata=None,
    ):
        """Creates an ElectrophysiologyFeatureExtractionTrace resource based on an ephys file.

        Args:
            id_ (str): Nexus id of the trace file.
            name (str): name of the trace file.
            ecode (str): name of the eCode of interest.
            recording_metadata (dict): metadata such as ton, toff, v_unit associated to the traces
                of ecode in this file.
        """

        if recording_metadata is None:
            recording_metadata = {}

        if "protocol_name" not in recording_metadata and ecode:
            recording_metadata["protocol_name"] = ecode

        if id_:
            resource = self.access_point.retrieve(id_)
        elif name:
            resource = self.access_point.fetch_one(
                {
                    "type": "Trace",
                    "name": name,
                    "distribution": {"encodingFormat": "application/nwb"},
                },
                use_version=False,
            )
        else:
            raise NexusAccessPointException("At least id_ or name should be informed.")

        if not resource:
            raise NexusAccessPointException("No matching resource for %s %s" % id_, name)

        self.access_point.register(
            {
                "type": ["Entity", "ElectrophysiologyFeatureExtractionTrace"],
                "eModel": self.emodel,
                "name": f"{resource.name}_{ecode}",
                "subject": self.get_subject(for_search=False),
                "brainLocation": self.brain_region,
                "trace": {"id": resource.id},
                "cell": {"name": resource.name},
                "ecode": ecode,
                "recording_metadata": recording_metadata,
            },
            {
                "type": "ElectrophysiologyFeatureExtractionTrace",
                "eModel": self.emodel,
                "name": f"{resource.name}_{ecode}",
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
                "trace": {"id": resource.id},
                "cell": {"name": resource.name},
                "ecode": ecode,
            },
        )

    def store_extraction_target(
        self, ecode, target_amplitudes, tolerances, use_for_rheobase, efeatures, efel_settings
    ):
        """Creates an ElectrophysiologyFeatureExtractionTarget resource used as target for the
        e-features extraction process.

        Args:
            ecode (str): name of the eCode of interest.
            target_amplitudes (list): amplitude of the step of the protocol. Expressed as a
                percentage of the threshold amplitude (rheobase).
            tolerances (list): tolerance around the target amplitude in which an
                experimental trace will be seen as a hit during efeatures extraction.
            use_for_rheobase (bool): should the ecode be used to compute the rheobase
                of the cells during extraction.
            efeatures (list): list of efeature names to extract for this ecode.
            efel_settings (dict): eFEL settings.
        """

        features = [{"name": f} for f in efeatures]

        self.access_point.register(
            {
                "type": ["Entity", "ElectrophysiologyFeatureExtractionTarget"],
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "brainLocation": self.brain_region,
                "stimulus": {
                    "stimulusType": {"label": ecode},
                    "stimulusTarget": target_amplitudes,
                    "tolerance": tolerances,
                    "threshold": use_for_rheobase,
                    "recordingLocation": "soma",
                },
                "feature": features,
                "efel_settings": efel_settings,
            },
            {
                "type": "ElectrophysiologyFeatureExtractionTarget",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
                "stimulus": {
                    "stimulusType": {"label": ecode},
                    "stimulusTarget": target_amplitudes[0],
                },
            },
        )

    def store_opt_validation_target(
        self,
        type_,
        ecode,
        protocol_type,
        target_amplitude,
        efeatures,
        extra_recordings,
        efel_settings,
    ):
        """Creates resources used as target optimisation and validation.

        Args:
            type_ (str): type of the Nexus Entity.
            ecode (str): name of the eCode of the protocol.
            protocol_type (str): type of the protocol ("StepProtocol" or "StepThresholdProtocol").
            target_amplitude (float): amplitude of the step of the protocol. Expressed as a
                percentage of the threshold amplitude (rheobase).
            efeatures (list): list of efeatures name used as targets for this protocol.
            extra_recordings (list): definition of additional recordings used for this protocol.
            efel_settings (dict): eFEL settings.
        """

        if protocol_type not in [
            "StepProtocol",
            "StepThresholdProtocol",
            "RinProtocol",
            "RMPProtocol",
        ]:
            raise NexusAccessPointException("protocol_type %s unknown." % protocol_type)

        features = []
        for f in efeatures:
            features.append(
                {
                    "name": f,
                    "onsetTime": {"unitCode": "ms", "value": None},
                    "offsetTime": {"unitCode": "ms", "value": None},
                }
            )

            if efel_settings and "stim_start" in efel_settings:
                features[-1]["onsetTime"]["value"] = efel_settings["stim_start"]
            if efel_settings and "stim_end" in efel_settings:
                features[-1]["offsetTime"]["value"] = efel_settings["stim_end"]

        self.access_point.register(
            {
                "type": ["Entity", "Target", type_],
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "brainLocation": self.brain_region,
                "protocolType": protocol_type,
                "stimulus": {
                    "stimulusType": {"label": ecode},
                    "target": target_amplitude,
                    "recordingLocation": "soma",
                },
                "feature": features,
                "extraRecordings": extra_recordings,
            },
            {
                "type": type_,
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
                "protocolType": protocol_type,
                "stimulus": {
                    "stimulusType": {"label": ecode},
                    "target": target_amplitude,
                    "recordingLocation": "soma",
                },
            },
        )

    def store_emodel_targets(
        self,
        ecode,
        efeatures,
        amplitude,
        extraction_tolerance,
        protocol_type,
        used_for_extraction_rheobase=False,
        used_for_optimization=False,
        used_for_validation=False,
        extra_recordings=None,
        efel_settings=None,
    ):
        """Register the efeatures and their associated protocols that will be used as target during
        efeatures extraction, optimisation and validation.

        Args:
        ecode (str): name of the eCode of the protocol.
        efeatures (list): list of efeatures name used as targets for this ecode.
        amplitude (float): amplitude of the step of the protocol. Expressed as a percentage of
            the threshold amplitude (rheobase).
        extraction_tolerance (list): tolerance around the target amplitude in which an
            experimental trace will be seen as a hit during efeatures extraction.
        protocol_type (str): type of the protocol ("StepProtocol" or "StepThresholdProtocol",
            "RinProtocol", "RMPProtocol"). If using StepThresholdProtocols, it is mandatory to
            have another target as RMPProtocol and another target as RinProtocol.
        used_for_extraction_rheobase (bool): should the ecode be used to compute the rheobase
            of the cells during extraction.
        used_for_optimization (bool): should the ecode be used as a target during optimisation.
        used_for_validation (bool): should the ecode be used as a target during validation.
            Both used_for_optimization and used_for_validation cannot be True.
        extra_recordings (list): definitions of additional recordings to use for this protocol.
        efel_settings (dict): eFEL settings.
        """

        if efel_settings is None:
            efel_settings = {}

        if extra_recordings is None:
            extra_recordings = []

        if used_for_optimization and used_for_validation:
            raise NexusAccessPointException(
                "Both used_for_optimization and used_for_validation cannot be True for the"
                " same Ecode."
            )

        self.store_extraction_target(
            ecode=ecode,
            target_amplitudes=[amplitude],
            tolerances=[extraction_tolerance],
            use_for_rheobase=used_for_extraction_rheobase,
            efeatures=efeatures,
            efel_settings=efel_settings,
        )

        if used_for_optimization:
            self.store_opt_validation_target(
                "ElectrophysiologyFeatureOptimisationTarget",
                ecode=ecode,
                protocol_type=protocol_type,
                target_amplitude=amplitude,
                efeatures=efeatures,
                extra_recordings=extra_recordings,
                efel_settings=efel_settings,
            )

        elif used_for_validation:
            self.store_opt_validation_target(
                "ElectrophysiologyFeatureValidationTarget",
                ecode=ecode,
                protocol_type=protocol_type,
                target_amplitude=amplitude,
                efeatures=efeatures,
                extra_recordings=extra_recordings,
                efel_settings=efel_settings,
            )

    def store_optimisation_parameter(
        self,
        name,
        value,
        location,
        mechanism_name=None,
        distribution="constant",
        auto_handle_mechanism=False,
    ):
        """Creates an ElectrophysiologyFeatureOptimisationParameter resource specifying a
        parameter of the model.

        Args:
            name (str): name of the parameter.
            value (list or float): value of the parameter. If value is a float, the parameter will
                be fixed. If value is a list of two elements, the first will be used as a lower
                bound and the second as an upper bound during the optimization.
            location (str): locations at which the parameter is present. The element of location
                have to be section list names.
            mechanism_name (str): name of the mechanism associated to the parameter.
            distribution (str): distribution of the parameters along the sections.
            auto_handle_mechanism (bool): if True, looks for the matching SubCellularModel or create
                it if it does not exist.
        """

        if isinstance(value, (list, tuple)):
            min_value, max_value = value
        else:
            min_value, max_value = value, value

        if auto_handle_mechanism and mechanism_name and mechanism_name != "pas":
            self.store_mechanism(name=mechanism_name)

        resource_description = {
            "name": name,
            "type": ["Entity", "Parameter", "ElectrophysiologyFeatureOptimisationParameter"],
            "eModel": self.emodel,
            "subject": self.get_subject(for_search=False),
            "brainLocation": self.brain_region,
            "parameter": {
                "name": name,
                "minValue": min_value,
                "maxValue": max_value,
                "unitCode": "",
            },
            "location": location,
            "channelDistribution": distribution,
        }

        if mechanism_name:
            resource_description["subCellularMechanism"] = mechanism_name

        search_filters = resource_description.copy()
        search_filters["subject"] = self.get_subject(for_search=True)
        search_filters["type"] = "ElectrophysiologyFeatureOptimisationParameter"

        self.access_point.register(resource_description, search_filters)

    def store_channel_distribution(
        self,
        name,
        function,
        parameters,
        soma_reference_location=0.5,
    ):
        """Creates an ElectrophysiologyFeatureOptimisationChannelDistribution defining a channel
        distribution.

        Args:
            name (str): name of the distribution.
            function (str): (only knows the python math library).
            parameters (list): names of the parameters used by the distribution function.
            soma_reference_location (float): The location (comp_x) along the soma used as a
                starting point when computing distances.
        """

        if soma_reference_location < 0.0 or soma_reference_location > 1.0:
            raise NexusAccessPointException("soma_reference_location should be between 0. and 1.")

        self.access_point.register(
            {
                "type": ["Entity", "ElectrophysiologyFeatureOptimisationChannelDistribution"],
                "channelDistribution": name,
                "name": name,
                "function": function,
                "parameter": parameters,
                "somaReferenceLocation": soma_reference_location,
            },
            {
                "type": "ElectrophysiologyFeatureOptimisationChannelDistribution",
                "channelDistribution": name,
            },
        )

    def store_mechanism(self, name=None, id_=None, stochastic=None):
        """Creates an SubCellularModel based on a SubCellularModelScript.

        Args:
            name (str): name of the mechanism.
            id_ (str): Nexus id of the mechanism.
            stochastic (bool): is the mechanism stochastic.
        """

        if id_:
            resource = self.access_point.retrieve(id_)
        elif name:
            resources = self.access_point.fetch(
                {"type": "SubCellularModelScript", "name": name}, use_version=False
            )
            # Genetic channel can have several versions, we want the most recent one:
            if len(resources) > 1 and all(hasattr(r, "version") for r in resources):
                resource = sorted(resources, key=lambda x: x.version)[-1]
            else:
                resource = resources[0]
#             resource = self.access_point.fetch_one(
#                 {"type": "SubCellularModelScript", "name": name}, use_version=False
#             )
        else:
            raise NexusAccessPointException("At least name or id_ should be informed.")

        if not resource:
            raise NexusAccessPointException("No matching resource for mechanism %s %s" % id_, name)

        if not name:
            name = resource.name

        if stochastic is None:
            stochastic = True if "Stoch" in name else False

        self.access_point.register(
            {
                "type": ["Entity", "SubCellularModel"],
                "name": name,
                "subCellularMechanism": name,
                "modelScript": {"id": resource.id, "type": "SubCellularModelScript"},
                "stochastic": stochastic,
            },
            {
                "type": "SubCellularModel",
                "name": name,
                "subCellularMechanism": name,
            },
        )

    def store_pipeline_settings(
        self,
        extraction_threshold_value_save=1,
        efel_settings=None,
        stochasticity=False,
        morph_modifiers=None,
        optimizer="IBEA",
        optimisation_params=None,
        optimisation_timeout=600.0,
        threshold_efeature_std=0.05,
        max_ngen=100,
        validation_threshold=5.0,
        optimization_batch_size=5,
        max_n_batch=3,
        n_model=3,
        plot_extraction=True,
        plot_optimisation=True,
        additional_protocols=None,
        compile_mechanisms=False,
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
            additional_protocols(dict):
            compile_mechanisms (bool):
        """

        if efel_settings is None:
            efel_settings = {"interp_step": 0.025, "strict_stiminterval": True}

        if optimisation_params is None:
            optimisation_params = {"offspring_size": 100}

        resource_description = {
            "name": f"Pipeline settings {self.emodel}",
            "type": ["Entity", "Parameter", "EModelPipelineSettings"],
            "eModel": self.emodel,
            "subject": self.get_subject(for_search=False),
            "brainLocation": self.brain_region,
            "extraction_threshold_value_save": extraction_threshold_value_save,
            "efel_settings": efel_settings,
            "stochasticity": stochasticity,
            "optimizer": optimizer,
            "optimisation_params": optimisation_params,
            "optimisation_timeout": optimisation_timeout,
            "max_ngen": max_ngen,
            "validation_threshold": validation_threshold,
            "optimisation_batch_size": optimization_batch_size,
            "max_n_batch": max_n_batch,
            "n_model": n_model,
            "plot_extraction": plot_extraction,
            "plot_optimisation": plot_optimisation,
            "threshold_efeature_std": threshold_efeature_std,
            "morph_modifiers": morph_modifiers,
            "additional_protocols": additional_protocols,
            "compile_mechanisms": compile_mechanisms,
        }

        resource_search = {
            "name": f"Pipeline settings {self.emodel}",
            "type": "EModelPipelineSettings",
            "eModel": self.emodel,
            "subject": self.get_subject(for_search=True),
            "brainLocation": self.brain_region,
        }

        self.access_point.register(resource_description, resource_search)

    def _build_extraction_targets(self, resources_target):
        """Create a dictionary defining the target of the feature extraction process"""

        targets = []
        protocols_threshold = []

        for resource in resources_target:

            ecode = resource.stimulus.stimulusType.label

            if isinstance(resource.stimulus.tolerance, (int, float)):
                tolerances = [resource.stimulus.tolerance]
            else:
                tolerances = resource.stimulus.tolerance

            if isinstance(resource.stimulus.stimulusTarget, (int, float)):
                amplitudes = [resource.stimulus.stimulusTarget]
            else:
                amplitudes = resource.stimulus.stimulusTarget

            efel_settings = {}
            if hasattr(resource, "efel_settings"):
                efel_settings = self.access_point.forge.as_json(resource.efel_settings)

            efeatures = self.access_point.forge.as_json(resource.feature)
            if not isinstance(efeatures, list):
                efeatures = [efeatures]

            for amp, tol in zip(amplitudes, tolerances):
                for f in efeatures:
                    targets.append(
                        {
                            "efeature": f["name"],
                            "protocol": ecode,
                            "amplitude": amp,
                            "tolerance": tol,
                            "efel_settings": efel_settings,
                        }
                    )

            if hasattr(resource.stimulus, "threshold") and resource.stimulus.threshold:
                protocols_threshold.append(ecode)

        return targets, set(protocols_threshold)

    def _build_extraction_metadata(self, targets):
        """
        Create a dictionary that informs which files should be used for which
        target. It also specifies the metadata (such as units or ljp) associated
        to the files.

        This function also download the files in ./nexus_tmp, if they are not
        already present.
        """

        traces_metadata = {}

        for protocol in list(set(t["protocol"] for t in targets)):

            resources_ephys = self.access_point.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureExtractionTrace",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=True),
                    "brainLocation": self.brain_region,
                    "ecode": protocol,
                },
            )

            if resources_ephys is None:
                raise NexusAccessPointException(
                    "Could not get ephys files for ecode %s, emodel %s" % (protocol, self.emodel)
                )

            for resource in resources_ephys:

                recording_metadata = self.access_point.forge.as_json(resource.recording_metadata)

                resource_trace = self.access_point.retrieve(resource.trace.id)
                recording_metadata["filepath"] = self.access_point.resource_location(
                    resource_trace
                )[0]

                cell_name = resource.cell.name
                ecode = resource.ecode

                if cell_name not in traces_metadata:
                    traces_metadata[cell_name] = {}

                if ecode not in traces_metadata[cell_name]:
                    traces_metadata[cell_name][ecode] = []

                traces_metadata[cell_name][ecode].append(recording_metadata)

        return traces_metadata

    def get_extraction_metadata(self):
        """Gather the metadata used to build the config dictionaries given as an
        input to BluePyEfe.

        Returns:
            traces_metadata (dict)
            targets (dict)
            protocols_threshold (list)
        """

        traces_metadata = {}
        targets = {}
        protocols_threshold = []

        resources_extraction_target = self.access_point.fetch(
            filters={
                "type": "ElectrophysiologyFeatureExtractionTarget",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            },
        )

        if resources_extraction_target is None:
            logger.warning(
                "NexusForge warning: could not get extraction metadata for emodel %s", self.emodel
            )
            return traces_metadata, targets, protocols_threshold

        targets, protocols_threshold = self._build_extraction_targets(resources_extraction_target)
        if not protocols_threshold:
            raise NexusAccessPointException(
                "No eCode have been informed to compute the rheobase during extraction."
            )

        traces_metadata = self._build_extraction_metadata(targets)

        return traces_metadata, targets, protocols_threshold

    def register_efeature(self, name, val, protocol_name=None, protocol_amplitude=None):
        """Register an ElectrophysiologyFeature resource"""

        resource_description = {
            "type": ["Entity", "ElectrophysiologyFeature"],
            "eModel": self.emodel,
            "name": name,
            "subject": self.get_subject(for_search=False),
            "brainLocation": self.brain_region,
            "feature": {
                "name": name,
                "value": [],
                "series": [
                    {
                        "statistic": "mean",
                        "unitCode": "dimensionless",
                        "value": val[0],
                    },
                    {
                        "statistic": "standard deviation",
                        "unitCode": "dimensionless",
                        "value": val[1],
                    },
                ],
            },
        }

        pdf_amp, pdf_amp_rel = self.search_figure_efeatures(protocol_name, name)
        pdfs = {}
        if pdf_amp:
            pdfs["amp"] = self.access_point.forge.attach(pdf_amp)
        if pdf_amp_rel:
            pdfs["amp_rel"] = self.access_point.forge.attach(pdf_amp_rel)
        if pdfs:
            resource_description["pdfs"] = pdfs

        if protocol_name and protocol_amplitude:

            resource_description["stimulus"] = {
                "stimulusType": {
                    "id": "http://bbp.epfl.ch/neurosciencegraph/ontologies"
                    "/stimulustypes/{}".format(protocol_name),
                    "label": protocol_name,
                },
                "stimulusTarget": float(protocol_amplitude),
                "recordingLocation": "soma",
            }

            resource_description["name"] = f"{name}_{protocol_name}_{protocol_amplitude}"

        else:
            resource_description["stimulus"] = {
                "stimulusType": {"id": "", "label": "global"},
                "recordingLocation": "soma",
            }

        self.access_point.register(resource_description)

    def store_efeatures(
        self,
        efeatures,
        current,
    ):
        """Store the efeatures and currents obtained from BluePyEfe in ElectrophysiologyFeature
        resources.

        Args:
            efeatures (dict): of the format:
                {
                    'protocol_name':[
                        {'feature': feature_name, value: [mean, std]},
                        {'feature': feature_name2, value: [mean, std]}
                    ]
                }
            current (dict): of the format:
                {
                    "hypamp": [mean, std],
                    "thresh": [mean, std]
                }
            name_Rin_protocol (str): not used.
            name_rmp_protocol (str): not used.
            validation_protocols (list): not used.
        """

        # TODO: add dependencies on Files

        for protocol in efeatures:

            for feature in efeatures[protocol]["soma"]:

                protocol_name = "_".join(protocol.split("_")[:-1])
                prot_amplitude = protocol.split("_")[-1]

                self.register_efeature(
                    name=feature["feature"],
                    val=feature["val"],
                    protocol_name=protocol_name,
                    protocol_amplitude=prot_amplitude,
                )

        for cur in ["holding_current", "threshold_current"]:
            self.register_efeature(name=cur, val=current[cur])

    def store_protocols(self, stimuli):
        """Store the protocols obtained from BluePyEfe in
            ElectrophysiologyFeatureExtractionProtocol resources.

        Args:
            stimuli (dict): of the format:
                {
                    'protocol_name':
                        {"step": ..., "holding": ...}
                }
            validation_protocols (list): not used by API NexusForge
        """

        # TODO: How to get the ontology for the stimulus ? is the url string
        # always of the same format ?

        # TODO: add dependencies on Files

        for protocol_name, protocol in stimuli.items():

            prot_name = "_".join(protocol_name.split("_")[:-1])
            prot_amplitude = protocol_name.split("_")[-1]

            self.access_point.register(
                {
                    "type": ["Entity", "ElectrophysiologyFeatureExtractionProtocol"],
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=False),
                    "brainLocation": self.brain_region,
                    "stimulus": {
                        "stimulusType": {
                            "id": "http://bbp.epfl.ch/neurosciencegraph/ontologies/stimulustypes"
                            "/{}".format(prot_name),
                            "label": prot_name,
                        },
                        "stimulusTarget": float(prot_amplitude),
                    },
                    "path": "",
                    "protocolDefinition": {
                        "step": {
                            "delay": protocol["step"]["delay"],
                            "amplitude": protocol["step"]["amp"],
                            "thresholdPercentage": protocol["step"]["thresh_perc"],
                            "duration": protocol["step"]["duration"],
                            "totalDuration": protocol["step"]["totduration"],
                        },
                        "holding": {
                            "delay": protocol["holding"]["delay"],
                            "amplitude": protocol["holding"]["amp"],
                            "duration": protocol["holding"]["duration"],
                            "totalDuration": protocol["holding"]["totduration"],
                        },
                    },
                },
            )

    def store_emodel(
        self,
        scores,
        params,
        optimizer_name,
        seed,
        githash="",
        validated=None,
        scores_validation=None,
        features=None,
    ):
        """Store an emodel obtained from BluePyOpt in an EModel resource.

        Args:
            scores (dict): scores of the efeatures. Of the format {"objective_name": score}.
            params (dict): values of the parameters. Of the format {"param_name": param_value}.
            optimizer_name (str): name of the optimizer (IBEA, CMA, ...).
            seed (int): seed used by the optimizer.
            githash (string): githash associated with the configuration files.
            validated (bool): None indicate that the model did not go through validation.\
                False indicates that it failed validation. True indicates that it
                passed validation.
            scores_validation (dict): scores of the validation efeatures. Of the format
                {"objective_name": score}.
            features (dict): values of the efeatures. Of the format {"objective_name": value}.
        """

        scores_validation_resource = format_dict_for_resource(scores_validation)
        scores_resource = format_dict_for_resource(scores)
        features_resource = format_dict_for_resource(features)
        parameters_resource = format_dict_for_resource(params)

        pdf_dependencies = self._build_pdf_dependencies(seed, githash)

        pip_freeze = os.popen("pip freeze").read()

        resource_description = {
            "type": ["Entity", "EModel"],
            "eModel": self.emodel,
            "subject": self.get_subject(for_search=False),
            "brainLocation": self.brain_region,
            "name": "{} {}".format(self.emodel, seed),
            "fitness": sum(list(scores.values())),
            "parameter": parameters_resource,
            "score": scores_resource,
            "features": features_resource,
            "scoreValidation": scores_validation_resource,
            "passedValidation": validated,
            "optimizer": str(optimizer_name),
            "seed": seed,
            "pip_freeze": pip_freeze,
            "pdfs": pdf_dependencies,
        }

        search = {
            "type": "EModel",
            "subject": self.get_subject(for_search=True),
            "brainLocation": self.brain_region,
            "seed": seed,
        }

        if githash:
            resource_description["githash"] = githash
            search["githash"] = githash

        if self.ttype:
            resource_description["ttype"] = self.ttype

        self.access_point.register(resource_description, search, replace=True)

    def _build_pdf_dependencies(self, seed, githash):
        """Find all the pdfs associated to an emodel"""

        pdfs = {}

        opt_pdf = self.search_figure_emodel_optimisation(seed, githash)
        if opt_pdf:
            pdfs["optimisation"] = self.access_point.forge.attach(opt_pdf)

        traces_pdf = self.search_figure_emodel_traces(seed, githash)
        if traces_pdf:
            pdfs["traces"] = self.access_point.forge.attach(traces_pdf)

        scores_pdf = self.search_figure_emodel_score(seed, githash)
        if scores_pdf:
            pdfs["scores"] = self.access_point.forge.attach(scores_pdf)

        parameters_pdf = self.search_figure_emodel_parameters()
        if parameters_pdf:
            pdfs["parameters"] = self.access_point.forge.attach(parameters_pdf)

        return pdfs

    def get_emodels(self, emodels):
        """Get the list of emodels.

        Returns:
            models (list): return the emodels, of the format:
            [
                {
                    "emodel": ,
                    "species": ,
                    "brain_region": ,
                    "fitness": ,
                    "parameters": ,
                    "scores": ,
                    "validated": ,
                    "optimizer": ,
                    "seed": ,
                }
            ]
        """

        models = []

        resources = self.fetch_emodel()

        if resources is None:
            logger.warning("NexusForge warning: could not get emodels for emodel %s", self.emodel)
            return models

        for resource in resources:

            params = {
                p["name"]: p["value"] for p in self.access_point.forge.as_json(resource.parameter)
            }
            scores = {
                p["name"]: p["value"] for p in self.access_point.forge.as_json(resource.score)
            }

            scores_validation = {}
            if hasattr(resource, "scoreValidation"):
                scores_validation = {
                    p["name"]: p["value"]
                    for p in self.access_point.forge.as_json(resource.scoreValidation)
                }

            passed_validation = None
            if hasattr(resource, "passedValidation"):
                passed_validation = resource.passedValidation

            if hasattr(resource, "githash"):
                githash = resource.githash
            else:
                githash = ""

            # WARNING: should be self.brain_region.brainRegion.label in the future

            model = {
                "emodel": self.emodel,
                "species": self.species,
                "brain_region": self.brain_region["brainRegion"]["label"],
                "fitness": resource.fitness,
                "parameters": params,
                "scores": scores,
                "scores_validation": scores_validation,
                "validated": passed_validation,
                "optimizer": resource.optimizer,
                "seed": resource.seed,
                "githash": githash,
            }

            models.append(model)

        return models

    def get_distributions(self, distributions):
        """Fetch channel distribution from Nexus by names."""

        distributions_definitions = {}

        for dist in distributions:

            resource = self.access_point.fetch_one(
                filters={
                    "type": "ElectrophysiologyFeatureOptimisationChannelDistribution",
                    "channelDistribution": dist,
                }
            )

            distributions_definitions[dist] = {
                    "fun": resource.function,
                    "soma_ref_point": resource.somaReferenceLocation,
                }

            if hasattr(resource, "parameter"):
                if isinstance(resource.parameter, list):
                    distributions_definitions[dist]["parameters"] = resource.parameter
                else:
                    distributions_definitions[dist]["parameters"] = [resource.parameter]

        return distributions_definitions

    def get_parameters(self):
        """Get the definition of the parameters to optimize from the
            optimization parameters resources, as well as the
            locations of the mechanisms. Also returns the names of the mechanisms.

        Returns:
            params_definition (dict): of the format:
                definitions = {
                        'distributions':
                            {'distrib_name': {
                                'function': function,
                                'parameters': ['param_name']}
                             },
                        'parameters':
                            {'sectionlist_name': [
                                    {'name': param_name1, 'val': [lbound1, ubound1]},
                                    {'name': param_name2, 'val': 3.234}
                                ]
                             }
                    }
            mech_definition (dict): of the format:
                mechanisms_definition = {
                    section_name1: {
                        "mech":[
                            mech_name1,
                            mech_name2
                        ]
                    },
                    section_name2: {
                        "mech": [
                            mech_name3,
                            mech_name4
                        ]
                    }
                }
            mech_names (list): list of mechanisms names

        """

        resources_params = self.access_point.fetch(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationParameter",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            }
        )

        if resources_params is None:
            raise NexusAccessPointException(
                "Could not get model parameters for emodel %s" % self.emodel
            )

        params_definition = {"parameters": {}}
        mech_definition = {}
        mechanisms_names = []
        distributions = []

        for resource in resources_params:

            param_def = {"name": resource.parameter.name}

            if resource.parameter.minValue == resource.parameter.maxValue:
                param_def["val"] = resource.parameter.minValue
            else:
                param_def["val"] = [resource.parameter.minValue, resource.parameter.maxValue]

            if resource.channelDistribution != "constant":
                param_def["dist"] = resource.channelDistribution
                distributions.append(resource.channelDistribution)

            if resource.location in params_definition["parameters"]:
                params_definition["parameters"][resource.location].append(param_def)
            else:
                params_definition["parameters"][resource.location] = [param_def]

            if (
                hasattr(resource, "subCellularMechanism")
                and resource.subCellularMechanism is not None
            ):

                mechanisms_names.append(resource.subCellularMechanism)

                if resource.subCellularMechanism != "pas":
                    # TODO when instantaneous registering:
                    # resource_mech = self.access_point.fetch_one(
                    #    filters={
                    #        "type": "SubCellularModel",
                    #        "subCellularMechanism": resource.subCellularMechanism,
                    #    }
                    # )

                    resource_mech = self.access_point.fetch(
                        filters={
                            "type": "SubCellularModel",
                            "subCellularMechanism": resource.subCellularMechanism,
                        }
                    )
                    resource_mech = resource_mech[0]

                    is_stochastic = resource_mech.stochastic
                    filepath = self.access_point.download(
                        resource_mech.modelScript.id, "./mechanisms/"
                    )
                    # Rename the file in case its different from the name of the resource
                    filepath = pathlib.Path(filepath)
                    if filepath.stem != resource_mech.name:
                        filepath.rename(
                            pathlib.Path(filepath.parent / f"{resource_mech.name}{filepath.suffix}")
                        )

                else:
                    is_stochastic = False

                if resource.location in mech_definition:
                    if (
                        resource.subCellularMechanism
                        not in mech_definition[resource.location]["mech"]
                    ):
                        mech_definition[resource.location]["mech"].append(
                            resource.subCellularMechanism
                        )
                        mech_definition[resource.location]["stoch"].append(is_stochastic)
                else:
                    mech_definition[resource.location] = {
                        "mech": [resource.subCellularMechanism],
                        "stoch": [is_stochastic],
                    }

        params_definition["distributions"] = self.get_distributions(set(distributions))
        
        # Remove the parameters of to the distributions that are not used
        tmp_params = {}
        for loc, params in params_definition["parameters"].items():
            if "distribution_" in loc:
                if loc.split("distribution_")[1] not in distributions:
                    continue
            tmp_params[loc] = params
        params_definition["parameters"] = tmp_params
        
        # It is necessary to sort the parameters as it impacts the order of
        # the parameters in the checkpoint.pkl
        # TODO: Find a better solution. Right now, if a new parameter is added,
        # it will break the order as it sorted alphabetically
        ordered_params_definition = OrderedDict()

        for loc in sorted(params_definition["parameters"].keys()):
            ordered_params_definition[loc] = sorted(
                params_definition["parameters"][loc], key=lambda k: k["name"].lower()
            )

        params_definition["parameters"] = ordered_params_definition

        return params_definition, mech_definition, set(mechanisms_names)

    def get_opt_targets(self, include_validation):
        """Get the optimisation and validation targets from Nexus."""

        resources_opt_target = self.access_point.fetch(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationTarget",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            }
        )

        if resources_opt_target is None:
            logger.warning(
                "NexusForge warning: could not get optimisation targets for emodel %s", self.emodel
            )

        if include_validation:
            resources_val_target = self.access_point.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureValidationTarget",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=True),
                    "brainLocation": self.brain_region,
                }
            )

            if resources_val_target is None:
                logger.warning(
                    "NexusForge warning: could not get validation targets for emodel %s",
                    self.emodel,
                )
                resources_val_target = []

            return resources_opt_target + resources_val_target

        return resources_opt_target

    def fetch_extraction_protocol(self, resource_target):
        """Fetch a singular extraction protocol resource based on an ecode name and amplitude"""

        resources_protocol = self.access_point.fetch(
            filters={
                "type": "ElectrophysiologyFeatureExtractionProtocol",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
                "stimulus": {
                    "stimulusType": {"label": resource_target.stimulus.stimulusType.label},
                },
            }
        )

        # This makes up for the fact that the sitmulus target (amplitude) cannot
        # be used directly for the fetch as filters does not alllow to check
        # equality of lists.
        if resources_protocol:
            resources = []
            for r in resources_protocol:
                tmp_amp = None
                if isinstance(r.stimulus.stimulusTarget, list):
                    tmp_amp = int(r.stimulus.stimulusTarget[0])
                else:
                    tmp_amp = int(r.stimulus.stimulusTarget)
                if tmp_amp == int(resource_target.stimulus.target):
                    resources.append(r)

        if resources is None:
            raise NexusAccessPointException(
                "Could not get protocol %s %s %% for emodel %s"
                % (
                    resource_target.stimulus.stimulusType.label,
                    resource_target.stimulus.target,
                    self.emodel,
                )
            )

        if len(resources) > 1:
            raise NexusAccessPointException(
                "More than one protocol %s %s %% for emodel %s"
                % (
                    resource_target.stimulus.stimulusType.label,
                    resource_target.stimulus.target,
                    self.emodel,
                )
            )

        protocol_name = "{}_{}".format(
            resources[0].stimulus.stimulusType.label,
            resources[0].stimulus.stimulusTarget[0],
        )

        return resources[0], protocol_name

    def get_protocols(self, include_validation=False):
        """Get the protocol definitions used to instantiate the CellEvaluator.

        Args:
            include_validation (bool): if True, returns the protocols for validation as well

        Returns:
            protocols_out (dict): protocols definitions. Of the format:
                {
                     protocolname: {
                         "type": "StepProtocol" or "StepThresholdProtocol",
                         "stimuli": {...}
                         "extra_recordings": ...
                     }
                }
        """

        protocols_out = {}

        for resource_target in self.get_opt_targets(include_validation):

            if resource_target.protocolType not in ["StepProtocol", "StepThresholdProtocol"]:
                continue

            resource_protocol, protocol_name = self.fetch_extraction_protocol(resource_target)

            stimulus = self.access_point.forge.as_json(resource_protocol.protocolDefinition.step)
            stimulus["holding_current"] = resource_protocol.protocolDefinition.holding.amplitude

            if hasattr(resource_target, "extraRecordings"):
                extra_recordings = resource_target.extraRecordings
            else:
                extra_recordings = []

            protocols_out[protocol_name] = {
                "type": resource_target.protocolType,
                "stimuli": stimulus,
                "extra_recordings": extra_recordings,
            }

        return protocols_out

    def fetch_extraction_efeature(self, name, stimulus, amplitude):
        """Fetch a singular extraction protocol resource based on an ecode name and amplitude"""

        resources_feature = self.access_point.fetch(
            filters={
                "type": "ElectrophysiologyFeature",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
                "feature": {"name": name},
                "stimulus": {"stimulusType": {"label": stimulus}},
            }
        )

        # This makes up for the fact that the sitmulus target (amplitude) cannot
        # be used directly for the fetch as filters does not alllow to check
        # equality of lists.
        resources = None
        if resources_feature:
            resources = []
            for r in resources_feature:

                tmp_amp = None

                if isinstance(r.stimulus.stimulusTarget, list):
                    tmp_amp = int(r.stimulus.stimulusTarget[0])
                else:
                    tmp_amp = int(r.stimulus.stimulusTarget)

                if tmp_amp == int(amplitude):
                    resources.append(r)

        if resources is None:
            raise NexusAccessPointException(
                "Could not get feature %s for %s %s %% for emodel %s"
                % (name, stimulus, amplitude, self.emodel)
            )

        if len(resources) > 1:
            raise NexusAccessPointException(
                "More than one feature %s for %s %s %% for emodel %s"
                % (name, stimulus, amplitude, self.emodel)
            )

        return resources[0]

    def get_features(self, include_validation=False):
        """Get the e-features used as targets in the CellEvaluator.

        Args:
            include_validation (bool): should the features for validation be returned as well

        Returns:
            efeatures_out (dict): efeatures definitions. Of the format:
                {
                    "protocol_name": {"soma.v":
                        [{"feature": feature_name, val:[mean, std]}]
                    }
                }
        """

        efeatures_out = {}

        for resource_target in self.get_opt_targets(include_validation):

            for feature in resource_target.feature:

                resource_feature = self.fetch_extraction_efeature(
                    feature.name,
                    resource_target.stimulus.stimulusType.label,
                    resource_target.stimulus.target,
                )

                feature_mean = next(
                    s.value for s in resource_feature.feature.series if s.statistic == "mean"
                )
                feature_std = next(
                    s.value
                    for s in resource_feature.feature.series
                    if s.statistic == "standard deviation"
                )

                feature_name = feature.name

                if resource_target.protocolType == "RinProtocol":
                    if feature.name == "ohmic_input_resistance_vb_ssse":
                        protocol_name = "RinProtocol"
                    elif feature.name == "voltage_base":
                        protocol_name = "SearchHoldingCurrent"
                        feature_name = "steady_state_voltage_stimend"
                    else:
                        continue

                elif resource_target.protocolType == "RMPProtocol":
                    if feature.name == "steady_state_voltage_stimend":
                        protocol_name = "RMPProtocol"
                    else:
                        continue

                elif resource_target.protocolType == "RinProtocol":
                    if feature.name == "voltage_base":
                        protocol_name = "SearchHoldingCurrent"
                        feature_name = "steady_state_voltage_stimend"
                    else:
                        continue

                else:
                    protocol_name = "{}_{}".format(
                        resource_target.stimulus.stimulusType.label,
                        resource_target.stimulus.target,
                    )
                    feature_name = feature.name

                if protocol_name not in efeatures_out:
                    efeatures_out[protocol_name] = {"soma.v": []}

                efeatures_out[protocol_name]["soma.v"].append(
                    {
                        "feature": feature_name,
                        "val": [feature_mean, feature_std],
                        "strict_stim": True,
                    }
                )

                if hasattr(feature.onsetTime, "value") and feature.onsetTime.value is not None:
                    efeatures_out[protocol_name]["soma.v"][-1][
                        "stim_start"
                    ] = feature.onsetTime.value
                if hasattr(feature.offsetTime, "value") and feature.offsetTime.value is not None:
                    efeatures_out[protocol_name]["soma.v"][-1][
                        "stim_end"
                    ] = feature.offsetTime.value

        # Add holding current and threshold current as target efeatures
        for current in ["holding_current", "threshold_current"]:

            resource_feature = self.access_point.fetch_one(
                filters={
                    "type": "ElectrophysiologyFeature",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=True),
                    "brainLocation": self.brain_region,
                    "feature": {"name": current},
                    "stimulus": {"stimulusType": {"label": "global"}},
                }
            )

            feature_mean = next(
                s.value for s in resource_feature.feature.series if s.statistic == "mean"
            )
            feature_std = next(
                s.value
                for s in resource_feature.feature.series
                if s.statistic == "standard deviation"
            )

            if current == "holding_current":
                protocol_name = "SearchHoldingCurrent"
            elif current == "threshold_current":
                protocol_name = "SearchThresholdCurrent"

            if protocol_name not in efeatures_out:
                efeatures_out[protocol_name] = {"soma.v": []}

            efeatures_out[protocol_name]["soma.v"].append(
                {"feature": "bpo_{}".format(current), "val": [feature_mean, feature_std]}
            )

        # Remove the empty protocols
        efeatures_out = {k: v for k, v in efeatures_out.items() if len(v["soma.v"])}

        return efeatures_out

    def get_morphologies(self):
        """Get the name and path (or data) to the morphologies used for optimisation.

        Returns:
            morphology_definition (list): [{'name': morph_name, 'path': 'morph_path'}]
        """

        resource_morphology = self.access_point.fetch_one(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationNeuronMorphology",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            }
        )

        filepath = self.access_point.download(resource_morphology.morphology.id)

        return {
            "name": resource_morphology.name,
            "path": str(filepath),
        }

    def get_name_validation_protocols(self):
        """Get the names of the protocols used for validation"""

        names = []

        resources_val_target = self.access_point.fetch(
            filters={
                "type": "ElectrophysiologyFeatureValidationTarget",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            }
        )

        if resources_val_target is None:
            logger.warning(
                "NexusForge warning: could not get validation targets for emodel %s",
                self.emodel,
            )
            return names

        for resource_target in resources_val_target:

            if resource_target.protocolType not in ["StepProtocol", "StepThresholdProtocol"]:
                continue

            _, protocol_name = self.fetch_extraction_protocol(resource_target)

            names.append(protocol_name)

        return names

    def has_protocols_and_features(self):
        """Check if the efeatures and protocol exist."""

        try:
            self.get_features()
        except NexusAccessPointException as e:
            if "Could not get " in str(e):
                return False
            raise e

        try:
            self.get_protocols()
        except NexusAccessPointException as e:
            if "Could not get protocol" in str(e):
                return False
            raise e

        return True

    def has_best_model(self, seed, githash):
        """Check if the best model has been stored."""

        if self.fetch_emodel(seed=seed, githash=githash):
            return True

        return False

    def is_checked_by_validation(self, seed, githash):
        """Check if the emodel with a given seed has been checked by Validation task.

        Reminder: the logic of validation is as follows:
            if None: did not go through validation
            if False: failed validation
            if True: passed validation
        """

        resources = self.fetch_emodel(seed=seed, githash=githash)

        if resources is None:
            return False

        if len(resources) > 1:
            raise NexusAccessPointException(
                "More than one model for emodel "
                "%s, seed %s, githash %s" % (self.emodel, seed, githash)
            )

        if hasattr(resources[0], "passedValidation") and resources[0].passedValidation is not None:
            return True

        return False

    def is_validated(self, githash):
        """Check if enough models have been validated.

        Reminder: the logic of validation is as follows:
            if None: did not go through validation
            if False: failed validation
            if True: passed validation
        """

        resources = self.fetch_emodel(githash=githash)

        if resources is None:
            return False

        n_validated = 0

        for resource in resources:
            if hasattr(resource, "passedValidation") and resource.passedValidation:
                n_validated += 1

        return n_validated >= self.pipeline_settings.n_model

    def get_morph_modifiers(self):
        """Get the morph modifiers if any."""
        return self.pipeline_settings.morph_modifiers
