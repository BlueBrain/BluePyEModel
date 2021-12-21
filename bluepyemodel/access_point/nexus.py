"""Access point using Nexus Forge"""
import logging
import os
import pathlib

import numpy
import pandas
from kgforge.specializations.resources import Dataset

from bluepyemodel.access_point.access_point import DataAccessPoint
from bluepyemodel.access_point.forge_access_point import AccessPointException
from bluepyemodel.access_point.forge_access_point import NexusForgeAccessPoint
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.emodel_pipeline.utils import yesno
from bluepyemodel.evaluation.fitness_calculator_configuration import FitnessCalculatorConfiguration
from bluepyemodel.model.neuron_model_configuration import NeuronModelConfiguration
from bluepyemodel.tools import search_pdfs

# pylint: disable=simplifiable-if-expression,too-many-arguments,undefined-variable,unused-argument

logger = logging.getLogger("__main__")


BPEM_NEXUS_SCHEMA = [
    "ElectrophysiologyFeatureExtractionTrace",
    "ElectrophysiologyFeatureExtractionTarget",
    "ElectrophysiologyFeatureOptimisationTarget",
    "EModel",
    "EModelPipelineSettings",
    "EModelConfiguration",
    "FitnessCalculatorConfiguration",
]


class NexusAccessPointException(Exception):
    """For Exceptions related to the NexusAccessPoint"""


def format_dict_for_resource(d):
    """Translates a dictionary to a list of the format used by resources"""

    out = []

    if d is None:
        return out

    for k, v in d.items():

        if v is None or numpy.isnan(v):
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
        access_token=None,
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
            access_token (str): Nexus connection token.
        """

        super().__init__(emodel, ttype, iteration_tag)

        self.species = species
        self.brain_region = self.get_brain_region(brain_region)

        self.access_point = NexusForgeAccessPoint(
            project=project,
            organisation=organisation,
            endpoint=endpoint,
            forge_path=forge_path,
            iteration_tag=iteration_tag,
            access_token=access_token,
        )

        self.pipeline_settings = self.load_pipeline_settings(strict=False)

    def get_subject(self, for_search=False):
        """Get the ontology of a species based on the species name."""

        if self.species == "human":
            subject = {
                "type": "Subject",
                "species": {"id": "http://purl.obolibrary.org/obo/NCBITaxon_9606"},
            }
            if not for_search:
                subject["species"]["label"] = "Homo sapiens"

        elif self.species == "rat":
            subject = {
                "type": "Subject",
                "species": {"id": "http://purl.obolibrary.org /obo/NCBITaxon_7370"},
            }
            if not for_search:
                subject["species"]["label"] = "Musca domestica"

        elif self.species == "mouse":
            subject = {
                "type": "Subject",
                "species": {"id": "http://purl.obolibrary.org/obo/NCBITaxon_10090"},
            }
            if not for_search:
                subject["species"]["label"] = "Mus musculus"

        else:
            raise NexusAccessPointException(f"Unknown species {self.species}.")

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

    def fetch_emodel(self, seed=None, use_version=True):
        """Fetch an emodel"""

        filters = {
            "type": "EModel",
            "eModel": self.emodel,
            "ttype": self.ttype,
            "subject": self.get_subject(for_search=True),
            "brainLocation": self.brain_region,
        }

        if seed:
            filters["seed"] = int(seed)

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

        resource = self.access_point.forge.search(
            {
                "type": "EModelPipelineSettings",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            }
        )

        resource = self.access_point.fetch_one(
            filters={
                "type": "EModelPipelineSettings",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            },
            strict=strict,
        )

        settings = {}

        if resource:

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
                "name_gene_map",
                "optimisation_batch_size",
                "max_n_batch",
                "path_extract_config",
                "name_Rin_protocol",
                "name_rmp_protocol",
                "validation_protocols",
                "additional_protocols",
                "compile_mechanisms",
                "threshold_based_evaluator",
                "model_configuration_name",
            ]:
                if setting in resource_dict:
                    settings[setting] = resource_dict[setting]

        else:
            logger.warning(
                "No EModelPipelineSettings for emodel %s, default values will be used", self.emodel
            )

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

    def load_ic_map(self):
        """Get the ion channel/genes map from Nexus"""

        resource_ic_map = self.access_point.fetch_one(
            {"type": "IonChannelMapping", "name": "icmapping"}, use_version=False
        )

        return self.access_point.download(resource_ic_map.id)

    def get_t_types(self, table_name):
        """Get the list of t-types available for the present emodel"""

        df, _ = self.load_channel_gene_expression(table_name)
        return df.loc[self.emodel].index.get_level_values("t-type").unique().tolist()

    def download_mechanisms(self, mechanisms):
        """Download the mod files if not already downloaded"""

        for mechanism in mechanisms:

            if mechanism == "pas":
                continue

            resources = self.access_point.fetch(
                {"type": "SubCellularModelScript", "name": mechanism}, use_version=False
            )

            # Genetic channel can have several versions, we want the most recent one:
            if len(resources) > 1 and all(hasattr(r, "version") for r in resources):
                resource = sorted(resources, key=lambda x: x.version)[-1]
            else:
                resource = resources[0]

            mode_file_name = f"{mechanism}.mod"
            if os.path.isfile(f"./mechanisms/{mode_file_name}"):
                continue

            filepath = self.access_point.download(resource.id, "./mechanisms/")

            # Rename the file in case it's different from the name of the resource
            filepath = pathlib.Path(filepath)
            if filepath.stem != mechanism:
                filepath.rename(pathlib.Path(filepath.parent / mode_file_name))

    def download_morphology(self, name):
        """Download a morphology by name if not already downloaded"""

        resource = self.access_point.fetch_one(
            {"type": "NeuronMorphology", "name": name}, use_version=False
        )

        return self.access_point.download(resource.id, "./morphology/")

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
            raise NexusAccessPointException(f"No matching resource for {id_} {name}")

        self.access_point.register(
            {
                "type": "ElectrophysiologyFeatureExtractionTrace",
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
            raise NexusAccessPointException(f"protocol_type {protocol_type} unknown.")

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
        used_for_optimization=True,
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
        extra_recordings (list): definitions of additional recordings to use for this protocol.
        efel_settings (dict): eFEL settings.
        """

        if efel_settings is None:
            efel_settings = {}

        if extra_recordings is None:
            extra_recordings = []

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
        additional_protocols=None,
        compile_mechanisms=False,
        name_Rin_protocol=None,
        name_rmp_protocol=None,
        model_configuration_name=None,
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
            "threshold_based_evaluator": threshold_based_evaluator,
            "optimizer": optimizer,
            "optimisation_params": optimisation_params,
            "optimisation_timeout": optimisation_timeout,
            "max_ngen": max_ngen,
            "validation_threshold": validation_threshold,
            "optimisation_batch_size": optimization_batch_size,
            "max_n_batch": max_n_batch,
            "n_model": n_model,
            "name_gene_map": name_gene_map,
            "plot_extraction": plot_extraction,
            "plot_optimisation": plot_optimisation,
            "threshold_efeature_std": threshold_efeature_std,
            "morph_modifiers": morph_modifiers,
            "additional_protocols": additional_protocols,
            "compile_mechanisms": compile_mechanisms,
            "name_Rin_protocol": name_Rin_protocol,
            "name_rmp_protocol": name_rmp_protocol,
            "model_configuration_name": model_configuration_name,
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
                    f"Could not get ephys files for ecode {protocol}, emodel {self.emodel}"
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

        resources = self.access_point.fetch(
            {
                "type": "ElectrophysiologyFeatureExtractionTarget",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            }
        )

        if resources is None:
            logger.warning(
                "Could not get extraction metadata from Nexus for emodel %s", self.emodel
            )
            return traces_metadata, targets, protocols_threshold

        targets, protocols_threshold = self._build_extraction_targets(resources)
        if not protocols_threshold:
            raise NexusAccessPointException(
                "No eCode have been informed to compute the rheobase during extraction."
            )

        traces_metadata = self._build_extraction_metadata(targets)

        return traces_metadata, targets, protocols_threshold

    def store_fitness_calculator_configuration(self, configuration):
        """Store a fitness calculator configuration as a resource of type
        FitnessCalculatorConfiguration"""

        resource = {
            "type": ["FitnessCalculatorConfiguration"],
            "emodel": self.emodel,
            "ttype": self.ttype,
        }

        resource.update(configuration.as_dict())

        self.access_point.register(
            resource,
            filters_existance={
                "type": "FitnessCalculatorConfiguration",
                "emodel": self.emodel,
                "ttype": self.ttype,
            },
            replace=True
        )

    def get_fitness_calculator_configuration(self):
        """Get the configuration of the fitness calculator (efeatures and protocols)"""

        resource = self.access_point.fetch_one(
            {
                "type": "FitnessCalculatorConfiguration",
                "emodel": self.emodel,
                "ttype": self.ttype,
            }
        )

        config_dict = self.access_point.forge.as_json(resource)

        configuration = FitnessCalculatorConfiguration(
            efeatures=config_dict["efeatures"],
            protocols=config_dict["protocols"],
            name_rmp_protocol=self.pipeline_settings.name_rmp_protocol,
            name_rin_protocol=self.pipeline_settings.name_Rin_protocol,
            threshold_efeature_std=self.pipeline_settings.threshold_efeature_std,
            validation_protocols=self.pipeline_settings.validation_protocols,
        )

        return configuration

    def has_fitness_calculator_configuration(self):
        """Check if the fitness calculator configuration exists"""

        is_present = True
        try:
            self.access_point.fetch_one(
                {
                    "type": "FitnessCalculatorConfiguration",
                    "emodel": self.emodel,
                    "ttype": self.ttype,
                }
            )
        except AccessPointException:
            is_present = False

        return is_present

    def store_emodel(
        self,
        scores,
        params,
        optimizer_name,
        seed,
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

        pdf_dependencies = self._build_pdf_dependencies(seed)
        pip_freeze = os.popen("pip freeze").read()

        resource_description = {
            "type": ["Entity", "EModel"],
            "eModel": self.emodel,
            "subject": self.get_subject(for_search=False),
            "brainLocation": self.brain_region,
            "name": f"{self.emodel} {seed}",
            "fitness": sum(list(scores.values())),
            "parameter": parameters_resource,
            "score": scores_resource,
            "features": features_resource,
            "scoreValidation": scores_validation_resource,
            "passedValidation": validated,
            "optimizer": str(optimizer_name),
            "seed": int(seed),
            "pip_freeze": pip_freeze,
        }

        search = {
            "type": "EModel",
            "subject": self.get_subject(for_search=True),
            "brainLocation": self.brain_region,
            "seed": int(seed),
        }

        if self.ttype:
            resource_description["ttype"] = self.ttype
            search["ttype"] = self.ttype

        self.access_point.register(
            resource_description=resource_description,
            filters_existance=search,
            replace=True,
            tag=True,
            distributions=pdf_dependencies,
        )

    def _build_pdf_dependencies(self, seed):
        """Find all the pdfs associated to an emodel"""

        pdfs = []

        opt_pdf = search_pdfs.search_figure_emodel_optimisation(
            self.emodel, seed, self.ttype, self.iteration_tag
        )
        if opt_pdf:
            pdfs.append(opt_pdf)

        traces_pdf = search_pdfs.search_figure_emodel_traces(
            self.emodel, seed, self.ttype, self.iteration_tag
        )
        if traces_pdf:
            pdfs += [p for p in traces_pdf if p]

        scores_pdf = search_pdfs.search_figure_emodel_score(
            self.emodel, seed, self.ttype, self.iteration_tag
        )
        if scores_pdf:
            pdfs += [p for p in scores_pdf if p]

        parameters_pdf = search_pdfs.search_figure_emodel_parameters(
            self.emodel, self.ttype, self.iteration_tag
        )
        if parameters_pdf:
            pdfs += [p for p in parameters_pdf if p]

        return pdfs

    def get_emodels(self, emodels=None):
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

        if emodels is None:
            emodels = [self.emodel]

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

            if hasattr(resource, "iteration"):
                iteration_tag = resource.iteration
            else:
                iteration_tag = ""

            if hasattr(resource, "ttype"):
                ttype = resource.ttype
            else:
                ttype = None

            # WARNING: should be self.brain_region.brainRegion.label in the future

            model = {
                "emodel": self.emodel,
                "ttype": ttype,
                "species": self.species,
                "brain_region": self.brain_region["brainRegion"]["label"],
                "fitness": resource.fitness,
                "parameters": params,
                "scores": scores,
                "scores_validation": scores_validation,
                "validated": passed_validation,
                "optimizer": resource.optimizer,
                "seed": int(resource.seed),
                "iteration_tag": iteration_tag,
            }

            models.append(model)

        return models

    def store_model_configuration(self, configuration, path=None):
        """Store a model configuration as a resource of type EModelConfiguration"""

        resource = {"type": ["EModelConfiguration"], "emodel": self.emodel, "ttype": self.ttype}

        resource.update(configuration.as_dict())

        self.access_point.register(
            resource, filters_existance={"type": "EModelConfiguration", "name": resource["name"]}
        )

    def get_available_mechanisms(self):
        """Get the list of names of the available mechanisms"""

        resources = self.access_point.fetch({"type": "SubCellularModelScript"}, use_version=False)

        return {r.name for r in resources}

    def get_available_morphologies(self):
        """Get the list of names of the available morphologies"""

        resources = self.access_point.fetch({"type": "NeuronMorphology"}, use_version=False)

        return {r.name for r in resources}

    def get_model_configuration(self, configuration_name=None):
        """Get the configuration of the model, including parameters, mechanisms and distributions"""

        if configuration_name is None:
            configuration_name = self.pipeline_settings.model_configuration_name

        resource = self.access_point.fetch_one(
            {
                "type": "EModelConfiguration",
                "name": configuration_name,
            }
        )

        config_dict = self.access_point.forge.as_json(resource)
        for entry in ["distributions", "parameters", "mechanisms"]:
            if entry in config_dict:
                if isinstance(config_dict[entry], dict):
                    config_dict[entry] = [config_dict[entry]]

        morph_path = self.download_morphology(config_dict["morphology"]["name"])
        config_dict["morphology"]["path"] = morph_path

        model_configuration = NeuronModelConfiguration(
            configuration_name=configuration_name,
            available_mechanisms=self.get_available_mechanisms(),
            available_morphologies=self.get_available_morphologies(),
        )

        model_configuration.init_from_dict(config_dict)
        self.download_mechanisms(model_configuration.mechanism_names)

        return model_configuration

    def has_best_model(self, seed):
        """Check if the best model has been stored."""

        if self.fetch_emodel(seed=seed):
            return True

        return False

    def is_checked_by_validation(self, seed):
        """Check if the emodel with a given seed has been checked by Validation task.

        Reminder: the logic of validation is as follows:
            if None: did not go through validation
            if False: failed validation
            if True: passed validation
        """

        resources = self.fetch_emodel(seed=seed)

        if resources is None:
            return False

        if len(resources) > 1:
            raise NexusAccessPointException(
                f"More than one model for emodel {self.emodel}, seed {seed}, "
                f"iteration_tag {self.iteration_tag}"
            )

        if hasattr(resources[0], "passedValidation") and resources[0].passedValidation is not None:
            return True

        return False

    def is_validated(self):
        """Check if enough models have been validated.

        Reminder: the logic of validation is as follows:
            if None: did not go through validation
            if False: failed validation
            if True: passed validation
        """

        resources = self.fetch_emodel()

        if resources is None:
            return False

        n_validated = 0

        for resource in resources:
            if hasattr(resource, "passedValidation") and resource.passedValidation:
                n_validated += 1

        return n_validated >= self.pipeline_settings.n_model
