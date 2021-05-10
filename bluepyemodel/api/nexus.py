"""API using Nexus Forge"""

import getpass
import logging
import os
import pathlib
from collections import OrderedDict

from entity_management.state import refresh_token
from kgforge.core import KnowledgeGraphForge
from kgforge.core import Resource

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger("__main__")


# pylint: disable=unused-argument,len-as-condition,bare-except,simplifiable-if-expression


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
]


class AccessPointException(Exception):
    """For Exceptions related to the NexusForgeAccessPoint"""


class NexusAPIException(Exception):
    """For Exceptions related to the NexusAPI"""


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


class NexusForgeAccessPoint:
    """Access point to Nexus Knowledge Graph using Nexus Forge"""

    def __init__(
        self,
        project="emodel_pipeline",
        organisation="demo",
        endpoint="https://bbp.epfl.ch/nexus/v1",
        forge_path=None,
        limit=1000,
        debug=False,
        cross_bucket=True,
        access_token=None,
    ):

        self.limit = limit
        self.debug = debug
        self.cross_bucket = cross_bucket

        self.access_token = access_token
        if not self.access_token:
            self.access_token = self.get_access_token()

        bucket = organisation + "/" + project
        self.forge = self.connect_forge(bucket, endpoint, self.access_token, forge_path)

    @staticmethod
    def get_access_token():
        """Define access token either from bbp-workflow or provided by the user"""

        try:
            access_token = refresh_token()
        except:  # noqa
            logger.info("Please get your Nexus access token from https://bbp.epfl.ch/nexus/web/.")
            access_token = getpass.getpass()

        return access_token

    @staticmethod
    def connect_forge(bucket, endpoint, access_token, forge_path=None):
        """Creation of a forge session"""

        if not forge_path:
            forge_path = (
                "https://raw.githubusercontent.com/BlueBrain/nexus-forge/"
                + "master/examples/notebooks/use-cases/prod-forge-nexus.yml"
            )

        forge = KnowledgeGraphForge(
            forge_path, bucket=bucket, endpoint=endpoint, token=access_token
        )

        return forge

    def register(self, resource_description, force_replace=False):
        """Register a resource from its dictionary description."""

        if "type" not in resource_description:
            raise AccessPointException("The resource description should contain 'type'.")

        # TODO: ask François to make it so search supports lists for this to work
        # previous_resources = self.fetch(resource_description)
        previous_resources = None

        if previous_resources and not force_replace:
            logger.warning(
                "The resource you are trying to register already exist and will be ignored."
            )
            return

        if previous_resources:
            for resource in previous_resources:
                self.forge.deprecate(resource)

        self.forge.register(self.forge.from_json(resource_description))

    def fetch(self, filters):
        """Retrieve resources based on filters.

        Args:
            filters (dict): keys and values used for the "WHERE". Should include "type" or "id".

        Returns:
            resources (list): list of resources
        """

        if "type" not in filters and "id" not in filters:
            raise AccessPointException("Search filters should contain either 'type' or 'id'.")

        resources = self.forge.search(
            filters, cross_bucket=self.cross_bucket, limit=self.limit, debug=self.debug
        )

        if resources:
            return resources

        logger.debug("No resources for filters: %s", filters)

        return None

    def fetch_one(self, filters):
        """Fetch one and only one resource based on filters."""

        resources = self.fetch(filters)

        if resources is None:
            raise AccessPointException("Could not get resource for filters %s" % filters)

        if len(resources) > 1:
            raise AccessPointException("More than one resource for filters %s" % filters)

        return resources[0]

    def download(self, resource_id, download_directory):
        """Download datafile from nexus if it doesn't already exist."""

        resource = self.forge.retrieve(resource_id, cross_bucket=True)

        if resource is None:
            raise AccessPointException("Could not find resource for id: %s" % resource_id)

        filename = resource.distribution.name
        file_path = pathlib.Path(download_directory) / filename

        if not file_path.is_file():
            self.forge.download(resource, "distribution.contentUrl", download_directory)

        return str(file_path)

    def deprecate(self, filters):
        """Deprecate resources based on filters."""

        resources = self.fetch(filters)

        if resources:
            for resource in resources:
                self.forge.deprecate(resource)


class NexusAPI(DatabaseAPI):
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
        """

        super().__init__(emodel)

        self.species = species
        self.brain_region = self.get_brain_region(brain_region)

        self.access_point = NexusForgeAccessPoint(
            project=project,
            organisation=organisation,
            endpoint=endpoint,
            forge_path=forge_path,
        )

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
            raise NexusAPIException("Unknown species %s." % self.species)

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

    def fetch_emodel(self, seed=None, githash=None):
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

        resources = self.access_point.fetch(filters)

        return resources

    def deprecate_project(self):
        """Deprecate all resources used or produced by BluePyModel. Use with extreme caution."""

        if not yesno("Confirm deprecation of all BluePyEmodel resources in Nexus project"):
            return

        for type_ in BPEM_NEXUS_SCHEMA:
            self.access_point.deprecate({"type": type_})

    def store_morphology(
        self,
        name=None,
        morphology_id=None,
        morphology_path=None,
        seclist_names=None,
        secarray_names=None,
        section_index=None,
    ):
        """Creates an ElectrophysiologyFeatureOptimisationNeuronMorphology resource based on a
        NeuronMorphology.

        Args:
            name (str): name of the morphology.
            morphology_id (str): nexus id of the NeuronMorphology.
            morphology_path (str): path to the file containing the morphology.
            seclist_names (list): Names of the lists of sections (ex: 'somatic')
            secarray_names (list): names of the sections (ex: 'soma')
            section_index (int): index to a specific section, used for non-somatic recordings.
        """

        if not morphology_id and not morphology_path:
            raise NexusAPIException("At least morphology_id or morphology_path should be informed.")

        if not morphology_id:

            if not name:
                name = pathlib.Path(morphology_path).stem

            resources = self.access_point.fetch({"type": "NeuronMorphology", "name": name})

            if resources:
                morphology_id = resources[0].id

            else:
                # TODO: pass file attachement logic to AccessPoint class
                distribution = self.access_point.forge.attach(morphology_path)
                resource = Resource(type="NeuronMorphology", name=name, distribution=distribution)
                self.access_point.forge.register(resource)
                morphology_id = resource.id

        else:

            resources = self.access_point.fetch({"id": morphology_id})

            if not resources:
                raise NexusAPIException("No matching resource for ephys_file_id %s" % morphology_id)

            if not name:
                name = resources[0].name

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
                "morphology": {"id": morphology_id},
                "sectionListNames": seclist_names,
                "sectionArrayNames": secarray_names,
                "sectionIndex": section_index,
            },
        )

    def store_recordings_metadata(
        self,
        cell_id=None,
        ecode=None,
        ephys_file_id=None,
        ephys_file_path=None,
        recording_metadata=None,
    ):
        """Creates an ElectrophysiologyFeatureExtractionTrace resource based on an ephys file.

        Args:
            cell_id (str): Nexus id of the cell on which the recording was performed.
            ecode (str): name of the eCode of interest.
            ephys_file_id (str): Nexus id of the file.
            ephys_file_path (str): path to the file.
            recording_metadata (dict): metadata such as ton, toff, v_unit associated to the traces
                of ecode in this file.
        """

        if recording_metadata is None:
            recording_metadata = {}

        if "protocol_name" not in recording_metadata and ecode:
            recording_metadata["protocol_name"] = ecode

        if not ephys_file_id and not ephys_file_path:
            raise NexusAPIException("At least ephys_file_id or ephys_file_path should be informed.")

        if not ephys_file_id:

            name = pathlib.Path(ephys_file_path).stem

            if not cell_id:
                cell_id = name

            resources = self.access_point.fetch({"type": "Trace", "name": name})

            if resources:
                ephys_file_id = resources[0].id

            else:
                # TODO: pass file attachement logic to AccessPoint class
                distribution = self.access_point.forge.attach(ephys_file_path)
                resource = Resource(type="Trace", name=name, distribution=distribution)
                self.access_point.forge.register(resource)
                ephys_file_id = resource.id

        else:

            resources = self.access_point.fetch({"id": ephys_file_id})

            if not resources:
                raise NexusAPIException("No matching resource for ephys_file_id %s" % ephys_file_id)

            if not cell_id:
                cell_id = resources[0].name

        self.access_point.register(
            {
                "type": ["Entity", "ElectrophysiologyFeatureExtractionTrace"],
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "brainLocation": self.brain_region,
                "trace": {"id": ephys_file_id},
                "cell": {"id": cell_id, "name": cell_id},
                "ecode": ecode,
                "recording_metadata": recording_metadata,
            },
        )

    def store_extraction_target(
        self, ecode, target_amplitudes, tolerances, use_for_rheobase, efeatures
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
            },
        )

    def store_opt_validation_target(
        self, type_, ecode, protocol_type, target_amplitude, efeatures, extra_recordings
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
        """

        if protocol_type not in [
            "StepProtocol",
            "StepThresholdProtocol",
            "RinProtocol",
            "RMPProtocol",
        ]:
            raise NexusAPIException("protocol_type %s unknown." % protocol_type)

        features = []
        for f in efeatures:
            features.append(
                {
                    "name": f,
                    "onsetTime": {"unitCode": "ms", "value": None},
                    "offsetTime": {"unitCode": "ms", "value": None},
                }
            )

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
        )

    def store_emodel_targets(
        self,
        ecode,
        efeatures,
        amplitude,
        extraction_tolerance,
        protocol_type="",
        used_for_extraction_rheobase=False,
        used_for_optimization=False,
        used_for_validation=False,
        extra_recordings=None,
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
        """

        if extra_recordings is None:
            extra_recordings = []

        if used_for_optimization and used_for_validation:
            raise NexusAPIException(
                "Both used_for_optimization and used_for_validation cannot be True for the"
                " same Ecode."
            )

        self.store_extraction_target(
            ecode=ecode,
            target_amplitudes=[amplitude],
            tolerances=[extraction_tolerance],
            use_for_rheobase=used_for_extraction_rheobase,
            efeatures=efeatures,
        )

        if used_for_optimization:
            self.store_opt_validation_target(
                "ElectrophysiologyFeatureOptimisationTarget",
                ecode=ecode,
                protocol_type=protocol_type,
                target_amplitude=amplitude,
                efeatures=efeatures,
                extra_recordings=extra_recordings,
            )

        elif used_for_validation:
            self.store_opt_validation_target(
                "ElectrophysiologyFeatureValidationTarget",
                ecode=ecode,
                protocol_type=protocol_type,
                target_amplitude=amplitude,
                efeatures=efeatures,
                extra_recordings=extra_recordings,
            )

    def store_optimisation_parameter(
        self, parameter_name, value, mechanism_name, location, distribution="constant"
    ):
        """Creates an ElectrophysiologyFeatureOptimisationParameter resource specifying a
        parameter of the model.

        Args:
            parameter_name (str): name of the parameter.
            value (list or float): value of the parameter. If value is a float, the parameter will
                be fixed. If value is a list of two elements, the first will be used as a lower
                bound and the second as an upper bound during the optimization.
            mechanism_name (str): name of the mechanism associated to the parameter.
            location (list): locations at which the parameter is present. The element of location
                have to be section list names.
            distribution (str): distribution of the parameters along the sections.
        """

        if isinstance(value, (list, tuple)):
            min_value, max_value = value
        else:
            min_value, max_value = value, value

        self.access_point.register(
            {
                "type": ["Entity", "Parameter", "ElectrophysiologyFeatureOptimisationParameter"],
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "brainLocation": self.brain_region,
                "parameter": {
                    "name": parameter_name,
                    "minValue": min_value,
                    "maxValue": max_value,
                    "unitCode": "",
                },
                "subCellularMechanism": mechanism_name,
                "location": [location],
                "channelDistribution": distribution,
            },
        )

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
            raise NexusAPIException("soma_reference_location should be between 0. and 1.")

        self.access_point.register(
            {
                "type": ["Entity", "ElectrophysiologyFeatureOptimisationChannelDistribution"],
                "channelDistribution": name,
                "function": function,
                "parameter": parameters,
                "somaReferenceLocation": soma_reference_location,
            }
        )

    def store_mechanism(
        self, name=None, mechanism_script_id=None, mechanism_script_path=None, stochastic=None
    ):

        """Creates an SubCellularModel based on a SubCellularModelScript.

        Args:
            name (str): name of the mechanism.
            mechanism_script_id (str): Nexus id of the mechanism.
            mechanism_script_path (path): path to the mechanism file.
            stochastic (bool): is the mechanism stochastic.
        """

        if mechanism_script_id:

            resource = self.access_point.fetch_one({"id": mechanism_script_id})

            if not name:
                name = resource.name

        elif mechanism_script_path:

            distribution = self.access_point.forge.attach(mechanism_script_path)

            if not name:
                name = pathlib.Path(mechanism_script_path).stem

            # TODO: pass file attachement logic to AccessPoint class
            resource = Resource(type="SubCellularModelScript", name=name, distribution=distribution)
            self.access_point.forge.register(resource)

            mechanism_script_id = resource.id

        elif name:

            resources = self.access_point.fetch_one(
                {"type": "SubCellularModelScript", "name": name}
            )
            mechanism_script_id = resources.id

        else:
            raise NexusAPIException(
                "At least name, mechanism_script_id or mechanism_script_path should be informed."
            )

        if stochastic is None:
            stochastic = True if "Stoch" in name else False

        self.access_point.register(
            {
                "type": ["Entity", "SubCellularModel"],
                "name": name,
                "subCellularMechanism": name,
                "modelScript": {"id": mechanism_script_id, "type": "SubCellularModelScript"},
                "stochastic": stochastic,
            }
        )

    def _build_extraction_targets(self, resources_target):
        """Create a dictionary definning the target of the feature extraction process"""

        targets = {}
        protocols_threshold = []

        for resource in resources_target:

            ecode = resource.stimulus.stimulusType.label

            # TODO: MAKE THIS WORK WITH TON AND TOFF
            efeatures = self.access_point.forge.as_json(resource.feature)
            if not isinstance(efeatures, list):
                efeatures = [efeatures]
            efeatures = [f["name"] for f in efeatures]

            if isinstance(resource.stimulus.tolerance, (int, float)):
                tolerances = [resource.stimulus.tolerance]
            else:
                tolerances = resource.stimulus.tolerance

            if isinstance(resource.stimulus.stimulusTarget, (int, float)):
                amplitudes = [resource.stimulus.stimulusTarget]
            else:
                amplitudes = resource.stimulus.stimulusTarget

            if ecode in targets:
                targets[ecode]["tolerances"] += tolerances
                targets[ecode]["amplitudes"] += amplitudes
                targets[ecode]["efeatures"] += efeatures

            else:
                targets[ecode] = {
                    "tolerances": tolerances,
                    "amplitudes": amplitudes,
                    "efeatures": efeatures,
                    "location": resource.stimulus.recordingLocation,
                }

            if hasattr(resource.stimulus, "threshold") and resource.stimulus.threshold:
                protocols_threshold.append(ecode)

        return targets, set(protocols_threshold)

    def _build_extraction_metadata(self, targets):
        """
        Create a dictionary that informs which files should be used for which
        target. It also specifies the metadata (such as units or ljp) associated
        to the files.

        This function also download the files in ./ephys_data, if they are not already
        present.
        """

        traces_metadata = {}

        for ecode in targets:

            resources_ephys = self.access_point.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureExtractionTrace",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=True),
                    "brainLocation": self.brain_region,
                    "ecode": ecode,
                }
            )

            if resources_ephys is None:
                raise NexusAPIException(
                    "Could not get ephys files for ecode %s,  emodel %s" % ecode,
                    self.emodel,
                )

            for resource in resources_ephys:

                cell_name = resource.cell.name
                ecode = resource.ecode

                if cell_name not in traces_metadata:
                    traces_metadata[cell_name] = {}

                if ecode not in traces_metadata[cell_name]:
                    traces_metadata[cell_name][ecode] = []

                download_directory = "./ephys_data/{}/{}/".format(self.emodel, cell_name)
                filepath = self.access_point.download(resource.trace.id, download_directory)

                recording_metadata = self.access_point.forge.as_json(resource.recording_metadata)
                recording_metadata["filepath"] = str(filepath)

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
            }
        )
        if resources_extraction_target is None:
            logger.warning(
                "NexusForge warning: could not get extraction metadata for emodel %s", self.emodel
            )
            return traces_metadata, targets, protocols_threshold

        targets, protocols_threshold = self._build_extraction_targets(resources_extraction_target)
        if not protocols_threshold:
            raise NexusAPIException(
                "No eCode have been informed to compute the rheobase during extraction."
            )

        traces_metadata = self._build_extraction_metadata(targets)

        return traces_metadata, targets, protocols_threshold

    def register_efeature(self, name, val, protocol_name=None, protocol_amplitude=None):
        """Register an ElectrophysiologyFeature resource"""

        resource_description = {
            "type": ["Entity", "ElectrophysiologyFeature"],
            "eModel": self.emodel,
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

        if protocol_name and protocol_amplitude:

            # TODO: How to get the ontology for the stimulus ?
            #  is the url string always of the same format ?

            resource_description["stimulus"] = {
                "stimulusType": {
                    "id": "http://bbp.epfl.ch/neurosciencegraph/ontologies"
                    "/stimulustypes/{}".format(protocol_name),
                    "label": protocol_name,
                },
                "stimulusTarget": float(protocol_amplitude),
                "recordingLocation": "soma",
            }

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
        name_Rin_protocol,
        name_rmp_protocol,
        validation_protocols,
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

    def store_protocols(self, stimuli, validation_protocols):
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
                    "definition": {
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
        """

        parameters_resource = []
        scores_resource = []
        scores_validation_resource = []

        if scores_validation:
            for k, v in scores_validation.items():
                scores_validation_resource.append({"name": k, "value": v, "unitCode": ""})

        if scores is not None:
            for k, v in scores.items():
                scores_resource.append({"name": k, "value": v, "unitCode": ""})

        if params is not None:
            for k, v in params.items():
                parameters_resource.append({"name": k, "value": v, "unitCode": ""})

        resources = self.fetch_emodel(seed=seed, githash=githash)

        if resources:
            logger.warning(
                "The resource Emodel %s, seed %s, githash %s already exist. "
                "Deprecate it if you wish to replace it.",
                self.emodel,
                seed,
                githash,
            )

        resource_dependencies = self._build_model_dependencies()

        pip_freeze = os.popen("pip freeze").read()

        self.access_point.register(
            {
                "type": ["Entity", "EModel"],
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "brainLocation": self.brain_region,
                "name": "{}_{}_{}".format(self.emodel, githash, seed),
                "fitness": sum(list(scores.values())),
                "parameter": parameters_resource,
                "score": scores_resource,
                "scoreValidation": scores_validation_resource,
                "passedValidation": validated,
                "optimizer": str(optimizer_name),
                "seed": seed,
                "githash": githash,
                "pip_freeze": pip_freeze,
                "resource_dependencies": resource_dependencies,
            },
        )

    def _build_model_dependencies(self):
        """Find all resources used during the building of the emodel"""

        parameters = self.access_point.fetch(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationParameter",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            }
        )

        distributions = []
        for resource in parameters:
            if resource.channelDistribution != "constant":
                distributions.append(resource.channelDistribution)
        distributions = self.get_distributions(set(distributions))

        mechanisms = []
        for resource in parameters:
            if (
                hasattr(resource, "subCellularMechanism")
                and resource.subCellularMechanism is not None
                and resource.subCellularMechanism != "pas"
            ):
                mechanisms.append(
                    self.access_point.fetch_one(
                        filters={
                            "type": "SubCellularModel",
                            "subCellularMechanism": resource.subCellularMechanism,
                        }
                    )
                )

        opt_targets = self.get_opt_targets(include_validation=True)

        target_protocols = []
        for resource_target in opt_targets:
            if resource_target.protocolType not in ["StepProtocol", "StepThresholdProtocol"]:
                continue
            resource_protocol, _ = self.fetch_extraction_protocol(resource_target)
            target_protocols.append(resource_protocol)

        target_efeatures = []
        for resource_target in opt_targets:
            for feature in resource_target.feature:
                target_efeatures.append(
                    self.fetch_extraction_efeature(
                        feature.name,
                        resource_target.stimulus.stimulusType.label,
                        resource_target.stimulus.target,
                    )
                )

        morphology = self.access_point.fetch_one(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationNeuronMorphology",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
                "brainLocation": self.brain_region,
            }
        )

        return {
            "opt_targets": opt_targets,
            "target_protocols": target_protocols,
            "target_efeatures": target_efeatures,
            "parameters": parameters,
            "distributions": distributions,
            "mechanisms": mechanisms,
            "morphology": morphology,
        }

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
                githash = None

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

            # TODO: HANDLE SEVERAL PARAMETERS
            if hasattr(resource, "parameter"):
                distributions_definitions[dist] = {
                    "fun": resource.function,
                    "parameters": [resource.parameter],
                }
            else:
                distributions_definitions[dist] = {"fun": resource.function}

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
            raise NexusAPIException("Could not get model parameters for emodel %s" % self.emodel)

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
                    resource_mech = self.access_point.fetch_one(
                        filters={
                            "type": "SubCellularModel",
                            "subCellularMechanism": resource.subCellularMechanism,
                        }
                    )
                    is_stochastic = resource_mech.stochastic
                    _ = self.access_point.download(resource_mech.modelScript.id, "./mechanisms/")
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
            resources_protocol = [
                r
                for r in resources_protocol
                if int(r.stimulus.stimulusTarget) == int(resource_target.stimulus.target)
            ]

        if resources_protocol is None:
            raise NexusAPIException(
                "Could not get protocol %s %s %% for emodel %s"
                % (
                    resource_target.stimulus.stimulusType.label,
                    resource_target.stimulus.target,
                    self.emodel,
                )
            )

        if len(resources_protocol) > 1:
            raise NexusAPIException(
                "More than one protocol %s %s %% for emodel %s"
                % (
                    resource_target.stimulus.stimulusType.label,
                    resource_target.stimulus.target,
                    self.emodel,
                )
            )

        protocol_name = "{}_{}".format(
            resources_protocol[0].stimulus.stimulusType.label,
            resources_protocol[0].stimulus.stimulusTarget,
        )

        return resources_protocol[0], protocol_name

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

            stimulus = self.access_point.forge.as_json(resource_protocol.definition.step)
            stimulus["holding_current"] = resource_protocol.definition.holding.amplitude

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
        if resources_feature:
            resources_feature = [
                r for r in resources_feature if int(r.stimulus.stimulusTarget) == int(amplitude)
            ]

        if resources_feature is None:
            raise NexusAPIException(
                "Could not get feature %s for %s %s %% for emodel %s"
                % (name, stimulus, amplitude, self.emodel)
            )

        if len(resources_feature) > 1:
            raise NexusAPIException(
                "More than one feature %s for %s %s %% for emodel %s"
                % (name, stimulus, amplitude, self.emodel)
            )

        return resources_feature[0]

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

        download_directory = "./morphology/{}/".format(self.emodel)
        filepath = self.access_point.download(resource_morphology.morphology.id, download_directory)

        return {
            "name": resource_morphology.name,
            "path": str(filepath),
        }

    def get_morph_modifiers(self):
        """Retrieve the morph modifiers if any."""

        # TODO:
        return None

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
        except NexusAPIException as e:
            if "Could not get " in str(e):
                return False
            raise e

        try:
            self.get_protocols()
        except NexusAPIException as e:
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
            raise NexusAPIException(
                "More than one model for emodel "
                "%s, seed %s, githash %s" % (self.emodel, seed, githash)
            )

        if hasattr(resources[0], "passedValidation") and resources[0].passedValidation is not None:
            return True

        return False

    def is_validated(self, githash, n_models_to_pass_validation):
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

        return n_validated >= n_models_to_pass_validation
