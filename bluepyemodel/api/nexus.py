"""API using Nexus Forge"""

import getpass
import logging
import pathlib
from collections import OrderedDict

from entity_management.state import refresh_token
from kgforge.core import KnowledgeGraphForge
from kgforge.core import Resource

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger("__main__")

# pylint: disable=unused-argument,len-as-condition,bare-except


class NexusAPIException(Exception):
    """For Exceptions related to the NexusAPI access point"""


class NexusAPI(DatabaseAPI):
    """Access point to Nexus Knowledge Graph using Nexus Forge"""

    def __init__(
        self,
        emodel,
        species,
        project="emodel_pipeline",
        organisation="Cells",
        endpoint="https://staging.nexus.ocp.bbp.epfl.ch/v1",
        forge_path=None,
    ):
        """Init

        Args:
            emodel (str): name of the emodel
            species (str): name of the species.
            project (str): name of the Nexus project.
            organisation (str): name of the Nexus organization to which the project belong.
            endpoint (str): Nexus endpoint.
            forge_path (str): path to a .yml used as configuration by nexus-forge.
        """

        super().__init__(emodel)

        self.species = species

        bucket = organisation + "/" + project

        if not forge_path:
            forge_path = (
                "https://raw.githubusercontent.com/BlueBrain/nexus-forge/"
                + "master/examples/notebooks/use-cases/prod-forge-nexus.yml"
            )

        try:
            access_token = refresh_token()
        except:  # noqa
            logger.info("Please get your Nexus access token from https://bbp.epfl.ch/nexus/web/.")
            access_token = getpass.getpass()

        self.forge = KnowledgeGraphForge(
            forge_path, bucket=bucket, endpoint=endpoint, token=access_token
        )

    def get_subject(self, for_search=False):
        """Get the ontology of a species based no the species name. The id is not used
        during search as if it is specified the search fail systematically (see with Nexus team)."""

        if self.species == "human":

            subject = {
                "type": "Subject",
                "species": {
                    "label": "Homo sapiens",
                },
            }

            if not for_search:
                subject["species"]["id"] = "http://purl.obolibrary.org/obo/NCBITaxon_9606"

            return subject

        if self.species == "rat":

            subject = {
                "type": "Subject",
                "species": {
                    "label": "Musca domestica",
                },
            }

            if not for_search:
                subject["species"]["id"] = "http://purl.obolibrary.org /obo/NCBITaxon_7370"

            return subject

        if self.species == "mouse":

            subject = {"type": "Subject", "species": {"label": "Mus musculus"}}

            if not for_search:
                subject["species"]["id"] = "http://purl.obolibrary.org/obo/NCBITaxon_10090"

            return subject

        raise NexusAPIException("Unknown species %s." % self.species)

    def register(self, resources):
        """
        Register a resource or resources using the forge.

        Args:
            resources (list): resources to store in the forge
        """

        # TODO: How to handle updating a resource versus registering ?
        if isinstance(resources, Resource):
            resources = [resources]

        if not isinstance(resources, list):
            raise NexusAPIException("resources should be a Resource or a list of Resources")

        self.forge.register(resources)

    def fetch(self, filters, cross_bucket=True, limit=1000, debug=False):
        """
        Retrieve resources based on filters.

        Args:
            filters (dict): keys and values used for the "WHERE". Should include
            cross_bucket (bool): specifies if the search should be performed in the current bucket,
                or in any of the other buckets accessible using the Aggregator View.
            limit (int): maximum number of resources that will be returned.

        Returns:
            resources (list): list of dict
        """

        if "type" not in filters and "id" not in filters:
            raise NexusAPIException("Search filters should contain either 'type' or 'id'.")

        resources = self.forge.search(filters, cross_bucket=cross_bucket, limit=limit, debug=debug)

        if resources:
            return resources

        logger.warning("No resources for filters: %s", filters)
        return None

    def download(self, resource_id, download_directory):
        """Download data file from nexus if it doesn't already exist."""

        resource = self.forge.retrieve(resource_id, cross_bucket=True)

        if resource is None:
            raise NexusAPIException("Could not download resource for id: %s" % resource_id)

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

    def store_morphology(
        self,
        name,
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

            resources = self.fetch({"type": "NeuronMorphology", "name": name})

            if resources:
                morphology_id = resources[0].id

            else:
                distribution = self.forge.attach(morphology_path)
                resource = Resource(type="NeuronMorphology", name=name, distribution=distribution)

                self.forge.register(resource)

                morphology_id = resource.id

        resource = self.forge.from_json(
            {
                "type": [
                    "Entity",
                    "ElectrophysiologyFeatureOptimisationNeuronMorphology",
                ],
                "name": name,
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "morphology": {"id": morphology_id},
                "sectionListNames": seclist_names,
                "sectionArrayNames": secarray_names,
                "sectionIndex": section_index,
            }
        )

        self.register(resource)

    def store_recordings_metadata(
        self, cell_id, ecode, ephys_file_id=None, ephys_file_path=None, recording_metadata=None
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

        if "protocol_name" not in recording_metadata:
            recording_metadata["protocol_name"] = ecode

        if not ephys_file_id and not ephys_file_path:
            raise NexusAPIException(
                "At least mechanism_script_id or mechanism_script_path should be informed."
            )

        if not ephys_file_id:

            name = pathlib.Path(ephys_file_path).stem

            resources = self.fetch({"type": "Trace", "name": name})

            if resources:
                ephys_file_id = resources[0].id

            else:
                distribution = self.forge.attach(ephys_file_path)
                resource = Resource(type="Trace", name=name, distribution=distribution)

                self.forge.register(resource)

                ephys_file_id = resource.id

        resource = self.forge.from_json(
            {
                "type": ["Entity", "ElectrophysiologyFeatureExtractionTrace"],
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "trace": {"id": ephys_file_id},
                "cell": {"id": cell_id, "name": cell_id},
                "ecode": ecode,
                "recording_metadata": recording_metadata,
            }
        )

        self.register(resource)

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

        resource = self.forge.from_json(
            {
                "type": ["Entity", "ElectrophysiologyFeatureExtractionTarget"],
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "stimulus": {
                    "stimulusType": {"label": ecode},
                    "stimulusTarget": target_amplitudes,
                    "tolerance": tolerances,
                    "threshold": use_for_rheobase,
                    "recordingLocation": "soma",
                },
                "feature": features,
            }
        )

        self.register(resource)

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

        resource = self.forge.from_json(
            {
                "type": ["Entity", "Target", type_],
                "protocolType": protocol_type,
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "stimulus": {
                    "stimulusType": {"label": ecode},
                    "target": target_amplitude,
                    "recordingLocation": "soma",
                },
                "feature": features,
                "extraRecordings": extra_recordings,
            }
        )

        self.register(resource)

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

        resource = self.forge.from_json(
            {
                "type": ["Entity", "Parameter", "ElectrophysiologyFeatureOptimisationParameter"],
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "parameter": {
                    "name": parameter_name,
                    "minValue": min_value,
                    "maxValue": max_value,
                    "unitCode": "",
                },
                "subCellularMechanism": mechanism_name,
                "location": [location],
                "channelDistribution": distribution,
            }
        )

        self.register(resource)

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

        if soma_reference_location > 1.0 or soma_reference_location < 0.0:
            raise NexusAPIException("soma_reference_location should be between 0. and 1.")

        resource = self.forge.from_json(
            {
                "type": ["Entity", "ElectrophysiologyFeatureOptimisationChannelDistribution"],
                "channelDistribution": name,
                "function": function,
                "parameter": parameters,
                "somaReferenceLocation": soma_reference_location,
            }
        )

        self.register(resource)

    def store_mechanism(
        self, name, mechanism_script_id=None, mechanism_script_path=None, stochastic=False
    ):

        """Creates an SubCellularModel based on a SubCellularModelScript.

        Args:
            name (str): name of the mechanism.
            mechanism_script_id (str): Nexus id of the mechanism.
            mechanism_script_path (path): path to the mechanism file.
            stochastic (bool): is the mechanism stochastic.
        """

        if not mechanism_script_id and not mechanism_script_path:
            raise NexusAPIException(
                "At least mechanism_script_id or mechanism_script_path should be informed."
            )

        if not mechanism_script_id:

            resources = self.fetch({"type": "SubCellularModelScript", "name": name})

            if resources:
                mechanism_script_id = resources[0].id

            else:
                distribution = self.forge.attach(mechanism_script_path)
                resource = Resource(
                    type="SubCellularModelScript", name=name, distribution=distribution
                )

                self.forge.register(resource)

                mechanism_script_id = resource.id

        resource = self.forge.from_json(
            {
                "type": ["Entity", "SubCellularModel"],
                "name": name,
                "subCellularMechanism": name,
                "modelScript": {"id": mechanism_script_id, "type": "SubCellularModelScript"},
                "stochastic": stochastic,
            }
        )

        self.register(resource)

    def get_extraction_metadata(self):
        """Gather the metadata used to build the config dictionary used as an
        input by BluePyEfe.

        Returns:
            traces_metadata (dict)
            targets (dict)
            protocols_threshold (list)
        """

        traces_metadata = {}
        targets = {}
        protocols_threshold = []

        resources_extraction_target = self.fetch(
            filters={
                "type": "ElectrophysiologyFeatureExtractionTarget",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
            }
        )

        if resources_extraction_target is None:
            logger.warning(
                "NexusForge warning: could not get extraction metadata for emodel %s", self.emodel
            )
            return traces_metadata, targets, protocols_threshold

        for resource in resources_extraction_target:

            ecode = resource.stimulus.stimulusType.label

            # TODO: MAKE THIS WORK WITH TON AND TOFF
            efeatures = self.forge.as_json(resource.feature)
            if not isinstance(efeatures, list):
                efeatures = [efeatures]
            efeatures = [f["name"] for f in efeatures]

            if ecode in targets:
                targets[ecode]["tolerances"] += resource.stimulus.tolerance
                targets[ecode]["amplitudes"] += resource.stimulus.stimulusTarget
                targets[ecode]["efeatures"] += efeatures

            else:
                targets[ecode] = {
                    "tolerances": resource.stimulus.tolerance,
                    "amplitudes": resource.stimulus.stimulusTarget,
                    "efeatures": efeatures,
                    "location": resource.stimulus.recordingLocation,
                }

            if (
                hasattr(resource.stimulus, "threshold")
                and resource.stimulus.threshold
                and ecode not in protocols_threshold
            ):
                protocols_threshold.append(ecode)

        for ecode in targets:

            resources_ephys = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureExtractionTrace",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=True),
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
                filepath = self.download(resource.trace.id, download_directory)

                recording_metadata = self.forge.as_json(resource.recording_metadata)
                recording_metadata["filepath"] = str(filepath)

                traces_metadata[cell_name][ecode].append(recording_metadata)

        if not (protocols_threshold):
            raise NexusAPIException(
                "No ecode have been specified to compute the rheobase during extraction."
            )

        return traces_metadata, targets, protocols_threshold

    def store_efeatures(
        self,
        efeatures,
        current,
        name_Rin_protocol,
        name_rmp_protocol,
        validation_protocols,
    ):
        """Store the efeatures and currents obtained from BluePyEfe in
        ElectrophysiologyFeature resources.

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

        resources = []

        # TODO: How to get the ontology for the stimulus ? is the url string always of the format ?

        for protocol in efeatures:

            for feature in efeatures[protocol]["soma"]:
                prot_name = "_".join(protocol.split("_")[:-1])
                prot_amplitude = protocol.split("_")[-1]

                resource = self.forge.from_json(
                    {
                        "type": ["Entity", "ElectrophysiologyFeature"],
                        "eModel": self.emodel,
                        "subject": self.get_subject(for_search=False),
                        "feature": {
                            "name": feature["feature"],
                            "value": [],
                            "series": [
                                {
                                    "statistic": "mean",
                                    "unitCode": "dimensionless",
                                    "value": feature["val"][0],
                                },
                                {
                                    "statistic": "standard deviation",
                                    "unitCode": "dimensionless",
                                    "value": feature["val"][1],
                                },
                            ],
                        },
                        "stimulus": {
                            "stimulusType": {
                                "id": "http://bbp.epfl.ch/neurosciencegraph/ontologies"
                                "/stimulustypes/{}".format(prot_name),
                                "label": prot_name,
                            },
                            "stimulusTarget": float(prot_amplitude),
                            "recordingLocation": "soma",
                        },
                    }
                )

                resources.append(resource)

        for cur in ["holding_current", "threshold_current"]:
            resource = self.forge.from_json(
                {
                    "type": ["Entity", "ElectrophysiologyFeature"],
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=False),
                    "feature": {
                        "name": cur,
                        "value": [],
                        "series": [
                            {
                                "statistic": "mean",
                                "unitCode": "dimensionless",
                                "value": current[cur][0],
                            },
                            {
                                "statistic": "standard deviation",
                                "unitCode": "dimensionless",
                                "value": current[cur][1],
                            },
                        ],
                    },
                    "stimulus": {
                        "stimulusType": {"id": "", "label": "global"},
                        "recordingLocation": "soma",
                    },
                }
            )

            resources.append(resource)

        self.register(resources)

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

        resources = []

        # TODO: How to get the ontology for the stimulus ? is the url string
        # always of the format ?

        for protocol_name, protocol in stimuli.items():
            prot_name = "_".join(protocol_name.split("_")[:-1])
            prot_amplitude = protocol_name.split("_")[-1]

            resource = self.forge.from_json(
                {
                    "type": ["Entity", "ElectrophysiologyFeatureExtractionProtocol"],
                    "name": "Electrophysiology feature extraction protocol",
                    "description": "Electrophysiology feature extraction protocol",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=False),
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
                }
            )

            resources.append(resource)

        self.register(resources)

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
        """Store an emodel obtained from BluePyOpt in a EModel ressource

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

        resources = self._get_emodel(seed=seed, githash=githash)

        if resources:

            if len(resources) > 1:
                raise Exception(
                    "More than one emodel %s, seed %s, githash %s" % (self.emodel, seed, githash)
                )

            logger.warning(
                "Emodel %s, seed %s, githash %s already exists and will be replaced",
                self.emodel,
                seed,
                githash,
            )
            self.forge.deprecate(resources[0])

        resource = self.forge.from_json(
            {
                "type": ["Entity", "EModel"],
                "name": "{}_{}_{}".format(self.emodel, githash, seed),
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=False),
                "fitness": sum(list(scores.values())),
                "parameter": parameters_resource,
                "score": scores_resource,
                "scoreValidation": scores_validation_resource,
                "passedValidation": validated,
                "optimizer": str(optimizer_name),
                "seed": seed,
                "githash": githash,
            }
        )

        self.register(resource)

    def _get_emodel(self, seed=None, githash=None):
        """Internal use only"""

        filters = {
            "type": "EModel",
            "eModel": self.emodel,
            "subject": self.get_subject(for_search=True),
        }

        if seed:
            filters["seed"] = seed

        if githash:
            filters["githash"] = githash

        resources = self.fetch(filters)

        return resources

    def get_emodels(self, emodels):
        """Get the list of emodels.

        Returns:
            models (list): return the emodels, of the format:
            [
                {
                    "emodel": ,
                    "species": ,
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

        resources = self._get_emodel()

        if resources is None:
            logger.warning("NexusForge warning: could not get emodels for emodel %s", self.emodel)
            return models

        for resource in resources:

            params = {p["name"]: p["value"] for p in self.forge.as_json(resource.parameter)}
            scores = {p["name"]: p["value"] for p in self.forge.as_json(resource.score)}

            scores_validation = {}
            if hasattr(resource, "scoreValidation"):
                scores_validation = {
                    p["name"]: p["value"] for p in self.forge.as_json(resource.scoreValidation)
                }

            passed_validation = None
            if hasattr(resource, "passedValidation"):
                passed_validation = resource.passedValidation

            if hasattr(resource, "githash"):
                githash = resource.githash
            else:
                githash = None

            model = {
                "emodel": self.emodel,
                "species": self.species,
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

    def _get_mechanism(self, mechanism_name):
        """Fetch mechanisms from Nexus by names."""

        resources = self.fetch(
            filters={
                "type": "SubCellularModel",
                "subCellularMechanism": mechanism_name,
            }
        )

        if resources is None:
            raise NexusAPIException(
                "Missing mechanism %s for emodel %s" % mechanism_name, self.emodel
            )

        if len(resources) != 1:
            raise NexusAPIException(
                "More than one mechanism %s for emodel %s" % mechanism_name, self.emodel
            )

        return resources[0].stochastic, resources[0].modelScript.id

    def _get_distributions(self, distributions):
        """Fetch channel distribution from Nexus by names."""

        distributions_definitions = {}

        for dist in distributions:

            resources = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureOptimisationChannelDistribution",
                    "channelDistribution": dist,
                }
            )

            if resources is None:
                raise NexusAPIException("Missing distribution %s for emodel %s" % dist, self.emodel)

            if len(resources) != 1:
                raise NexusAPIException(
                    "More than one distribution %s for emodel %s" % dist, self.emodel
                )

            # TODO: HANDL SEVERAL PARAMETERS
            if hasattr(resources[0], "parameter"):
                distributions_definitions[dist] = {
                    "fun": resources[0].function,
                    "parameters": [resources[0].parameter],
                }
            else:
                distributions_definitions[dist] = {"fun": resources[0].function}

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

        params_definition = {"distributions": {}, "parameters": {}}
        mech_definition = {}
        mechanisms_names = []
        distributions = []

        resources_params = self.fetch(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationParameter",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
            }
        )

        if resources_params is None:
            logger.warning(
                "NexusForge warning: could not get parameters for emodel %s", self.emodel
            )
            return params_definition, mech_definition, mechanisms_names

        for resource in resources_params:

            param_def = {
                "name": resource.parameter.name,
            }

            if resource.parameter.minValue == resource.parameter.maxValue:
                param_def["val"] = resource.parameter.minValue
            else:
                param_def["val"] = [
                    resource.parameter.minValue,
                    resource.parameter.maxValue,
                ]

            if resource.channelDistribution != "constant":
                param_def["dist"] = resource.channelDistribution
                distributions.append(resource.channelDistribution)

            loc = resource.location

            if loc in params_definition["parameters"]:
                params_definition["parameters"][loc].append(param_def)
            else:
                params_definition["parameters"][loc] = [param_def]

            if (
                hasattr(resource, "subCellularMechanism")
                and resource.subCellularMechanism is not None
            ):

                mechanisms_names.append(resource.subCellularMechanism)

                if resource.subCellularMechanism != "pas":
                    is_stochastic, id_ = self._get_mechanism(resource.subCellularMechanism)
                    _ = self.download(id_, "./mechanisms/")
                else:
                    is_stochastic = False

                if loc in mech_definition:

                    if resource.subCellularMechanism not in mech_definition[loc]["mech"]:
                        mech_definition[loc]["mech"].append(resource.subCellularMechanism)
                        mech_definition[loc]["stoch"].append(is_stochastic)

                else:
                    mech_definition[loc] = {
                        "mech": [resource.subCellularMechanism],
                        "stoch": [is_stochastic],
                    }

        params_definition["distributions"] = self._get_distributions(set(distributions))

        # It is necessary to sort the parameters as it impacts the order of
        # the parameters in the checkpoint.pkl
        # TODO: Find a better soolution. Right now, if a new parameter is added,
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

        resources_opt_target = self.fetch(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationTarget",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
            }
        )

        if resources_opt_target is None:
            logger.warning(
                "NexusForge warning: could not get optimisation targets for emodel %s", self.emodel
            )

        if include_validation:
            resources_val_target = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureValidationTarget",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=True),
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

    def get_protocols(self, include_validation=False):
        """Get the protocols definition used to instantiate the CellEvaluator.

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

            resources_protocol = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureExtractionProtocol",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=True),
                    "stimulus": {
                        "stimulusType": {"label": resource_target.stimulus.stimulusType.label},
                    },
                }
            )

            if resources_protocol:
                resources_protocol = [
                    r
                    for r in resources_protocol
                    if r.stimulus.stimulusTarget[0] == resource_target.stimulus.target
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
                resources_protocol[0].stimulus.stimulusTarget[0],
            )

            stimulus = self.forge.as_json(resources_protocol[0].definition.step)
            stimulus["holding_current"] = resources_protocol[0].definition.holding.amplitude

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

        efeatures_out = {
            "RMPProtocol": {"soma.v": []},
            "RinProtocol": {"soma.v": []},
            "SearchHoldingCurrent": {"soma.v": []},
            "SearchThresholdCurrent": {"soma.v": []},
        }

        for resource_target in self.get_opt_targets(include_validation):

            for feature in resource_target.feature:

                resources_feature = self.fetch(
                    filters={
                        "type": "ElectrophysiologyFeature",
                        "eModel": self.emodel,
                        "subject": self.get_subject(for_search=True),
                        "feature": {"name": feature.name},
                        "stimulus": {
                            "stimulusType": {"label": resource_target.stimulus.stimulusType.label},
                        },
                    }
                )

                if resources_feature:
                    resources_feature = [
                        r
                        for r in resources_feature
                        if r.stimulus.stimulusTarget[0] == resource_target.stimulus.target
                    ]

                if resources_feature is None:
                    raise NexusAPIException(
                        "Could not get feature %s for %s %s %% for emodel %s"
                        % (
                            feature.name,
                            resource_target.stimulus.stimulusType.label,
                            resource_target.stimulus.target,
                            self.emodel,
                        )
                    )

                if len(resources_feature) > 1:
                    raise NexusAPIException(
                        "More than one feature %s for %s %s %% for emodel %s"
                        % (
                            feature.name,
                            resource_target.stimulus.stimulusType.label,
                            resource_target.stimulus.target,
                            self.emodel,
                        )
                    )

                feature_mean = next(
                    s.value for s in resources_feature[0].feature.series if s.statistic == "mean"
                )
                feature_std = next(
                    s.value
                    for s in resources_feature[0].feature.series
                    if s.statistic == "standard deviation"
                )

                if (
                    resource_target.protocolType == "RinProtocol"
                    and feature.name == "ohmic_input_resistance_vb_ssse"
                ):

                    efeatures_out["RinProtocol"]["soma.v"].append(
                        {
                            "feature": feature.name,
                            "val": [feature_mean, feature_std],
                            "strict_stim": True,
                        }
                    )

                elif (
                    resource_target.protocolType == "RMPProtocol"
                    and feature.name == "steady_state_voltage_stimend"
                ):
                    efeatures_out["RMPProtocol"]["soma.v"].append(
                        {
                            "feature": feature.name,
                            "val": [feature_mean, feature_std],
                            "strict_stim": True,
                        }
                    )

                elif (
                    resource_target.protocolType == "RinProtocol" and feature.name == "voltage_base"
                ):

                    efeatures_out["SearchHoldingCurrent"]["soma.v"].append(
                        {
                            "feature": "steady_state_voltage_stimend",
                            "val": [feature_mean, feature_std],
                            "strict_stim": True,
                        }
                    )

                else:

                    protocol_name = "{}_{}".format(
                        resource_target.stimulus.stimulusType.label,
                        resource_target.stimulus.target,
                    )

                    if protocol_name not in efeatures_out:
                        efeatures_out[protocol_name] = {"soma.v": []}

                    efeatures_out[protocol_name]["soma.v"].append(
                        {
                            "feature": feature.name,
                            "val": [feature_mean, feature_std],
                            "strict_stim": True,
                        }
                    )

                    if hasattr(feature.onsetTime, "value") and feature.onsetTime.value is not None:
                        efeatures_out[protocol_name]["soma.v"][-1][
                            "stim_start"
                        ] = feature.onsetTime.value

                    if (
                        hasattr(feature.offsetTime, "value")
                        and feature.offsetTime.value is not None
                    ):
                        efeatures_out[protocol_name]["soma.v"][-1][
                            "stim_end"
                        ] = feature.offsetTime.value

        # Add holding current and threshold current as targets
        for current in ["holding_current", "threshold_current"]:

            resources_feature = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeature",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=True),
                    "feature": {"name": current},
                    "stimulus": {"stimulusType": {"label": "global"}},
                }
            )

            if resources_feature is None:
                logger.warning(
                    "Could not get %s for emodel %s. Will not be able to perform "
                    "threshold-based optimisation.",
                    current,
                    self.emodel,
                )
                continue

            if len(resources_feature) > 1:
                raise NexusAPIException("More than one %s for emodel %s" % current, self.emodel)

            feature_mean = next(
                s.value for s in resources_feature[0].feature.series if s.statistic == "mean"
            )
            feature_std = next(
                s.value
                for s in resources_feature[0].feature.series
                if s.statistic == "standard deviation"
            )

            if current == "holding_current":
                protocol_name = "SearchHoldingCurrent"
            elif current == "threshold_current":
                protocol_name = "SearchThresholdCurrent"

            efeatures_out[protocol_name]["soma.v"].append(
                {
                    "feature": "bpo_{}".format(current),
                    "val": [feature_mean, feature_std],
                }
            )

        # Remove the empty protocols
        efeatures_out = {k: v for k, v in efeatures_out.items() if len(v["soma.v"])}

        return efeatures_out

    def get_morphologies(self):
        """Get the name and path (or data) to the morphologies used for optimisation.

        Returns:
            morphology_definition (list): [{'name': morph_name, 'path': 'morph_path'}]
        """

        resources_morphology = self.fetch(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationNeuronMorphology",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
            }
        )

        if resources_morphology is None:
            raise NexusAPIException(
                "Could not get morphology for optimisation of emodel %s" % self.emodel
            )

        if len(resources_morphology) > 1:
            raise NexusAPIException(
                "More than one morphology for optimisation of emodel %s" % self.emodel
            )

        download_directory = "./morphology/{}/".format(self.emodel)
        filepath = self.download(resources_morphology[0].morphology.id, download_directory)

        morphology_definition = [
            {
                "name": resources_morphology[0].name,
                "path": str(filepath),
            }
        ]

        return morphology_definition

    def get_name_validation_protocols(self):
        """Get the names of the protocols used for validation"""

        names = []

        resources_val_target = self.fetch(
            filters={
                "type": "ElectrophysiologyFeatureValidationTarget",
                "eModel": self.emodel,
                "subject": self.get_subject(for_search=True),
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

            resources_protocol = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureExtractionProtocol",
                    "eModel": self.emodel,
                    "subject": self.get_subject(for_search=True),
                    "stimulus": {
                        "stimulusType": {"label": resource_target.stimulus.stimulusType.label},
                    },
                }
            )

            if resources_protocol:
                resources_protocol = [
                    r
                    for r in resources_protocol
                    if r.stimulus.stimulusTarget[0] == resource_target.stimulus.target
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
                resources_protocol[0].stimulus.stimulusTarget[0],
            )

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

        if self._get_emodel(seed=seed, githash=githash):
            return True

        return False

    def is_checked_by_validation(self, seed, githash):
        """Check if the emodel with a given seed has been checked by Validation task."""

        resources = self._get_emodel(seed=seed, githash=githash)

        if resources is None:
            return False

        if len(resources) > 1:
            raise NexusAPIException(
                "More than one emodel for emodel "
                "%s, seed %s, githash %s" % (self.emodel, seed, githash)
            )

        if hasattr(resources, "passedValidation") and resources.passedValidation is not None:
            return True

        return False

    def is_validated(self, githash, n_models_to_pass_validation):
        """Check if enough models have been validated."""

        resources = self._get_emodel(githash=githash)

        if resources is None:
            return False

        n_validated = 0

        for resource in resources:
            if hasattr(resource, "passedValidation") and resource.passedValidation:
                n_validated += 1

        return n_validated >= n_models_to_pass_validation
