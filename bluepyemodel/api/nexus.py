"""API using Nexus Forge"""

import getpass
import logging
import pathlib

from kgforge.core import KnowledgeGraphForge
from kgforge.core import Resource

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger("__main__")


class NexusAPIException(Exception):
    """ For Exceptions related to the NexusAPI access point"""


class NexusAPI(DatabaseAPI):
    """Access point to Nexus Knowledge Graph through Nexus Forge"""

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

        self.subject = self.get_subject(species)

        bucket = organisation + "/" + project

        if not forge_path:
            forge_path = (
                "https://raw.githubusercontent.com/BlueBrain/nexus-forge/"
                + "master/examples/notebooks/use-cases/prod-forge-nexus.yml"
            )

        token = getpass.getpass()

        self.forge = KnowledgeGraphForge(forge_path, bucket=bucket, endpoint=endpoint, token=token)

        self.search_params = {
            "debug": False,
            "limit": 1000,
            "offset": None,
            "deprecated": False,
            "cross_bucket": True,
            "bucket": None,
        }

    @staticmethod
    def get_subject(species):

        if species == "human":
            return {
                "type": "Subject",
                "species": {
                    "id": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
                    "label": "Homo sapiens",
                },
            }

        # if species == "rat":
        #    pass  # TODO

        # if species == "mouse":
        #    pass  # TODO

        raise NexusAPIException("Unknown species {}.".format(species))

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

    def fetch(self, filters):
        """
        Retrieve resources based on filters.

        Args:
            filters (dict): keys and values used for the "WHERE". Should include

        Returns:
            resources (list): list of dict

        """

        if "type" not in filters and "id" not in filters:
            raise NexusAPIException("Search filters should contain either 'type' or 'id'.")

        resources = self.forge.search(filters, **self.search_params)

        if resources:
            return resources

        logger.warning("No resources for filters: %s", filters)
        return None

    def download(self, resource_id, download_directory):
        """ Download data file from nexus if it doesn't already exist. """

        resource = self.fetch({"id": resource_id})

        if resource is None:
            raise NexusAPIException("Could not download resource for id: {}".format(resource_id))

        filename = resource.distribution.name
        file_path = pathlib.Path(download_directory) / filename

        if not file_path.is_file():
            self.forge.download(resource, "distribution.contentUrl", download_directory)

        return file_path

    def deprecate(self, filters):
        """ Deprecate resources based on filters. """

        resources = self.fetch(filters)

        if resources is not None:

            for resource in resources:
                self.forge.deprecate(resource)

    def store_morphology(
        self,
        morphology_id,
        morphology_name=None,
        seclist_names=None,
        secarray_names=None,
        section_index=None,
    ):
        """Creates an ElectrophysiologyFeatureOptimisationNeuronMorphology resource based on a
        NeuronMorphology.

        Args:
            morphology_id (str): nexus id of the NeuronMorphology.
            morphology_name (str): name of the morphology.
            seclist_names (list):
            secarray_names (list):
            section_index (int):
        """

        resource = self.forge.from_json(
            {
                "type": [
                    "Entity",
                    "ElectrophysiologyFeatureOptimisationNeuronMorphology",
                ],
                "name": morphology_name,
                "eModel": self.emodel,
                "subject": self.subject,
                "morphology": {"id": morphology_id},
                "sectionListNames": seclist_names,
                "sectionArrayNames": secarray_names,
                "sectionIndex": section_index,
            }
        )

        self.register(resource)

    def store_ephys_trace(
        self,
        ephys_file_id,
        cell_id,
        ecode,
        time_unit,
        voltage_unit,
        current_unit,
        liquid_junction_potential,
        onset_time=None,
        offset_time=None,
        end_time=None,
        time_mid=None,
        time_mid2=None,
    ):
        """Creates an ElectrophysiologyFeatureExtractionTrace resource based on an ephys file.

        Args:
            ephys_file_id (str):
            cell_id (str):
            ecode (str):
            time_unit (str):
            voltage_unit (str):
            current_unit (str):
            liquid_junction_potential (float): in mV. Will be SUBSTRACTED from the voltage trace.
            onset_time (float): in ms.
            offset_time (float): in ms.
            end_time (float): in ms.
            time_mid (float): in ms.
            time_mid2 (float): in ms.
        """

        resource = self.forge.from_json(
            {
                "type": ["Entity", "ElectrophysiologyFeatureExtractionTrace"],
                "eModel": self.emodel,
                "subject": self.subject,
                "trace": {"id": ephys_file_id},
                "cell": {"id": cell_id},
                "stimulus": {
                    "stimulusType": {"label": ecode},
                    "onsetTime": {"unitCode": "ms", "value": onset_time},
                    "offsetTime": {"unitCode": "ms", "value": offset_time},
                    "endTime": {"unitCode": "ms", "value": end_time},
                },
                "time": {"unitCode": time_unit},
                "voltage": {"unitCode": voltage_unit},
                "current": {"unitCode": current_unit},
                "timeBetween": {"unitCode": "ms", "value": time_mid},
                "timeBetween2": {"unitCode": "ms", "value": time_mid2},
                "liquidJunctionPotential": {"unitCode": "mV", "value": liquid_junction_potential},
            }
        )

        self.register(resource)

    def store_extraction_target(
        self, ecode, target_amplitudes, tolerances, use_for_rheobase, efeatures
    ):
        """Creates an ElectrophysiologyFeatureExtractionTarget resource used as target for the
        e-features extraction process.

        Args:
            ecode (str):
            target_amplitudes (list):
            tolerances (list):
            use_for_rheobase (bool):
            efeatures (list):
        """

        features = [{"name": f} for f in efeatures]

        resource = self.forge.from_json(
            {
                "type": ["Entity", "ElectrophysiologyFeatureExtractionTarget"],
                "eModel": self.emodel,
                "subject": self.subject,
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

    def _store_opt_validation_target(
        self, type_, ecode, protocol_type, target_amplitude, efeatures, extra_recordings
    ):
        """Creates an ElectrophysiologyFeatureExtractionTarget resource used as target for the
        e-features extraction process.

        Args:
            type_ (str): type of the Nexus Entity.
            ecode (str): name of the eCode of the protocol.
            protocol_type (list): type of the protocol ("StepProtocol" or "StepThresholdProtocol").
            target_amplitude (float): amplitude of the step of the protocol. Expressed as a
                percentage of the threshold amplitude (rheobase).
            efeatures (list): list of efeatures name used as targets for this protocol.
            extra_recordings (list): definition of additional recordings used for this protocol.
        """

        if protocol_type not in ["StepProtocol", "StepThresholdProtocol"]:
            raise NexusAPIException(f"protocol_type {protocol_type} unknown.")

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
                "subject": self.subject,
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

    def store_optimisation_target(
        self, ecode, protocol_type, target_amplitude, efeatures, extra_recordings
    ):
        """Creates an ElectrophysiologyFeatureOptimisationTarget resource specifying which
        features will be used as target during optimisation.

        Args:
            ecode (str): name of the eCode of the protocol.
            protocol_type (str): type of the protocol ("StepProtocol" or "StepThresholdProtocol").
            target_amplitude (float): amplitude of the step of the protocol. Expressed as a
                percentage of the threshold amplitude (rheobase).
            efeatures (list): list of efeatures name used as targets for this protocol.
            extra_recordings (list): definition of additional recordings used for this protocol.
        """

        self._store_opt_validation_target(
            "ElectrophysiologyFeatureOptimisationTarget",
            ecode,
            protocol_type,
            target_amplitude,
            efeatures,
            extra_recordings,
        )

    def store_validation_target(
        self, ecode, protocol_type, target_amplitude, efeatures, extra_recordings
    ):
        """Creates an ElectrophysiologyFeatureValidationTarget resource specifying which
        features will be used as target for validation.

        Args:
            ecode (str): name of the eCode of the protocol.
            protocol_type (str): type of the protocol ("StepProtocol" or "StepThresholdProtocol").
            target_amplitude (float): amplitude of the step of the protocol. Expressed as a
                percentage of the threshold amplitude (rheobase).
            efeatures (list): list of efeatures name used as targets for this protocol.
            extra_recordings (list): definition of additional recordings used for this protocol.
        """

        self._store_opt_validation_target(
            "ElectrophysiologyFeatureValidationTarget",
            ecode,
            protocol_type,
            target_amplitude,
            efeatures,
            extra_recordings,
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
                "subject": self.subject,
                "parameter": {
                    "name": parameter_name,
                    "minValue": min_value,
                    "maxValue": max_value,
                    "unitCode": "",
                },
                "subCellularMechanism": mechanism_name,
                "location": location,
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

    def store_mechanism(self, name, mechanism_id, stochastic):
        """Creates an SubCellularModel based on a SubCellularModelScript.

        Args:
            name (str): name of the mechanism.
            mechanism_id (str): Nexus id of the mechanism.
            stochastic (bool): is the mechanism stochastic.
        """

        resource = self.forge.from_json(
            {
                "type": ["Entity", "SubCellularModel"],
                "name": name,
                "subCellularMechanism": name,
                "modelScript": {"id": mechanism_id, "type": "SubCellularModelScript"},
                "stochastic": stochastic,
            }
        )

        self.register(resource)

    def get_extraction_metadata(self):
        """Gather the metadata used to build the config dictionary used as an
        input by BluePyEfe.

        Returns:
            cells (dict): return the cells recordings metadata
            protocols (dict): return the protocols metadata

        """

        targets = {}
        protocols_threshold = []
        traces_metadata = {}

        resources_extraction_target = self.fetch(
            filters={
                "type": "ElectrophysiologyFeatureExtractionTarget",
                "eModel": self.emodel,
                "subject": self.subject,
            }
        )

        if resources_extraction_target is None:
            logger.warning(
                "NexusForge warning: could not get extraction metadata " "for emodel %s",
                self.emodel,
            )
            return traces_metadata, targets, protocols_threshold

        for resource in resources_extraction_target:

            ecode = resource.stimulus.stimulusType.label

            targets[ecode] = {
                "tolerances": resource.stimulus.tolerance,
                "targets": resource.stimulus.stimulusTarget,
                "efeatures": resource.feature,
                "location": resource.stimulus.recordingLocation,
            }

            if resource.threshold:
                protocols_threshold.append(ecode)

        for ecode in targets:

            resources_ephys = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureExtractionTrace",
                    "eModel": self.emodel,
                    "subject": self.subject,
                    "stimulus": {"stimulusType": ecode},
                }
            )

            if resources_ephys is None:
                logger.warning(
                    "NexusForge warning: could not get ephys files for ecode %s,  emodel %s",
                    ecode,
                    self.emodel,
                )

            for resource in resources_ephys:

                cell_id = resource.cell.id
                ecode = resource.stimulus.stimulusType.label

                if cell_id not in traces_metadata:
                    traces_metadata[cell_id] = {}

                if ecode not in traces_metadata[cell_id]:
                    traces_metadata[cell_id][ecode] = []

                download_directory = "./ephys_data/{}/{}/".format(self.emodel, cell_id)
                filepath = self.download(resource.trace.id, download_directory)

                metadata = {
                    "filepath": filepath,
                    "ton": resource.stimulus.onsetTime.value,
                    "toff": resource.stimulus.offsetTime.value,
                    "tend": resource.stimulus.endTime.value,
                    "tmid": resource.timeBetween.value,
                    "tmid2": resource.timeBetween2.value,
                    "i_unit": resource.current.unitCode,
                    "v_unit": resource.voltage.unitCode,
                    "t_unit": resource.time.unitCode,
                }

                traces_metadata[cell_id][ecode].append(metadata)

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
            name_Rin_protocol (str): name of the protocol associated with the efeatures used for
                the computation of the input resistance scores during optimisation.
            name_rmp_protocol (str): name of the protocol associated with the efeatures used for
                the computation of the resting membrane potential scores during optimisation.
            validation_protocols (list): names and targets of the protocol that will be used for
                validation only.
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
                        "subject": self.subject,
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
                    "subject": self.subject,
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
                    "subject": self.subject,
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
                            "amplitude": protocol["step"]["amplitude"],
                            "thresholdPercentage": protocol["step"]["amp_perc"],
                            "duration": protocol["step"]["duration"],
                            "totalDuration": protocol["step"]["totduration"],
                        },
                        "holding": {
                            "delay": protocol["holding"]["delay"],
                            "amplitude": protocol["holding"]["amplitude"],
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
        validated=False,
        scores_validation=None,
    ):
        """Store an emodel obtained from BluePyOpt in a EModel ressource

        Args:
            scores (dict): scores of the efeatures. Of the format {"objective_name": score}.
            params (dict): values of the parameters. Of the format {"param_name": param_value}.
            optimizer_name (str): name of the optimizer (IBEA, CMA, ...).
            seed (int): seed used by the optimizer.
            githash (string): githash associated with the configuration files.
            validated (bool): has the model been through validation.
            scores_validation (dict): scores of the validation efeatures. Of the format
                {"objective_name": score}.
        """

        scores_validation_resource = []
        parameters_resource = []
        scores_resource = []

        # TODO: Check that passedValidation false/true/None logic is correct

        if scores_validation is not None:
            for k, v in scores_validation:
                scores_validation_resource.append({"name": k, "value": v, "unitCode": ""})

        if scores is not None:
            for k, v in scores:
                scores_resource.append({"name": k, "value": v, "unitCode": ""})

        if params is not None:
            for k, v in params:
                parameters_resource.append({"name": k, "value": v, "unitCode": ""})

        resource = self.forge.from_json(
            {
                "type": ["Entity", "EModel"],
                "name": "{} e-model".format(self.emodel),
                "description": "This entity is about an {} e-model".format(self.emodel),
                "eModel": self.emodel,
                "subject": self.subject,
                "fitness": sum(list(scores.values())),
                "parameter": parameters_resource,
                "score": scores_resource,
                "scoreValidation": scores_validation_resource,
                "passedValidation": validated,
                "optimizer": str(optimizer_name),
                "seed": seed,
                "generation": {
                    "type": "Generation",
                    "activity": {
                        "wasAssociatedWith": {
                            "type": ["Agent", "SoftwareAgent"],
                            "softwareSourceCode": {
                                "type": "SoftwareSourceCode",
                                "version": githash,
                            },
                        }
                    },
                },
            }
        )

        self.register(resource)

    def get_emodels(self):
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

        resources = self.fetch(
            filters={"type": "EModel", "eModel": self.emodel, "subject": self.subject}
        )

        if resources is None:
            logger.warning("NexusForge warning: could not get emodels for emodel %s", self.emodel)
            return models

        for resource in resources:
            params = {p.name: p.values for p in resource.parameter}
            scores = {p.name: p.values for p in resource.score}
            scores_validation = {p.name: p.values for p in resource.scoreValidation}

            model = {
                "emodel": self.emodel,
                "species": self.subject,
                "fitness": resource.fitness,
                "parameters": params,
                "scores": scores,
                "scores_validation": scores_validation,
                "validated": resource.passedValidation,
                "optimizer": resource.optimizer,
                "seed": resource.seed,
            }

            models.append(model)

        return models

    def _get_mechanism(self, mechanism_name):

        resources = self.fetch(
            filters={
                "type": "SubCellularModel",
                "eModel": self.emodel,
                "subject": self.subject,
                "subCellularMechanism": mechanism_name,
            }
        )

        if resources is None:
            raise NexusAPIException(f"Missing mechanism {mechanism_name} for emodel {self.emodel}")

        if len(resources) != 1:
            raise NexusAPIException(
                f"More than one mechanism {mechanism_name} for emodel {self.emodel}"
            )

        return resources[0].stochastic, resources[0].modelScript.id

    def _get_distributions(self, distributions):

        distributions_definitions = {}

        for dist in distributions:

            resources = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureOptimisationChannelDistribution",
                    "eModel": self.emodel,
                    "subject": self.subject,
                    "subCellularMechanism": dist,
                }
            )

            if resources is None:
                raise NexusAPIException(f"Missing distribution {dist} for emodel {self.emodel}")

            if len(resources) != 1:
                raise NexusAPIException(
                    f"More than one distribution {dist} for emodel {self.emodel}"
                )

            distributions_definitions[dist] = {
                "fun": resources[0].function,
                "parameters": resources[0].parameter,
            }

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
                "subject": self.subject,
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

            if resource["channelDistribution"] != "constant":
                param_def["dist"] = resource.channelDistribution
                distributions.append(resource.channelDistribution)

            for loc in resource.locations:

                if loc in params_definition["parameters"]:
                    params_definition["parameters"][loc].append(param_def)
                else:
                    params_definition["parameters"][loc] = [param_def]

                if resource["subCellularMechanism"] is not None:

                    mechanisms_names.append(resource.subCellularMechanism)

                    is_stochastic, id_ = self._get_mechanism(resource.subCellularMechanism)

                    _ = self.download(id_, "./mechanisms/")

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

        return params_definition, mech_definition, mechanisms_names

    @staticmethod
    def _handle_extra_recordings(extra_recordings):
        """ Fetch the information needed to be able to use the extra recordings. """

        extra_recordings_out = []

        for extra_recording in extra_recordings:

            # TODO:
            if extra_recording["type"] == "somadistanceapic":
                raise NexusAPIException(
                    "extra_recording of type somadistanceapic not implemented yet for"
                    " the NexusForge API."
                )

            extra_recordings_out.append(extra_recording)

        return extra_recordings_out

    def _get_opt_targets(self, include_validation):
        """ Get the optimisation and validation targets from Nexus. """

        resources_opt_target = self.fetch(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationTarget",
                "eModel": self.emodel,
                "subject": self.subject,
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
                    "subject": self.subject,
                }
            )

            logger.warning(
                "NexusForge warning: could not get validation targets for emodel %s", self.emodel
            )

            return resources_opt_target + resources_val_target

        return resources_opt_target

    def get_protocols(self, include_validation=False):
        """Get some of the protocols from the "Extracted protocols" resources depending
            on "Optimization and validation targets" ressources.

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

        for resource_target in self._get_opt_targets(include_validation):

            resources_protocol = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureExtractionProtocol",
                    "eModel": self.emodel,
                    "subject": self.subject,
                    "stimulus": {
                        "stimulusType": {
                            "label": resource_target["stimulus"]["stimulusType"]["label"]
                        },
                        "stimulusTarget": resource_target["stimulus"]["target"],
                    },
                }
            )

            if resources_protocol is None:
                raise NexusAPIException(
                    f"Could not get protocol {resource_target.stimulus.stimulusType.label}"
                    " {resource_target.stimulus.target} %% for emodel {self.emodel}"
                )

            if len(resources_protocol) > 1:
                raise NexusAPIException(
                    f"More than one protocol {resource_target.stimulus.stimulusType.label}"
                    " {resource_target.stimulus.target} %% for emodel {self.emodel}"
                )

            protocol_name = "{}_{}".format(
                resources_protocol[0].stimulus.stimulusType.label,
                resources_protocol[0].stimulus.target,
            )

            stimulus = resources_protocol[0].definition.step
            stimulus["holding_current"] = resources_protocol[0].definition.holding.amplitude

            extra_recordings = self._handle_extra_recordings(resource_target.extraRecordings)

            protocols_out[protocol_name] = {
                "type": resource_target.protocolType,
                "stimuli": stimulus,
                "extra_recordings": extra_recordings,
            }

        return protocols_out

    def get_features(self, include_validation=False):
        """Get the efeatures from the "Extracted e-features" ressources  depending
            on "Optimization and validation targets" ressources.

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

        for resource_target in self._get_opt_targets(include_validation):

            for feature in resource_target.feature:

                resources_feature = self.fetch(
                    filters={
                        "type": "ElectrophysiologyFeature",
                        "eModel": self.emodel,
                        "subject": self.subject,
                        "feature": {"name": feature.name},
                        "stimulus": {
                            "stimulusType": {"label": resource_target.stimulus.stimulusType.label},
                            "stimulusTarget": resource_target.stimulus.target,
                        },
                    }
                )

                if resources_feature is None:
                    raise NexusAPIException(
                        f"Could not get feature {feature.name} for "
                        "{resource_target.stimulus.stimulusType.label} "
                        "{resource_target.stimulus.target} %% for emodel {self.emodel}"
                    )

                if len(resources_feature) > 1:
                    raise NexusAPIException(
                        f"More than one feature {feature.name} for "
                        "protocol {resource_target.stimulus.stimulusType.label} "
                        "{resource_target.stimulus.target} %% for emodel {self.emodel}"
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

                    if feature.onsetTime.value is not None:
                        efeatures_out[protocol_name]["soma.v"][-1][
                            "stim_start"
                        ] = feature.onsetTime.value

                    if feature.offsetTime.value is not None:
                        efeatures_out[protocol_name]["soma.v"][-1][
                            "stim_end"
                        ] = feature.offsetTime.value

        for current in ["holding_current", "threshold_current"]:

            resources_feature = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeature",
                    "eModel": self.emodel,
                    "subject": self.subject,
                    "feature": {"name": current},
                    "stimulus": {"stimulusType": {"label": "global"}},
                }
            )

            if resources_feature is None:
                raise NexusAPIException(f"Could not get {current} for emodel {self.emodel}")

            if len(resources_feature) > 1:
                raise NexusAPIException(f"More than one {current} for emodel {self.emodel}")

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
                "subject": self.subject,
            }
        )

        if resources_morphology is None:
            raise NexusAPIException(
                f"Could not get morphology for optimisation of emodel {self.emodel}"
            )

        if len(resources_morphology) > 1:
            raise NexusAPIException(
                f"More than one morphology for optimisation of emodel {self.emodel}"
            )

        download_directory = "./morphology/{}/".format(self.emodel)
        filepath = self.download(resources_morphology[0].morphology.id, download_directory)

        morphology_definition = [
            {
                "name": resources_morphology[0].name,
                "path": filepath,
            }
        ]

        return morphology_definition
