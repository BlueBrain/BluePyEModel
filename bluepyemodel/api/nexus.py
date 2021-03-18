"""API using Nexus Forge"""

import getpass
import logging

from kgforge.core import KnowledgeGraphForge  # , Resource
from kgforge.core import Resource

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger("__main__")

# pylint: disable=W0231,W0221,W0613,W0715

# TODO: implement species
# TODO: Check that passedValidation false/true/None logic is correct


class Nexus_API(DatabaseAPI):
    """Access point to Nexus Knowledge Graph through Nexus Forge"""

    def __init__(
        self,
        project="emodel_pipeline",
        organisation="Cells",
        endpoint="https://staging.nexus.ocp.bbp.epfl.ch/v1",
        forge_path=None,
    ):
        """Init"""

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
            "cross_bucket": False,
            "bucket": None,
        }

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
            raise Exception("resources should be a Resource or a list of Resources")

        self.forge.register(resources)

    def fetch(self, filters):
        """
        Retrieve resources based on filters.

        Args:
            filters (dict): keys and values used for the "WHERE". Should include

        Returns:
            resources (list): list of dict

        """
        if "type" not in filters:
            raise Exception("Search filters should contain an entry 'type'.")

        resources = self.forge.search(filters, **self.search_params)

        if resources:
            return self.forge.as_dataframe(resources, store_metadata=True).to_dict(orient="records")

        logger.warning("No resources for filters: %s", filters)
        return None

    def get_extraction_metadata(self, emodel, species):
        """Gather the metadata used to build the config dictionary used as an
        input by BluePyEfe.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            cells (dict): return the cells recordings metadata
            protocols (dict): return the protocols metadata

        """

        targets = {}
        protocols_threshold = []
        traces_metadata = {}

        resources_extraction_target = self.fetch(
            filters={"type": "ElectrophysiologyFeatureExtractionTarget", "eModel": emodel}
        )

        if resources_extraction_target is None:
            logger.warning(
                "NexusForge warning: could not get extraction metadata " "for emodel %s", emodel
            )
            return traces_metadata, targets, protocols_threshold

        for t in resources_extraction_target:

            ecode = t["stimulus"]["stimulusType"]["label"]

            targets[ecode] = {
                "tolerances": t["stimulus"]["tolerance"],
                "targets": t["stimulus"]["stimulusTarget"],
                "efeatures": t["feature"],
                "location": t["stimulus"]["recordingLocation"],
            }

            if t["threshold"]:
                protocols_threshold.append(ecode)

        # TODO:
        resources_ephys_files = None
        # resources_ephys_files = self.fetch(
        #    filters={
        #        "type": "ElectrophysiologyFeatureExtractionTrace",
        #        "eModel": emodel,
        #        "stimulus": {"stimulusType" == tuple(targets.keys())},
        #    }
        # )

        if resources_ephys_files is None:
            logger.warning(
                "NexusForge warning: could not get ephys files metadata " "for emodel %s", emodel
            )
            return traces_metadata, targets, protocols_threshold

        for f in resources_ephys_files:

            cell_id = f["cell"]["id"]
            ecode = f["stimulus"]["stimulusType"]["stimulusType"]["label"]

            if cell_id not in traces_metadata:
                traces_metadata[cell_id] = {}

            if ecode not in traces_metadata[cell_id]:
                traces_metadata[cell_id][ecode] = []

            # TODO: Get or download traces or resources
            metadata = {
                "filepath": "",
                "ton": f["stimulus"]["onsetTime"]["value"],
                "toff": f["stimulus"]["offsetTime"]["value"],
                "tend": f["stimulus"]["endTime"]["value"],
                "tmid": f["timeBetween"]["value"],
                "tmid2": f["timeBetween2"]["value"],
                "i_unit": f["current"]["unitCode"],
                "v_unit": f["voltage"]["unitCode"],
                "t_unit": f["time"]["unitCode"],
            }

            traces_metadata[id][ecode].append(metadata)

        return traces_metadata, targets, protocols_threshold

    def store_efeatures(
        self,
        emodel,
        species,
        efeatures,
        current,
        name_Rin_protocol,
        name_rmp_protocol,
        validation_protocols,
    ):
        """Store the efeatures and currents obtained from BluePyEfe in
        ElectrophysiologyFeature resources.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
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
            name_Rin_protocol (str):
            name_rmp_protocol (str):
            validation_protocols (list):
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
                        "eModel": emodel,
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
                    "eModel": emodel,
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

    def store_protocols(self, emodel, species, stimuli, validation_protocols):
        """Store the protocols obtained from BluePyEfe in
            ElectrophysiologyFeatureExtractionProtocol resources.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
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
                    "eModel": "L23_PC",
                    "subject": {
                        "type": "Subject",
                        "species": {
                            "id": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
                            "label": "Homo sapiens",
                        },
                    },
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
        emodel,
        scores,
        params,
        optimizer_name,
        seed,
        githash="",
        validated=False,
        scores_validation=None,
        species=None,
    ):
        """Store an emodel obtained from BluePyOpt in a EModel ressource

        Args:
            emodel (str): name of the emodel
            scores (dict): scores of the efeatures
            params (dict): values of the parameters
            optimizer_name (str): name of the optimizer.
            seed (int): seed used for optimization.
            githash (string): githash for which the model has been generated.
            validated (bool): True if the model has been validated.
            scores_validation (dict):
            species (str): name of the species (rat, human, mouse)
        """

        scores_validation_resource = []
        parameters_resource = []
        scores_resource = []

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
                "name": "{} e-model".format(emodel),
                "description": "This entity is about an {} e-model".format(emodel),
                "eModel": emodel,
                "subject": {
                    "type": "Subject",
                    "species": {
                        "id": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
                        "label": "Homo sapiens",
                    },
                },
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

    def get_emodels(self, emodels, species):
        """Get the list of emodels matching the .

        Args:
            emodels (list): list of names of the emodel's metypes
            species (str): name of the species (rat, human, mouse)


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

        for emodel in emodels:

            resources = self.fetch(filters={"type": "EModel", "eModel": emodel})

            if resources is None:
                logger.warning("NexusForge warning: could not get emodels for emodel %s", emodel)
                continue

            for resource in resources:

                params = {p["name"]: p["values"] for p in resource["parameter"]}
                scores = {p["name"]: p["values"] for p in resource["score"]}
                scores_validation = {p["name"]: p["values"] for p in resource["scoreValidation"]}

                model = {
                    "emodel": emodel,
                    "species": species,
                    "fitness": resource["fitness"],
                    "parameters": params,
                    "scores": scores,
                    "scores_validation": scores_validation,
                    "validated": resource["passedValidation"],
                    "optimizer": resource["optimizer"],
                    "seed": resource["seed"],
                }

                models.append(model)

        return models

    def _get_mechanism(self, emodel, mechanism_name):

        resources = self.fetch(
            filters={
                "type": "SubCellularModel",
                "eModel": emodel,
                "subCellularMechanism": mechanism_name,
            }
        )

        if resources is None:
            raise Exception(
                "NexusForge error: missing mechanism %s for emodel %s", mechanism_name, emodel
            )

        if len(resources) != 1:
            raise Exception(
                "NexusForge error: more than one mechanism %s for emodel %s", mechanism_name, emodel
            )

        return resources[0]["stochastic"], resources[0]["modelScript"]["id"]

    def _get_distributions(self, emodel, distributions):

        distributions_definitions = {}

        for dist in distributions:

            resources = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureOptimisationChannelDistribution",
                    "eModel": emodel,
                    "subCellularMechanism": dist,
                }
            )

            if resources is None:
                raise Exception(
                    "NexusForge error: missing distribution %s for emodel %s", dist, emodel
                )

            if len(resources) != 1:
                raise Exception(
                    "NexusForge error: more than one distribution %s for emodel %s", dist, emodel
                )

            distributions_definitions[dist] = {
                "fun": resources[0]["function"],
                "parameters": resources[0]["parameter"],
            }

        return distributions_definitions

    def get_parameters(self, emodel, species):
        """Get the definition of the parameters to optimize from the
            optimization parameters resources, as well as the
            locations of the mechanisms. Also returns the names of the mechanisms.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

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
            filters={"type": "ElectrophysiologyFeatureOptimisationParameter", "eModel": emodel}
        )

        if resources_params is None:
            logger.warning("NexusForge warning: could not get parameters for emodel %s", emodel)
            return params_definition, mech_definition, mechanisms_names

        for resource in resources_params:

            param_def = {
                "name": resource["parameter"]["name"],
            }

            if resource["parameter"]["minValue"] == resource["parameter"]["maxValue"]:
                param_def["val"] = resource["parameter"]["minValue"]
            else:
                param_def["val"] = [
                    resource["parameter"]["minValue"],
                    resource["parameter"]["maxValue"],
                ]

            if resource["channelDistribution"] != "constant":
                param_def["dist"] = resource["channelDistribution"]
                distributions.append(resource["channelDistribution"])

            for loc in resource["locations"]:

                if loc in params_definition["parameters"]:
                    params_definition["parameters"][loc].append(param_def)
                else:
                    params_definition["parameters"][loc] = [param_def]

                if resource["subCellularMechanism"] is not None:

                    mechanisms_names.append(resource["subCellularMechanism"])

                    is_stochastic, path = self._get_mechanism(
                        emodel, resource["subCellularMechanism"]
                    )

                    if loc in mech_definition:

                        if resource["subCellularMechanism"] not in mech_definition[loc]["mech"]:
                            mech_definition[loc]["mech"].append(resource["subCellularMechanism"])
                            mech_definition[loc]["stoch"].append(is_stochastic)
                            mech_definition[loc]["path"].append(path)

                    else:
                        mech_definition[loc] = {
                            "mech": [resource["subCellularMechanism"]],
                            "stoch": [is_stochastic],
                            "path": [path],
                        }

        # TODO: Download mechanisms ?

        params_definition["distributions"] = self._get_distributions(emodel, set(distributions))

        return params_definition, mech_definition, mechanisms_names

    @staticmethod
    def _handle_extra_recordings(emodel, extra_recordings):

        extra_recordings_out = []

        for extra_recording in extra_recordings:

            if extra_recording["type"] == "somadistanceapic":
                raise Exception(
                    "extra_recording of type somadistanceapic not implemented yet for"
                    " the NexusForge API."
                )

            extra_recordings_out.append(extra_recording)

        return extra_recordings_out

    def _get_opt_targets(self, emodel, include_validation):

        resources_opt_target = self.fetch(
            filters={"type": "ElectrophysiologyFeatureOptimisationTarget", "eModel": emodel}
        )

        if resources_opt_target is None:
            logger.warning(
                "NexusForge warning: could not get optimisation targets for emodel %s", emodel
            )

        if include_validation:

            resources_val_target = self.fetch(
                filters={"type": "ElectrophysiologyFeatureValidationTarget", "eModel": emodel}
            )

            logger.warning(
                "NexusForge warning: could not get validation targets for emodel %s", emodel
            )

            return resources_opt_target + resources_val_target

        return resources_opt_target

    def get_protocols(self, emodel, species, delay=0.0, include_validation=False):
        """Get some of the protocols from the "Extracted protocols" resources depending
            on "Optimization and validation targets" ressources.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            delay (float): additional delay in ms to add at the start of
                the protocols.
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

        for resource_target in self._get_opt_targets(emodel, include_validation):

            resources_protocol = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeatureExtractionProtocol",
                    "eModel": emodel,
                    "stimulus": {
                        "stimulusType": {
                            "label": resource_target["stimulus"]["stimulusType"]["label"]
                        },
                        "stimulusTarget": resource_target["stimulus"]["target"],
                    },
                }
            )

            if resources_protocol is None:
                raise Exception(
                    "NexusForge error: could not get protocol %s %s %% for emodel %s",
                    resource_target["stimulus"]["stimulusType"]["label"],
                    resource_target["stimulus"]["target"],
                    emodel,
                )

            if len(resources_protocol) > 1:
                raise Exception(
                    "NexusForge error: more than one protocol %s %s %% for emodel %s",
                    resource_target["stimulus"]["stimulusType"]["label"],
                    resource_target["stimulus"]["target"],
                    emodel,
                )

            protocol_name = "{}_{}".format(
                resources_protocol[0]["stimulus"]["stimulusType"]["label"],
                resources_protocol[0]["stimulus"]["target"],
            )

            stimulus = resources_protocol[0]["definition"]["step"]
            stimulus["holding_current"] = resources_protocol[0]["definition"]["holding"][
                "amplitude"
            ]

            extra_recordings = self._handle_extra_recordings(
                emodel, resource_target["extraRecordings"]
            )

            protocols_out[protocol_name] = {
                "type": resource_target["protocolType"],
                "stimuli": stimulus,
                "extra_recordings": extra_recordings,
            }

        return protocols_out

    def get_features(
        self,
        emodel,
        species,
        include_validation=False,
    ):
        """Get the efeatures from the "Extracted e-features" ressources  depending
            on "Optimization and validation targets" ressources.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
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

        for resource_target in self._get_opt_targets(emodel, include_validation):

            for feature in resource_target["feature"]:

                resources_feature = self.fetch(
                    filters={
                        "type": "ElectrophysiologyFeature",
                        "eModel": emodel,
                        "feature": {"name": feature["name"]},
                        "stimulus": {
                            "stimulusType": {
                                "label": resource_target["stimulus"]["stimulusType"]["label"]
                            },
                            "stimulusTarget": resource_target["stimulus"]["target"],
                        },
                    }
                )

                if resources_feature is None:
                    raise Exception(
                        "NexusForge error: could not get feature %s for %s %s %% for emodel %s",
                        feature["name"],
                        resource_target["stimulus"]["stimulusType"]["label"],
                        resource_target["stimulus"]["target"],
                        emodel,
                    )

                if len(resources_feature) > 1:
                    raise Exception(
                        "NexusForge error: more than one feature %s for protocol %s %s %% for "
                        "emodel %s",
                        feature["name"],
                        resource_target["stimulus"]["stimulusType"]["label"],
                        resource_target["stimulus"]["target"],
                        emodel,
                    )

                feature_mean = next(
                    s["value"]
                    for s in resources_feature[0]["feature"]["series"]
                    if s["statistic"] == "mean"
                )
                feature_std = next(
                    s["value"]
                    for s in resources_feature[0]["feature"]["series"]
                    if s["statistic"] == "standard deviation"
                )

                if (
                    resource_target["protocolType"] == "RinProtocol"
                    and feature["name"] == "ohmic_input_resistance_vb_ssse"
                ):

                    efeatures_out["RinProtocol"]["soma.v"].append(
                        {
                            "feature": feature["name"],
                            "val": [feature_mean, feature_std],
                            "strict_stim": True,
                        }
                    )

                elif (
                    resource_target["protocolType"] == "RMPProtocol"
                    and feature["name"] == "steady_state_voltage_stimend"
                ):

                    efeatures_out["RMPProtocol"]["soma.v"].append(
                        {
                            "feature": feature["name"],
                            "val": [feature_mean, feature_std],
                            "strict_stim": True,
                        }
                    )

                elif (
                    resource_target["protocolType"] == "RinProtocol"
                    and feature["name"] == "voltage_base"
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
                        resource_target["stimulus"]["stimulusType"]["label"],
                        resource_target["stimulus"]["target"],
                    )

                    if protocol_name not in efeatures_out:
                        efeatures_out[protocol_name] = {"soma.v": []}

                    efeatures_out[protocol_name]["soma.v"].append(
                        {
                            "feature": feature["name"],
                            "val": [feature_mean, feature_std],
                            "strict_stim": True,
                        }
                    )

                    if feature["onsetTime"]["value"] is not None:
                        efeatures_out[protocol_name]["soma.v"][-1]["stim_start"] = feature[
                            "onsetTime"
                        ]["value"]

                    if feature["offsetTime"]["value"] is not None:
                        efeatures_out[protocol_name]["soma.v"][-1]["stim_end"] = feature[
                            "offsetTime"
                        ]["value"]

        for current in ["holding_current", "threshold_current"]:

            resources_feature = self.fetch(
                filters={
                    "type": "ElectrophysiologyFeature",
                    "eModel": emodel,
                    "feature": {"name": current},
                    "stimulus": {"stimulusType": {"label": "global"}},
                }
            )

            if resources_feature is None:
                raise Exception("NexusForge error: could not get %s for emodel %s", current, emodel)

            if len(resources_feature) > 1:
                raise Exception("NexusForge error: more than one %s for emodel %s", current, emodel)

            feature_mean = next(
                s["value"]
                for s in resources_feature[0]["feature"]["series"]
                if s["statistic"] == "mean"
            )
            feature_std = next(
                s["value"]
                for s in resources_feature[0]["feature"]["series"]
                if s["statistic"] == "standard deviation"
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

    def get_morphologies(self, emodel, species):
        """Get the name and path (or data) to the morphologies used for optimisation.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            morphology_definition (list): [{'name': morph_name, 'path': 'morph_path'}]
        """

        resources_morphology = self.fetch(
            filters={
                "type": "ElectrophysiologyFeatureOptimisationNeuronMorphology",
                "eModel": emodel,
            }
        )

        if resources_morphology is None:
            raise Exception(
                "NexusForge error: could not get morphology for optimisation of emodel %s", emodel
            )

        if len(resources_morphology) > 1:
            raise Exception(
                "NexusForge error: more than one morphology for optimisation of emodel %s", emodel
            )

        morphology_definition = [
            {
                "name": resources_morphology[0]["morphology"],
                "path": resources_morphology[0]["distribution"]["contentUrl"],
            }
        ]

        # TODO: Define path correctly and download morphology

        return morphology_definition
