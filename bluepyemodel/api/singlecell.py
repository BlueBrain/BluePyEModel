"""API to get data from Singlecell-like repositories."""

import copy
import json
import logging
from pathlib import Path

from bluepyefe.tools import NumpyEncoder

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger(__name__)

# pylint: disable=W0231,W0613

seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}


class Singlecell_API(DatabaseAPI):
    """Access point to configuration files organized as project 38."""

    def __init__(
        self,
        emodel_dir,
        final_path=None,
        recipes_path=None,
        legacy_dir_structure=False,
        extract_config=None,
    ):
        """Init

        Args:
            emodel_dir (str): path to the working directory.
            final_path (str): path to final.json which will contain the optimized models.
            recipes_path (str): path to the json file which should contain the path to the
                configuration files used for each etype. The content of this file should follow the
                format:
                {
                    "emodelname": {
                        "morph_path": "./morphologies/",
                        "morphology": [[morphologyname", "morphfile.asc"]],
                        "params": "config/params/pyr.json",
                        "protocol": "config/protocols/emodelname.json",
                        "features": "config/features/emodelname.json"
                    }
                }
            legacy_dir_structure (bool): if true, the path coming from the recipes will be replaced
                by "self.emodel_dir / emodel / json_path_from_recipes". To be deprecated.
            extract_config (str): path to a configuration json file used for feature extraction.
                It uses the old BluePyEfe 2 format. To be updated/deprecated.
        """

        self.emodel_dir = Path(emodel_dir)
        self.recipes_path = recipes_path
        self.legacy_dir_structure = legacy_dir_structure
        self.extract_config = extract_config

        if final_path is None:
            self.final_path = self.emodel_dir / "final.json"
        else:
            self.final_path = Path(final_path)

    def get_final(self):
        """Get emodels from json"""

        if self.final_path is None:
            raise Exception("Final_path is None")

        p = Path(self.final_path)
        p_tmp = p.with_name(p.stem + "_tmp" + p.suffix)

        if not (p.is_file()):
            logger.info("%s does not exist and will be create", self.final_path)
            return {}

        try:
            with open(self.final_path, "r") as f:
                final = json.load(f)
        except json.decoder.JSONDecodeError:
            try:
                with open(p_tmp, "r") as f:
                    final = json.load(f)
            except (json.decoder.JSONDecodeError, FileNotFoundError):
                logger.error("Cannot load final. final.json does not exist or is corrupted.")

        return final

    def save_final(self, final):
        """ Save final emodels in json"""

        if self.final_path is None:
            raise Exception("Final_path is None")

        p = Path(self.final_path)
        p_tmp = p.with_name(p.stem + "_tmp" + p.suffix)

        with open(self.final_path, "w+") as fp:
            json.dump(final, fp, indent=2)

        with open(p_tmp, "w+") as fp:
            json.dump(final, fp, indent=2)

    def get_recipes(self, emodel):
        """Load the recipes from a json file for an emodel.

        See docstring of __init__ for the format of the file of recipes.
        """

        if self.legacy_dir_structure:
            emodel = "_".join(emodel.split("_")[:2])
            recipes_path = self.emodel_dir / emodel / "config" / "recipes" / "recipes.json"
        else:
            recipes_path = self.recipes_path

        with open(recipes_path, "r") as f:
            return json.load(f)[emodel]

    def _get_json(self, emodel, recipe_entry):
        """Helper function to load a json using path in recipe."""

        json_path = Path(self.get_recipes(emodel)[recipe_entry])

        if self.legacy_dir_structure:
            emodel = "_".join(emodel.split("_")[:2])
            json_path = self.emodel_dir / emodel / json_path

        else:
            if not json_path.is_absolute():
                json_path = str(Path(self.emodel_dir) / json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        return data

    def get_extraction_metadata(self, emodel=None, species=None, threshold_nvalue_save=None):
        """Get the configuration parameters used for feature extraction.

        Args:
            emodel (str): name of the emodels
            species (str): name of the species (rat, human, mouse)
            threshold_nvalue_save (int): lower bounds of the number of values required to save an
                efeature.

        Returns:
            files_metadata (dict)
            targets (dict)
            protocols_threshold (list)
        """

        if self.extract_config is None or not Path(self.extract_config).is_file():
            return None, None, None

        with open(self.extract_config, "r") as f:
            config_dict = json.load(f)

        for prot in config_dict["protocols"]:

            if "targets" in config_dict["protocols"][prot]:
                config_dict["protocols"][prot]["amplitudes"] = config_dict["protocols"][prot][
                    "targets"
                ]
                config_dict["protocols"][prot].pop("targets")

        files_metadata = config_dict["cells"]
        targets = config_dict["protocols"]
        protocols_threshold = config_dict["options"].get("protocols_threshold", [])

        return files_metadata, targets, protocols_threshold

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
        """Save the efeatures and currents obtained from BluePyEfe.

        Args:
            emodel (str): name of the emodels.
            species (str): name of the species (rat, human, mouse).
            efeatures (dict): efeatures means and standard deviations. Format as returned by
                BluePyEfe 2.
            current (dict): threshold and holding current as returned by BluePyEfe. Format as
                returned by BluePyEfe 2.
            name_Rin_protocol (str): name of the protocol associated with the efeatures used for
                the computation of the input resistance scores during optimisation.
            name_rmp_protocol (str): name of the protocol associated with the efeatures used for
                the computation of the resting membrane potential scores during optimisation.
            validation_protocols (dict): names and targets of the protocol that will be used for
                validation only.
        """

        out_features = {
            "SearchHoldingCurrent": {
                "soma.v": [
                    {
                        "feature": "bpo_holding_current",
                        "val": current["holding_current"],
                        "strict_stim": True,
                    }
                ]
            },
            "SearchThresholdCurrent": {
                "soma.v": [
                    {
                        "feature": "bpo_threshold_current",
                        "val": current["threshold_current"],
                        "strict_stim": True,
                    }
                ]
            },
        }

        to_remove = {}

        for protocol in efeatures:

            # Handle a legacy case
            if "soma" in efeatures[protocol]:
                out_features[protocol] = {"soma.v": efeatures[protocol]["soma"]}
            else:
                out_features[protocol] = {"soma.v": efeatures[protocol]["soma.v"]}

            # Check if the protocol is to be used for validation
            ecode_name = str(protocol.split("_")[0])
            stimulus_target = float(protocol.split("_")[1])

            if ecode_name in validation_protocols:
                for target in validation_protocols[ecode_name]:
                    if int(target) == int(stimulus_target):
                        out_features[protocol]["validation"] = True
                        break

            if "validation" not in out_features[protocol]:
                out_features[protocol]["validation"] = False

            # Handle the features related to RMP, Rin and threshold and holding current.
            to_remove = {protocol: []}
            for i, efeat in enumerate(out_features[protocol]["soma.v"]):

                if protocol == name_rmp_protocol and efeat["feature"] == "voltage_base":
                    out_features["RMPProtocol"] = {
                        "soma.v": [
                            {
                                "feature": "steady_state_voltage_stimend",
                                "val": efeat["val"],
                                "strict_stim": True,
                            }
                        ]
                    }
                    to_remove[protocol].append(i)

                elif protocol == name_Rin_protocol and efeat["feature"] == "voltage_base":
                    out_features["SearchHoldingCurrent"]["soma.v"].append(
                        {
                            "feature": "steady_state_voltage_stimend",
                            "val": efeat["val"],
                            "strict_stim": True,
                        }
                    )
                    to_remove[protocol].append(i)

                elif (
                    protocol == name_Rin_protocol
                    and efeat["feature"] == "ohmic_input_resistance_vb_ssse"
                ):
                    out_features["RinProtocol"] = {"soma.v": [copy.copy(efeat)]}
                    to_remove[protocol].append(i)

        # If some features are part of the RMP, Rin and threshold and holding current protocols,
        # they should be removed from their original protocol.
        for protocol in to_remove:
            if to_remove[protocol]:
                out_features[protocol]["soma.v"] = [
                    f
                    for i, f in enumerate(out_features[protocol]["soma.v"])
                    if i not in to_remove[protocol]
                ]

            if not (out_features[protocol]["soma.v"]):
                out_features.pop(protocol)

        features_path = self.get_recipes(emodel)["features"]
        features_path.parent.mkdir(parents=True, exist_ok=True)

        s = json.dumps(out_features, indent=2, cls=NumpyEncoder)
        with open(features_path, "w") as f:
            f.write(s)

    def store_protocols(self, emodel, species, stimuli, validation_protocols):
        """Save the protocols obtained from BluePyEfe.

        Args:
            emodel (str): name of the emodel.
            species (str): name of the species (rat, human, mouse).
            stimuli (dict): protocols definition in the format returned by BluePyEfe 2.
            validation_protocols (dict): names and targets of the protocol that will be used for
                validation only.
        """

        for stimulus_name, stimulus in stimuli.items():

            stimulus["type"] = "StepThresholdProtocol"

            # Check if the protocol is to be used for validation
            ecode_name = str(stimulus_name.split("_")[0])
            stimulus_target = float(stimulus_name.split("_")[1])

            if ecode_name in validation_protocols:
                for target in validation_protocols[ecode_name]:
                    if int(target) == int(stimulus_target):
                        stimulus["validation"] = True
                        break

            if "validation" not in stimulus:
                stimulus["validation"] = False

        protocols_path = self.get_recipes(emodel)["protocol"]
        protocols_path.parent.mkdir(parents=True, exist_ok=True)

        s = json.dumps(stimuli, indent=2, cls=NumpyEncoder)
        with open(protocols_path, "w") as f:
            f.write(s)

    @staticmethod
    def get_model_name_for_final(emodel, githash, seed):
        """ Return model name used as key in final.json. """

        if githash:
            return "{}__{}__{}".format(emodel, githash, seed)

        logger.warning(
            "Githash is %s. It is strongly advised to use githash in the future.",
            githash,
        )

        return "{}__{}".format(emodel, seed)

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
        """Store an emodel obtained from BluePyOpt in the final.json. Note that if a model in the
        final.json has the same key (emodel__githash__seed), it will be overwritten.

        Args:
            emodel (str): name of the emodel.
            scores (dict): of the format {"objective_name": score}.
            params (dict): of the format {"param_name": param_value}.
            optimizer_name (str): name of the optimizer (IBEA, CMA, ...).
            seed (int): seed used by the optimizer.
            githash (str): githash associated with the configuration files.
            validated (bool): has the model been through validation.
            scores_validation (dict): of the format {"objective_name": score}.
            species (str): name of the species (rat, human, mouse).
        """

        if scores_validation is None:
            scores_validation = {}

        final = self.get_final()

        entry = {
            "emodel": emodel,
            "species": species,
            "score": sum(list(scores.values())),
            "params": params,
            "fitness": scores,
            "validation_fitness": scores_validation,
            "validated": validated,
            "optimizer": str(optimizer_name),
            "seed": int(seed),
            "githash": str(githash),
        }

        model_name = self.get_model_name_for_final(emodel, githash, seed)

        if model_name in final:
            logger.warning(
                "Entry %s was already in the final.json and will be overwritten", model_name
            )

        final[model_name] = entry

        self.save_final(final)

    def get_parameters(self, emodel, species=None):
        """Get the definition of the parameters that have to be optimized as well as the
         locations of the mechanisms. Also returns the name of the mechanisms.

        Args:
            emodel (str): name of the emodel.
            species (str): name of the species (rat, human, mouse).

        Returns:
            params_definition (dict)
            mech_definition (dict)
            mech_names (list)
        """

        parameters = self._get_json(emodel, "params")

        params_definition = {
            "distributions": parameters["distributions"],
            "parameters": parameters["parameters"],
        }

        if "__comment" in params_definition["parameters"]:
            params_definition["parameters"].pop("__comment")

        mech_definition = parameters["mechanisms"]
        mech_names = []

        for mechanisms in mech_definition.values():

            mech_names += mechanisms["mech"]

            stochastic = []

            for mech_name in mechanisms["mech"]:

                if "Stoch" in mech_name:
                    stochastic.append(True)
                else:
                    stochastic.append(False)

            mechanisms["stoch"] = stochastic

        return params_definition, mech_definition, list(set(mech_names))

    def _handle_extra_recording(self, emodel, extra_recordings):
        """ Fetch the information needed to be able to use the extra recordings. """

        extra_recordings_out = []

        for extra in extra_recordings:

            if extra["type"] == "somadistanceapic":

                morphologies = self.get_morphologies(emodel)
                morph_name = morphologies[0]["name"]
                p = self.emodel_dir / "apical_points_isec.json"

                if p.exists():

                    extra["sec_index"] = json.load(open(str(p)))[morph_name]

                    if extra["seclist_name"]:
                        extra["sec_name"] = seclist_to_sec[extra["seclist_name"]]
                    else:
                        raise Exception("Cannot get section name from seclist_name.")

                else:
                    raise Exception(
                        "No apical_points_isec.json found for extra_recordings of type "
                        "somadistanceapic."
                    )

            extra_recordings_out.append(extra)

        return extra_recordings_out

    def _read_protocol(self, emodel, protocol):

        stimulus_def = protocol["step"]
        stimulus_def["holding_current"] = protocol["holding"]["amp"]

        protocol_definition = {"type": protocol["type"], "stimuli": stimulus_def}

        if "extra_recordings" in protocol:
            protocol_definition["extra_recordings"] = self._handle_extra_recording(
                emodel, protocol["extra_recordings"]
            )

        return protocol_definition

    def _read_legacy_protocol(self, emodel, protocol, protocol_name):

        if protocol_name in ["RMP", "Rin", "RinHoldcurrent", "Main", "ThresholdDetection"]:
            return None

        if protocol["type"] not in ["StepThresholdProtocol", "StepProtocol"]:
            return None

        stimulus_def = protocol["stimuli"]["step"]
        if "holding" in protocol["stimuli"]:
            stimulus_def["holding_current"] = protocol["stimuli"]["holding"]["amp"]
        else:
            stimulus_def["holding_current"] = None

        protocol_definition = {"type": protocol["type"], "stimuli": stimulus_def}

        if "extra_recordings" in protocol:
            protocol_definition["extra_recordings"] = self._handle_extra_recording(
                emodel, protocol["extra_recordings"]
            )

        return protocol_definition

    def get_protocols(self, emodel, species=None, delay=0.0, include_validation=False):
        """Get the protocols from the configuration file and put them in the format required by
        MainProtocol.

        Args:
            emodel (str): name of the emodel.
            species (str): name of the species (rat, human, mouse).
            delay (float): additional delay in ms to add at the start of the protocols.
            include_validation (bool): if True, returns the protocols for validation as well as
                the ones for optimisation.

        Returns:
            protocols_out (dict): protocols definitions
        """

        protocols = self._get_json(emodel, "protocol")

        protocols_out = {}

        for protocol_name, protocol in protocols.items():

            if "validation" in protocol:
                if not include_validation and protocol["validation"]:
                    continue
                protocol_definition = self._read_protocol(emodel, protocol)
            else:
                protocol_definition = self._read_legacy_protocol(emodel, protocol, protocol_name)

            if protocol_definition:
                protocols_out[protocol_name] = protocol_definition

        return protocols_out

    def get_name_validation_protocols(self, emodel, species):
        """Get the names of the protocols used for validation """
        protocols = self._get_json(emodel, "protocol")

        names = []
        for prot_name, prot in protocols.items():

            if "validation" in prot and prot["validation"]:
                names.append(prot_name)

        return names

    def get_features(
        self,
        emodel,
        species=None,
        include_validation=False,
    ):
        """Get the efeatures from the configuration files and put then in the format required by
        MainProtocol.

        Args:
            emodel (str): name of the emodel.
            species (str): name of the species (rat, human, mouse).
            include_validation (bool): should the features for validation be returned as well as
                the ones for optimisation.

        Returns:
            efeatures_out (dict): efeatures definitions
        """

        efeatures = self._get_json(emodel, "features")
        protocols = self._get_json(emodel, "protocol")

        efeatures_out = {
            "RMPProtocol": {"soma.v": []},
            "RinProtocol": {"soma.v": []},
            "SearchHoldingCurrent": {"soma.v": []},
            "SearchThresholdCurrent": {"soma.v": []},
        }

        for prot_name in efeatures:

            if (
                "validation" in efeatures[prot_name]
                and not include_validation
                and efeatures[prot_name]["validation"]
            ):
                continue

            for loc in efeatures[prot_name]:

                if loc == "validation":
                    continue

                for efeat in efeatures[prot_name][loc]:

                    if "bAP" in prot_name:
                        efeat["stim_start"] = protocols[prot_name]["stimuli"]["step"]["delay"]
                        efeat["stim_end"] = protocols[prot_name]["stimuli"]["step"]["totduration"]

                    if prot_name == "Rin" and efeat["feature"] == "ohmic_input_resistance_vb_ssse":
                        efeatures_out["RinProtocol"]["soma.v"].append(efeat)

                    elif prot_name == "RMP" and efeat["feature"] == "voltage_base":
                        efeat["feature"] = "steady_state_voltage_stimend"
                        efeatures_out["RMPProtocol"]["soma.v"].append(efeat)

                    elif (
                        prot_name == "RinHoldCurrent" and efeat["feature"] == "bpo_holding_current"
                    ):
                        efeatures_out["SearchHoldingCurrent"]["soma.v"].append(efeat)

                    elif prot_name == "Rin" and efeat["feature"] == "voltage_base":
                        efeat["feature"] = "steady_state_voltage_stimend"
                        efeatures_out["SearchHoldingCurrent"]["soma.v"].append(efeat)

                    elif prot_name == "Threshold" and efeat["feature"] == "bpo_threshold_current":
                        efeatures_out["SearchThresholdCurrent"]["soma.v"].append(efeat)

                    else:
                        if prot_name not in efeatures_out:
                            efeatures_out[prot_name] = {loc: []}
                        if loc not in efeatures_out[prot_name]:
                            efeatures_out[prot_name][loc] = []
                        efeatures_out[prot_name][loc].append(efeat)

        return efeatures_out

    def get_morphologies(self, emodel, species=None):
        """Get the name and path to the morphologies from the recipes.

        Args:
            emodel (str): name of the emodel.
            species (str): name of the species (rat, human, mouse).

        Returns:
            morphology_definition (list): [{'name': morph_name, 'path': 'morph_path'}]. Might
            contain the additional entries "seclist_names" and "secarray_names" if they are
            present in the recipes.
        """

        recipes = self.get_recipes(emodel)

        morphology_definition = []

        for morph_def in recipes["morphology"]:

            morph_path = Path(recipes["morph_path"]) / morph_def[1]

            if not morph_path.is_absolute():
                morph_path = str(Path(self.emodel_dir) / morph_path)

            morphology_definition.append({"name": morph_def[1][:-4], "path": str(morph_path)})

            if "seclist_names" in recipes:
                morphology_definition[-1]["seclist_names"] = recipes["seclist_names"]

            if "secarray_names" in recipes:
                morphology_definition[-1]["secarray_names"] = recipes["secarray_names"]

        return morphology_definition

    def format_emodel_data(self, model_data, species=None):

        out_data = {
            "emodel": model_data["emodel"],
            "fitness": model_data["score"],
            "parameters": model_data["params"],
            "scores": model_data["fitness"],
        }

        for key in [
            "seed",
            "githash",
            "branch",
            "rank",
            "optimizer",
            "species",
        ]:
            if key in model_data:
                out_data[key] = model_data[key]

        if "validation_fitness" in model_data:
            out_data["scores_validation"] = model_data["validation_fitness"]
            out_data["validated"] = True
        else:
            out_data["scores_validation"] = {}
            out_data["validated"] = False

        return out_data

    def get_emodel(self, emodel, species=None):
        """Get dict with parameter of single emodel (including seed if any)

        WARNING: this search is based on the name of the model and not its
        emodel despite the name of the variable. To search by emode name
        use get_emodels.

        Args:
            emodel (str): name of the emodels.
            species (str): name of the species (rat, human, mouse).
        """

        final = self.get_final()

        if emodel in final:
            return self.format_emodel_data(final[emodel])

        logger.warning("Could not find models for emodel %s", emodel)

        return None

    def get_emodels(self, emodels, species):
        """Get the list of emodels dictionaries.

        Args:
            emodels (list): list of names of the emodels.
            species (str): name of the species (rat, human, mouse).
        """
        models = []
        for mod_data in self.get_final().values():
            if mod_data["emodel"] in emodels:
                models.append(self.format_emodel_data(mod_data))

        return models

    def get_emodel_etype_map(self):
        final = self.get_final()
        return {emodel: emodel.split("_")[0] for emodel in final}

    def get_emodel_names(self):
        """Get the list of all the names of emodels

        Returns:
            dict: keys are emodel names with seed, values are names without seed.
        """

        final = self.get_final()

        return {mod_name: mod["emodel"] for mod_name, mod in final.items()}

    def has_protocols_and_features(self, emodel, species=None):
        """Check if the efeatures and protocol exist."""

        features_exists = (
            Path(self.emodel_dir, "config", "features", emodel).with_suffix(".json").is_file()
        )
        protocols_exists = (
            Path(self.emodel_dir, "config", "protocols", emodel).with_suffix(".json").is_file()
        )

        return features_exists and protocols_exists

    def optimisation_state(self, emodel, checkpoint_dir, species=None, seed=1, githash=""):
        """Return the state of the optimisation.

        TODO: - should return three states: completed, in progress, empty
              - better management of checkpoints
        """

        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{emodel}_{githash}_{seed}.pkl"

        return checkpoint_path.is_file()

    def has_best_model(self, emodel, seed, githash):
        """ Check if the best model has been stored. """

        final = self.get_final()

        model_name = self.get_model_name_for_final(emodel, githash, seed)

        return model_name in final

    def is_checked_by_validation(self, emodel, seed, githash):
        """ Check if the emodel with a given seed has been checked by Validation task. """

        final = self.get_final()

        model_name = self.get_model_name_for_final(emodel, githash, seed)

        return bool(final.get(model_name, {}).get("validation_fitness"))

    def is_validated(self, emodel, githash, n_models_to_pass_validation):
        """ Check if enough models have been validated. """

        n_validated = 0
        final = self.get_final()

        for _, entry in final.items():

            if entry["githash"] == githash and entry["emodel"] == emodel and entry["validated"]:
                n_validated += 1

        return n_validated >= n_models_to_pass_validation
