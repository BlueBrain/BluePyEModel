"""API to get data from Singlecell-like repositories."""
import copy
import json
import logging
from collections import defaultdict
from pathlib import Path

import fasteners
from bluepyefe.tools import NumpyEncoder

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger(__name__)

seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}


class SinglecellAPI(DatabaseAPI):
    """Access point to configuration files organized as project 38."""

    def __init__(
        self,
        emodel,
        emodel_dir,
        final_path=None,
        recipes_path=None,
        legacy_dir_structure=False,
        extract_config=None,
    ):
        """Init

        Args:
            emodel (str): name of the emodel.
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

        super().__init__(emodel)

        self.emodel_dir = Path(emodel_dir)
        self.recipes_path = recipes_path
        self.legacy_dir_structure = legacy_dir_structure
        self.extract_config = extract_config

        self.morph_path = None

        if final_path is None:
            self.final_path = self.emodel_dir / "final.json"
        else:
            self.final_path = Path(final_path)

        Path(".tmp/").mkdir(exist_ok=True)
        self.rw_lock_final = fasteners.InterProcessReaderWriterLock(".tmp/final.lock")
        self.rw_lock_final_tmp = fasteners.InterProcessReaderWriterLock(".tmp/final_tmp.lock")

    def set_emodel(self, emodel):
        """Setter for the name of the emodel."""
        if emodel not in self.get_all_recipes():
            raise Exception("Cannot set the emodel name to %s which does not exist in the recipes.")

        self.emodel = emodel

    def get_final(self, lock_file=True):
        """Get emodel dictionary from final.json."""
        if self.final_path is None:
            raise Exception("Final_path is None")

        if not self.final_path.is_file():
            logger.info("%s does not exist and will be created", self.final_path)
            return {}

        try:
            if lock_file:
                self.rw_lock_final.acquire_read_lock()

            with open(self.final_path, "r") as f:
                final = json.load(f)

            if lock_file:
                self.rw_lock_final.release_read_lock()
        except json.decoder.JSONDecodeError:
            try:
                if lock_file:
                    self.rw_lock_final_tmp.acquire_read_lock()
                _tmp_final_path = self.final_path.with_name(
                    self.final_path.stem + "_tmp" + self.final_path.suffix
                )
                with open(_tmp_final_path, "r") as f:
                    final = json.load(f)

                if lock_file:
                    self.rw_lock_final_tmp.release_read_lock()
            except (json.decoder.JSONDecodeError, FileNotFoundError):
                logger.error("Cannot load final. final.json does not exist or is corrupted.")

        return final

    def save_final(self, final, lock_file=True):
        """Save final emodels in json"""

        if self.final_path is None:
            raise Exception("Final_path is None")

        if lock_file:
            self.rw_lock_final.acquire_write_lock()

        with open(self.final_path, "w+") as fp:
            json.dump(final, fp, indent=2)

        if lock_file:
            self.rw_lock_final.release_write_lock()
            self.rw_lock_final_tmp.acquire_write_lock()

        with open(self.final_path.with_name(self.final_path.stem + "_tmp.json"), "w+") as fp:
            json.dump(final, fp, indent=2)

        if lock_file:
            self.rw_lock_final_tmp.release_write_lock()

    def get_all_recipes(self):
        """Load the recipes from a json file.

        See docstring of __init__ for the format of the file of recipes.
        """

        if self.legacy_dir_structure:
            emodel = "_".join(self.emodel.split("_")[:2])
            recipes_path = self.emodel_dir / emodel / "config" / "recipes" / "recipes.json"
        else:
            recipes_path = self.recipes_path

        with open(recipes_path, "r") as f:
            return json.load(f)

    def get_recipes(self):
        """Load the recipes from a json file for an emodel.

        See docstring of __init__ for the format of the file of recipes.
        """

        return self.get_all_recipes()[self.emodel]

    def get_morph_modifiers(self):
        """Get the morph modifiers if any."""
        return self.get_recipes().get("morph_modifiers", None)

    def _get_json(self, recipe_entry):
        """Helper function to load a json using path in recipe."""

        json_path = Path(self.get_recipes()[recipe_entry])

        if self.legacy_dir_structure:
            emodel = "_".join(self.emodel.split("_")[:2])
            json_path = self.emodel_dir / emodel / json_path
        elif not json_path.is_absolute():
            json_path = self.emodel_dir / json_path

        with open(json_path, "r") as f:
            return json.load(f)

    def get_extraction_metadata(self):
        """Get the configuration parameters used for feature extraction.

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
        efeatures,
        current,
        name_Rin_protocol,
        name_rmp_protocol,
        validation_protocols,
    ):
        """Save the efeatures and currents obtained from BluePyEfe.

        Args:
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
            for efeat in out_features[protocol]["soma.v"]:
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

                elif protocol == name_Rin_protocol and efeat["feature"] == "voltage_base":
                    out_features["SearchHoldingCurrent"]["soma.v"].append(
                        {
                            "feature": "steady_state_voltage_stimend",
                            "val": efeat["val"],
                            "strict_stim": True,
                        }
                    )

                elif (
                    protocol == name_Rin_protocol
                    and efeat["feature"] == "ohmic_input_resistance_vb_ssse"
                ):
                    out_features["RinProtocol"] = {"soma.v": [copy.copy(efeat)]}

        # If some features are part of the RMP, Rin and threshold and holding current protocols,
        # they should be removed from their original protocol.
        # TODO: to rework when we will have the possibility of extracting different efeatures for
        # the different ampliudes of the same eCode.
        out_features.pop(name_Rin_protocol)
        out_features.pop(name_rmp_protocol)

        features_path = Path(self.get_recipes()["features"])
        features_path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(features_path), "w") as f:
            f.write(json.dumps(out_features, indent=2, cls=NumpyEncoder))

    def store_protocols(self, stimuli, validation_protocols):
        """Save the protocols obtained from BluePyEfe.

        Args:
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

        protocols_path = Path(self.get_recipes()["protocol"])
        protocols_path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(protocols_path), "w") as f:
            f.write(json.dumps(stimuli, indent=2, cls=NumpyEncoder))

    def get_model_name_for_final(self, githash, seed):
        """Return model name used as key in final.json."""

        if githash:
            return "{}__{}__{}".format(self.emodel, githash, seed)

        logger.warning(
            "Githash is %s. It is strongly advised to use githash in the future.",
            githash,
        )

        return "{}__{}".format(self.emodel, seed)

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
        """Store an emodel obtained from BluePyOpt in the final.json. Note that if a model in the
        final.json has the same key (emodel__githash__seed), it will be overwritten.

        Args:
            scores (dict): scores of the efeatures. Of the format {"objective_name": score}.
            params (dict): values of the parameters. Of the format {"param_name": param_value}.
            optimizer_name (str): name of the optimizer (IBEA, CMA, ...).
            seed (int): seed used by the optimizer.
            githash (str): githash associated with the configuration files.
            validated (bool): None indicate that the model did not go through validation.
                False indicates that it failed validation. True indicates that it
                passed validation.
            scores_validation (dict): scores of the validation efeatures. Of the format
                {"objective_name": score}.
        """

        if scores_validation is None:
            scores_validation = {}

        with self.rw_lock_final.write_lock():
            with self.rw_lock_final_tmp.write_lock():

                final = self.get_final(lock_file=False)
                model_name = self.get_model_name_for_final(githash, seed)

                if model_name in final:
                    logger.warning(
                        "Entry %s was already in the final.json and will be overwritten", model_name
                    )

                final[model_name] = {
                    "emodel": self.emodel,
                    "score": sum(list(scores.values())),
                    "params": params,
                    "fitness": scores,
                    "validation_fitness": scores_validation,
                    "validated": validated,
                    "optimizer": str(optimizer_name),
                    "seed": int(seed),
                    "githash": str(githash),
                }

                self.save_final(final, lock_file=False)

    def get_parameters(self):
        """Get the definition of the parameters that have to be optimized as well as the
         locations of the mechanisms. Also returns the name of the mechanisms.

        Returns:
            params_definition (dict)
            mech_definition (dict)
            mech_names (list)
        """

        parameters = self._get_json("params")

        params_definition = {
            "distributions": parameters["distributions"],
            "parameters": parameters["parameters"],
        }

        params_definition["parameters"].pop("__comment", None)

        mech_definition = parameters["mechanisms"]
        mech_names = []

        for mechanisms in mech_definition.values():
            if "mech" in mechanisms:
                mech_names += mechanisms["mech"]
                mechanisms["stoch"] = ["Stoch" in mech_name for mech_name in mechanisms["mech"]]

        params_definition["multiloc_map"] = self.get_recipes().get("multiloc_map", None)
        mech_definition["multiloc_map"] = self.get_recipes().get("multiloc_map", None)
        return params_definition, mech_definition, list(set(mech_names))

    def _read_protocol(self, protocol):
        stimulus_def = protocol["step"]
        stimulus_def["holding_current"] = protocol["holding"]["amp"]

        protocol_definition = {"type": protocol["type"], "stimuli": stimulus_def}

        if "extra_recordings" in protocol:
            protocol_definition["extra_recordings"] = protocol["extra_recordings"]

        return protocol_definition

    def _read_legacy_protocol(self, protocol, protocol_name):
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
            protocol_definition["extra_recordings"] = protocol["extra_recordings"]

        return protocol_definition

    def get_protocols(self, include_validation=False):
        """Get the protocols from the configuration file and put them in the format required by
        MainProtocol.

        Args:
            include_validation (bool): if True, returns the protocols for validation as well as
                the ones for optimisation.

        Returns:
            protocols_out (dict): protocols definitions
        """
        protocols = self._get_json("protocol")

        protocols_out = {}
        for protocol_name, protocol in protocols.items():

            if "validation" in protocol:
                if not include_validation and protocol["validation"]:
                    continue
                protocol_definition = self._read_protocol(protocol)
            else:
                protocol_definition = self._read_legacy_protocol(protocol, protocol_name)

            if protocol_definition:
                protocols_out[protocol_name] = protocol_definition

        return protocols_out

    def get_name_validation_protocols(self):
        """Get the names of the protocols used for validation"""

        protocols = self._get_json("protocol")

        names = []
        for prot_name, prot in protocols.items():

            if "validation" in prot and prot["validation"]:
                names.append(prot_name)

        return names

    def get_features(self, include_validation=False):
        """Get the features from the configuration files and put then in the format required by
        MainProtocol.

        Args:
            include_validation (bool): should the features for validation be returned as well as
                the ones for optimisation.

        Returns:
            features_out (dict): features definitions
        """

        features = self._get_json("features")

        features_out = {
            "RMPProtocol": {"soma.v": []},
            "RinProtocol": {"soma.v": []},
            "SearchHoldingCurrent": {"soma.v": []},
            "SearchThresholdCurrent": {"soma.v": []},
        }
        # pylint: disable=too-many-nested-blocks
        for prot_name in features:

            if (
                "validation" in features[prot_name]
                and not include_validation
                and features[prot_name]["validation"]
            ):
                continue

            for loc in features[prot_name]:

                if loc == "validation":
                    continue

                for feat in features[prot_name][loc]:

                    if prot_name == "Rin" and feat["feature"] == "ohmic_input_resistance_vb_ssse":
                        features_out["RinProtocol"]["soma.v"].append(feat)

                    elif prot_name == "RMP" and feat["feature"] == "voltage_base":
                        feat["feature"] = "steady_state_voltage_stimend"
                        features_out["RMPProtocol"]["soma.v"].append(feat)

                    elif prot_name == "RinHoldCurrent" and feat["feature"] == "bpo_holding_current":
                        features_out["SearchHoldingCurrent"]["soma.v"].append(feat)

                    elif prot_name == "Rin" and feat["feature"] == "voltage_base":
                        feat["feature"] = "steady_state_voltage_stimend"
                        features_out["SearchHoldingCurrent"]["soma.v"].append(feat)

                    elif prot_name == "Threshold" and feat["feature"] == "bpo_threshold_current":
                        features_out["SearchThresholdCurrent"]["soma.v"].append(feat)

                    else:
                        if prot_name not in features_out:
                            features_out[prot_name] = defaultdict(list)
                        features_out[prot_name][loc].append(feat)

        return features_out

    def get_morphologies(self):
        """Get the name and path to the morphologies from the recipes.

        Returns:
            morphology_definition (list): [{'name': morph_name, 'path': 'morph_path'}]. Might
            contain the additional entries "seclist_names" and "secarray_names" if they are
            present in the recipes.
        """

        recipes = self.get_recipes()

        morphology_definition = []

        morph_def = recipes["morphology"][0]

        if self.morph_path is None:
            self.morph_path = Path(recipes["morph_path"]) / morph_def[1]
            if not self.morph_path.is_absolute():
                self.morph_path = Path(self.emodel_dir) / self.morph_path
        else:
            self.morph_path = Path(self.morph_path)

        morphology_definition = {"name": self.morph_path.stem, "path": str(self.morph_path)}
        if "seclist_names" in recipes:
            morphology_definition["seclist_names"] = recipes["seclist_names"]

        if "secarray_names" in recipes:
            morphology_definition["secarray_names"] = recipes["secarray_names"]
        return morphology_definition

    def format_emodel_data(self, model_data):
        """Format emodel data."""
        out_data = {
            "emodel": self.emodel,
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
        ]:
            if key in model_data:
                out_data[key] = model_data[key]

        if "validation_fitness" in model_data:
            out_data["scores_validation"] = model_data["validation_fitness"]
            out_data["validated"] = True
        else:
            out_data["scores_validation"] = {}
            out_data["validated"] = None

        return out_data

    def get_emodel(self):
        """Get dict with parameter of single emodel (including seed if any)

        WARNING: this search is based on the name of the model and not the name of the emodel
        despite the name of the variable. To search by emodel name use get_emodels.

        Args:
            emodel (str): name of the emodels.
        """

        final = self.get_final()
        if self.emodel in final:
            return self.format_emodel_data(final[self.emodel])

        logger.warning("Could not find models for emodel %s", self.emodel)

        return None

    def get_emodels(self, emodels=None):
        """Get the list of emodels dictionaries.

        Args:
            emodels (list): list of names of the emodels.
        """

        if emodels is None:
            emodels = [self.emodel]

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

        return {mod_name: mod.get("emodel", mod_name) for mod_name, mod in final.items()}

    def has_protocols_and_features(self):
        """Check if the efeatures and protocol exist."""

        # TODO: Re-write this to use recipes instead of hardcoded path
        features_exists = (
            Path(self.emodel_dir, "config", "features", self.emodel).with_suffix(".json").is_file()
        )
        protocols_exists = (
            Path(self.emodel_dir, "config", "protocols", self.emodel).with_suffix(".json").is_file()
        )

        return features_exists and protocols_exists

    def has_best_model(self, seed, githash):
        """Check if the best model has been stored."""

        final = self.get_final()

        model_name = self.get_model_name_for_final(githash, seed)

        return model_name in final

    def is_checked_by_validation(self, seed, githash):
        """Check if the emodel with a given seed has been checked by Validation task."""

        final = self.get_final()

        model_name = self.get_model_name_for_final(githash, seed)

        model = final.get(model_name, {})
        if "validated" in model and model["validated"] is not None:
            return True

        return False

    def is_validated(self, githash, n_models_to_pass_validation):
        """Check if enough models have been validated."""

        n_validated = 0
        final = self.get_final()

        for _, entry in final.items():

            if (
                entry["githash"] == githash
                and entry["emodel"] == self.emodel
                and entry["validated"]
            ):
                n_validated += 1

        return n_validated >= n_models_to_pass_validation
