"""API to get data from Singlecell-like repositories."""
import copy
import json
import logging
from pathlib import Path

from bluepyefe.tools import NumpyEncoder

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger(__name__)

seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}

# pylint: disable=W0231,W0221,W0613


class Singlecell_API(DatabaseAPI):
    """API to reproduce Singlecell repositories."""

    def __init__(
        self,
        emodel_dir,
        final_path=None,
        recipes_path=None,
        legacy_dir_structure=False,
        extract_config=None,
    ):
        """
        Args:
            emodel_dir (str): path to the emodel directory
            final_path (str): path to final.json
            recipes_path (str): path to the recipes.json
        """
        self.emodel_dir = Path(emodel_dir)
        self.recipes_path = recipes_path
        self.legacy_dir_structure = legacy_dir_structure
        self.extract_config = extract_config

        if final_path is None:
            self.final_path = self.emodel_dir / "final.json"
        else:
            self.final_path = Path(final_path)

        self.fix_final_compatibility()
        self._seeds = {}

    def set_seed(self, emodel, seed, species=None):
        """Set the seed of an emodel"""
        if emodel in self._seeds:
            raise Exception("Seed already set for this emodel.")
        self._seeds[emodel] = seed

    def get_seed(self, emodel, seed, species=None):
        """Get the seed of an emodel"""
        return self._seeds["emodel"]

    def get_final(self):
        """Get emodels from json"""
        if self.final_path is None:
            raise Exception("Final_path is None")

        p = Path(self.final_path)
        if not (p.is_file()):
            logger.info("%s does not exist and will be create", self.final_path)
            return {}

        with open(self.final_path, "r") as f:
            final = json.load(f)

        return final

    def save_final(self, final):
        """ Save final emodels in json"""
        if self.final_path is None:
            raise Exception("Final_path is None")

        with open(self.final_path, "w+") as fp:
            json.dump(final, fp, indent=2)

    def fix_final_compatibility(self):
        """ Ensures that the base emodel name exists in each entry (for compatibility)"""
        if self.final_path.exists():

            final = self.get_final()

            for emodel in final:
                if "emodel" not in final[emodel]:

                    splitted = emodel.split("_")

                    if splitted[-1].isnumeric():
                        seed = int(splitted[-1])
                        emodel = "_".join(emodel.split("_")[:-1])
                    else:
                        seed = None

                    if seed and "seed" not in final[emodel]:
                        final[emodel]["seed"] = seed

                    final[emodel]["emodel"] = str(emodel)

            self.save_final(final)

        else:
            logger.warning("%s does not exist.", self.final_path)

    def _get_json(self, emodel, recipe_entry):
        """Helper function to load a  json using path in recipe."""

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

    def get_extraction_metadata(
        self, emodel=None, species=None, threshold_nvalue_save=None
    ):  # pylint: disable=unused-argument
        """Get config for feature extraction.

        Args:
            emodel (str): name of the emodels
            species (str): name of the species (rat, human, mouse)
            threshold_nvalue_save (int): lower bounds of the number of values required
                to save an efeature.

        Returns:
            config_dict (dict): config dict used in extract.extract_efeatures()
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
        """ Save the efeatures and currents obtained from BluePyEfe"""

        out_features = {}

        out_features["SearchHoldingCurrent"] = {
            "soma.v": [
                {
                    "feature": "bpo_holding_current",
                    "val": current["holding_current"],
                    "strict_stim": True,
                }
            ]
        }

        out_features["SearchThresholdCurrent"] = {
            "soma.v": [
                {
                    "feature": "bpo_threshold_current",
                    "val": current["threshold_current"],
                    "strict_stim": True,
                }
            ]
        }

        to_remove = {}

        for prot in efeatures:

            if "soma" in efeatures[prot]:
                out_features[prot] = {"soma.v": efeatures[prot]["soma"]}
            else:
                out_features[prot] = {"soma.v": efeatures[prot]["soma.v"]}

            prot_name = str(prot.split("_")[0])
            prot_target = float(prot.split("_")[1])

            for v, targets in validation_protocols.items():
                for target in targets:
                    if v.lower() in prot_name.lower() and float(target) == prot_target:
                        out_features[prot]["validation"] = True
                        break
            if "validation" not in out_features[prot]:
                out_features[prot]["validation"] = False

            to_remove = {prot: []}
            for i, efeat in enumerate(out_features[prot]["soma.v"]):

                if prot == name_rmp_protocol and efeat["feature"] == "voltage_base":
                    out_features["RMPProtocol"] = {
                        "soma.v": [
                            {
                                "feature": "steady_state_voltage_stimend",
                                "val": efeat["val"],
                                "strict_stim": True,
                            }
                        ]
                    }
                    to_remove[prot].append(i)

                if prot == name_Rin_protocol and efeat["feature"] == "voltage_base":
                    out_features["SearchHoldingCurrent"]["soma.v"].append(
                        {
                            "feature": "steady_state_voltage_stimend",
                            "val": efeat["val"],
                            "strict_stim": True,
                        }
                    )
                    to_remove[prot].append(i)

                if (
                    prot == name_Rin_protocol
                    and efeat["feature"] == "ohmic_input_resistance_vb_ssse"
                ):
                    out_features["RinProtocol"] = {"soma.v": [copy.copy(efeat)]}
                    to_remove[prot].append(i)

        for prot in to_remove:
            if to_remove[prot]:
                out_features[prot]["soma.v"] = [
                    f
                    for i, f in enumerate(out_features[prot]["soma.v"])
                    if i not in to_remove[prot]
                ]

            if not (out_features[prot]["soma.v"]):
                out_features.pop(prot)

        file_name = emodel + ".json"
        features_path = self.emodel_dir / "config" / "features" / file_name

        # create directory if needed.
        features_path.parent.mkdir(parents=True, exist_ok=True)

        s = json.dumps(out_features, indent=2, cls=NumpyEncoder)
        with open(features_path, "w") as f:
            f.write(s)

    def store_protocols(self, emodel, species, stimuli, validation_protocols):
        """ Save the protocols obtained from BluePyEfe"""

        for stim in stimuli:
            stimuli[stim]["type"] = "StepThresholdProtocol"

            prot_name = str(stim.split("_")[0])
            prot_target = float(stim.split("_")[1])

            for v, targets in validation_protocols.items():
                for target in targets:
                    if v.lower() in prot_name.lower() and int(target) == int(prot_target):
                        stimuli[stim]["validation"] = True
                        break

            if "validation" not in stimuli[stim]:
                stimuli[stim]["validation"] = False

        file_name = emodel + ".json"
        protocols_path = self.emodel_dir / "config" / "protocols" / file_name

        # create directory if needed.
        protocols_path.parent.mkdir(parents=True, exist_ok=True)

        s = json.dumps(stimuli, indent=2, cls=NumpyEncoder)
        with open(protocols_path, "w") as f:
            f.write(s)

    def get_recipes(self, emodel):
        """Load the recipes json for a given emodel.

        This assumes a specific folder structure per emodel, which consists in a folder per
        emodel in the emodel_dir, with the original config folder.
        """
        if self.legacy_dir_structure:
            emodel = "_".join(emodel.split("_")[:2])
            recipes_path = self.emodel_dir / emodel / "config" / "recipes" / "recipes.json"
        else:
            recipes_path = self.emodel_dir / self.recipes_path

        with open(recipes_path, "r") as f:
            return json.load(f)[emodel]

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
        """ Save an emodel obtained from BluePyOpt"""

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

        if githash:
            model_name = "{}__{}__{}".format(emodel, githash, seed)
        else:
            logger.warning(
                "Githash is %s. It is highly advise the use githash in the future.",
                githash,
            )
            model_name = "{}__{}".format(emodel, seed)

        final[model_name] = entry

        self.save_final(final)

    def get_parameters(self, emodel, species=None):
        """Get the definition of the parameters to optimize as well as the
         locations of the mechanisms. Also returns the name to the mechanisms.

        Args:
            emodel (str): name of the emodel

        Returns:
            params_definition (dict):
            mech_definition (dict):
            mech_names (list):

        """
        params = self._get_json(emodel, "params")
        params_definition = {
            "distributions": params["distributions"],
            "parameters": params["parameters"],
        }
        if "__comment" in params_definition["parameters"]:
            params_definition["parameters"].pop("__comment")

        mech_definition = params["mechanisms"]
        for mech in mech_definition.values():
            stoch = []
            for m in mech["mech"]:
                if "Stoch" in m:
                    stoch.append(True)
                else:
                    stoch.append(False)
            mech["stoch"] = stoch

        mech_names = []
        for mechs in mech_definition.values():
            mech_names += mechs["mech"]
        mech_names = list(set(mech_names))

        return params_definition, mech_definition, mech_names

    def _handle_extra_recording(self, emodel, extra_recordings):
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
                    logger.debug("No apical_points_isec.json found, bAP will be skipped")

            extra_recordings_out.append(extra)

        return extra_recordings_out

    def _read_protocol(self, emodel, prot, prot_name):

        stim_def = prot["step"]
        stim_def["holding_current"] = prot["holding"]["amp"]

        protocol_definition = {"type": prot["type"], "stimuli": stim_def}

        if "extra_recordings" in prot:
            protocol_definition["extra_recordings"] = self._handle_extra_recording(
                emodel, prot["extra_recordings"]
            )

        return protocol_definition

    def _read_legacy_protocol(self, emodel, prot, prot_name):

        if prot_name in ["RMP", "Rin", "RinHoldcurrent", "Main", "ThresholdDetection"]:
            return None

        if prot["type"] in ["StepThresholdProtocol", "StepProtocol"]:

            stim_def = prot["stimuli"]["step"]
            if "holding" in prot["stimuli"]:
                stim_def["holding_current"] = prot["stimuli"]["holding"]["amp"]
            else:
                stim_def["holding_current"] = None

            protocol_definition = {"type": prot["type"], "stimuli": stim_def}

            if "extra_recordings" in prot:
                protocol_definition["extra_recordings"] = self._handle_extra_recording(
                    emodel, prot["extra_recordings"]
                )

            return protocol_definition

        return None

    def get_protocols(self, emodel, species=None, delay=0.0, include_validation=False):
        """Get the protocols from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            delay (float): additional delay in ms to add at the start of
                the protocols.
            include_validation (bool): if True, returns the protocols for validation as well

        Returns:
            protocols_out (dict): protocols definitions

        """
        protocols = self._get_json(emodel, "protocol")
        protocols_out = {}
        for prot_name, prot in protocols.items():

            if "validation" in prot:

                if not (include_validation) and prot["validation"]:
                    continue

                prot_definition = self._read_protocol(emodel, prot, prot_name)
                if prot_definition:
                    protocols_out[prot_name] = prot_definition

            else:

                prot_definition = self._read_legacy_protocol(emodel, prot, prot_name)
                if prot_definition:
                    protocols_out[prot_name] = prot_definition

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
        """Get the efeatures from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            include_validation (bool): should the features for validation be returned as well

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
        """Get the name and path to the morphologies.

        Args:
            emodel (str): name of the emodel

        Returns:
            morphology_definition (dict): [{'name': morph_name,
                                            'path': 'morph_path'}

        """
        morphology_definition = []
        recipes = self.get_recipes(emodel)

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
            emodel (str): name of the emodels
            species (str): name of the species (rat, human, mouse)
        """
        final = self.get_final()

        if emodel in final:
            return self.format_emodel_data(final[emodel])

        logger.warning("Could not find the models for emodel %s", emodel)
        return None

    def get_emodels(self, emodels, species):
        """Get the list of emodels dictionaries.

        Args:
            emodels (list): list of names of the emodels
            species (str): name of the species (rat, human, mouse)
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

    def optimisation_state(self, emodel, checkpoint_dir, species=None, seed=1):
        """Return the state of the opttimisation.

        TODO: - should return three states: completed, in progress, empty
              - better management of checkpoints
        """
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{emodel}_{seed}.pkl"
        return checkpoint_path.is_file()
