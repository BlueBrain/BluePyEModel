"""API to get data from Singlecell-like repositories."""
import json
from pathlib import Path
import logging

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger(__name__)

# pylint: disable=W0231,W0221,W0613


class Singlecell_API(DatabaseAPI):
    """API to reproduce Singlecell repositories."""

    def __init__(
        self,
        working_dir,
        final_path=None,
        recipes_path=None,
        legacy_dir_structure=False,
    ):
        """
        Args:
            working_dir (str): path to the workign directory
            final_path (str): path to final.json
            recipes_path (str): path to the recipes.json
        """
        self.working_dir = Path(working_dir)
        self.recipes_path = recipes_path
        self.legacy_dir_structure = legacy_dir_structure
        if final_path is None:
            self.final_path = self.working_dir / "final.json"
        else:
            self.final_path = final_path

    def get_recipes(self, emodel):
        """Load the recipes json for a given emodel.

        This assumes a specific folder structure per emodel, which consists in a folder per
        emodel in the working_dir, with the original config folder.
        """
        if self.legacy_dir_structure:
            recipes_path = (
                self.working_dir / emodel / "config" / "recipes" / "recipes.json"
            )
        else:
            recipes_path = self.recipes_path

        with open(recipes_path, "r") as f:
            return json.load(f)[emodel]

    def store_model(
        self,
        emodel,
        scores,
        params,
        optimizer_name,
        seed,
        validated=False,
        species=None,
    ):
        """ Save a model obtained from BluePyOpt"""

        if self.final_path is None:
            raise Exception("Cannot store the model because final_path is None")

        p = Path(self.final_path)
        if not (p.is_file()):
            logger.info("%s does not exist and will be create", self.final_path)
            final = {}
        else:
            with open(self.final_path, "r") as f:
                final = json.load(f)

        entry = {
            "emodel": emodel,
            "species": species,
            "score": sum(list(scores.values())),
            "params": params,
            "fitness": scores,
            "validated": validated,
            "optimizer": str(optimizer_name),
            "seed": int(seed),
        }

        model_name = "{}_{}".format(emodel, seed)

        if model_name in final.keys():
            if final[model_name]["score"] > entry["score"]:
                final[model_name] = entry
        else:
            final[model_name] = entry

        with open(self.final_path, "w") as fp:
            json.dump(final, fp, indent=2)

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
        if self.legacy_dir_structure:
            json_path = self.working_dir / emodel / self.get_recipes(emodel)["params"]
        else:
            json_path = self.get_recipes(emodel)["params"]
        with open(json_path, "r") as f:
            params = json.load(f)

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

    def get_protocols(self, emodel, species, delay=0.0, include_validation=False):
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
        # TODO: handle extra recordings ?
        if self.legacy_dir_structure:
            json_path = self.working_dir / emodel / self.get_recipes(emodel)["protocol"]
        else:
            json_path = self.get_recipes(emodel)["protocol"]
        with open(json_path, "r") as f:
            protocols = json.load(f)

        protocols_out = {}
        for prot_name, prot in protocols.items():

            if prot_name in ("Main", "RinHoldcurrent", "ThresholdDetection"):
                continue

            if prot_name == "Rin":

                protocols_out["RinHoldCurrent"] = {
                    "type": "RinHoldCurrent",
                    "stimuli": {
                        "step": prot["stimuli"]["step"],
                        "holding": {
                            "delay": 0,
                            "amp": None,
                            "duration": prot["stimuli"]["holding"]["duration"],
                            "totduration": prot["stimuli"]["holding"]["totduration"],
                        },
                    },
                }

            elif prot_name == "RMP":
                # The name_rmp_protocol is used only for the efeatures, the
                # protocol itself is fixed:
                protocols_out["RMP"] = {
                    "type": "RMP",
                    "stimuli": {
                        "step": {
                            "delay": 250,
                            "amp": 0,
                            "duration": 400,
                            "totduration": 650,
                        },
                        "holding": {
                            "delay": 0,
                            "amp": 0,
                            "duration": 650,
                            "totduration": 650,
                        },
                    },
                }

            elif prot["type"] == "StepThresholdProtocol":
                protocols_out[prot_name] = prot

        return protocols_out

    def get_features(
        self,
        emodel,
        species,
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
        if self.legacy_dir_structure:
            json_path = self.working_dir / emodel / self.get_recipes(emodel)["features"]
        else:
            json_path = self.get_recipes(emodel)["features"]
        with open(json_path, "r") as f:
            efeatures = json.load(f)

        efeatures_out = {
            "RMP": {"soma.v": []},
            "RinHoldCurrent": {"soma.v": []},
            "Threshold": {"soma.v": []},
        }

        for prot_name in efeatures:
            for loc in efeatures[prot_name]:
                for efeat in efeatures[prot_name][loc]:

                    # Put Rin and RinHoldCurrent together
                    if prot_name == "Rin":
                        efeatures_out["RinHoldCurrent"][loc].append(efeat)

                    # Others
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
            morphology_definition.append(
                {"name": morph_def[1][:-4], "path": str(morph_path)}
            )

        return morphology_definition

    def get_models(self, emodel, species=None):
        """Get the emodel data."""
        with open(self.final_path, "r") as f:
            final = json.load(f)

        models = []
        for model_name in final:
            if final[model_name]["emodel"] == emodel:

                model = {
                    "emodel": emodel,
                    "fitness": final[model_name]["score"],
                    "parameters": final[model_name]["params"],
                    "scores": final[model_name]["fitness"],
                    "validated": "False",
                }

                for key in [
                    "seed",
                    "githash",
                    "branch",
                    "rank",
                    "optimiser",
                    "species",
                ]:
                    if key in final[model_name]:
                        model[key] = final[model_name][key]

                models.append(model)

        return models
