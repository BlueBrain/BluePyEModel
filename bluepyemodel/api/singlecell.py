"""API to reproduce Singlecell repositories."""
import json
from pathlib import Path
import logging

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger(__name__)

# pylint: disable=W0231,W0221


class Singlecell_API(DatabaseAPI):
    """API to reproduce Singlecell repositories."""

    def __init__(self, working_dir, final_path=None):
        """
        Args:
            final_path (str): Path to final.json
        """
        self.working_dir = Path(working_dir)
        if final_path is None:
            self.final_path = self.working_dir / "final.json"
        else:
            self.final_path = final_path
        logger.info("We are using final.json from: %s", self.final_path)

    def get_recipes(self, emodel):
        """Load the recipes json for a given emodel.

        This assumes a specific folder structure per emodel, which consists in a folder per
        emodel in the working_dir, with the original config folder.
        """
        recipes_path = self.working_dir / emodel / "config" / "recipes" / "recipes.json"
        with open(recipes_path, "r") as f:
            return json.load(f)[emodel]

    def store_efeatures(self, emodel, efeatures, current):
        """ Save the efeatures and currents obtained from BluePyEfe"""

    def store_protocols(self, emodel, stimuli):
        """ Save the protocols obtained from BluePyEfe"""

    def store_model(
        self,
        emodel,
        checkpoint_path,
        param_names,
        feature_names,
        optimizer_name,
        opt_params,
        validated=False,
    ):
        """ Save a model obtained from BluePyOpt"""

    def get_extraction_metadata(self, emodel):
        """Gather the metadata used to build the config dictionary used as an
        input by BluePyEfe.

        Args:
            emodel (str): name of the emodel

        Returns:
            cells (dict): return the cells recordings metadata
            protocols (dict): return the protocols metadata

        """

    def get_parameters(self, emodel):
        """Get the definition of the parameters to optimize as well as the
         locations of the mechanisms. Also returns the name to the mechanisms.

        Args:
            emodel (str): name of the emodel

        Returns:
            params_definition (dict):
            mech_definition (dict):
            mech_names (list):

        """
        json_path = self.working_dir / emodel / self.get_recipes(emodel)["params"]
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

    def get_protocols(
        self,
        emodel,
        name_holding_rin_protocol="Rin",
        name_rmp_protocol="RMP",
        exclude_protocol=None,
        delay=0.0,
    ):
        """Get the protocols from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            name_holding_rin_protocol (str): name of the protocol that will be
                used to compute the holding current and input resistance (e.g.:
                "IV_-40.0").
            name_rmp_protocol (str): name of the protocol that will be
                used to compute the resting membrane potential (e.g.: "IV_0.0").
            exclude_protocol (list): protocol names that should not be used for
                optimization (e.g.: ['IDThresh'])
            delay (float): additional delay in ms to add at the start of
                the protocols.

        Returns:
            protocols_out (dict): protocols definitions

        """

        # TODO: handle extra recordings ?
        if exclude_protocol is None:
            exclude_protocol = []

        json_path = self.working_dir / emodel / self.get_recipes(emodel)["protocol"]
        with open(json_path, "r") as f:
            protocols = json.load(f)

        protocols_out = {}
        for prot_name, prot in protocols.items():

            if prot_name in ("Main", "RinHoldcurrent", "ThresholdDetection"):
                continue
            for to_exclude in exclude_protocol:
                if to_exclude in prot_name:
                    flag = True
                    break
            else:
                flag = False
            if flag:
                continue

            if prot_name == name_holding_rin_protocol:

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

            elif prot_name == name_rmp_protocol:
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

    def get_features(self, emodel):
        """Get the efeatures from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            name_holding_rin_protocol (str): name of the protocol that will be
                used to compute the holding current and input resistance (e.g.:
                "IV_-40.0").
            exclude_efeatures (dict): features that should not be used for
                optimization. Of the form {'protocol_name': ['feature_name']}

        Returns:
            efeatures_out (dict): efeatures definitions

        """
        json_path = self.working_dir / emodel / self.get_recipes(emodel)["features"]
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

    def get_morphologies(self, emodel):
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

    def get_mechanism_paths(self, mechanism_names):
        """Get the path of the mod files

        Args:
            mechanism_names (list): names of the mechanisms

        Returns:
            mechanism_paths (dict): {'mech_name': 'mech_path'}

        """
        return []

    def get_emodel(self, emodel):
        """Get the emodel data."""
        with open(self.final_path, "r") as f:
            final = json.load(f)[emodel]

        return {
            "emodel": emodel,
            "fitness": final["score"],
            "parameters": final["params"],
            "scores": final["fitness"],
            "validated": True,
            "optimizer": "proj38",
        }

    def close(self):
        """Close database."""
