"""Access point to get data from Singlecell-like repositories"""

import glob
import json
import logging
from pathlib import Path

import fasteners
from bluepyefe.tools import NumpyEncoder

from bluepyemodel.access_point.access_point import DataAccessPoint
from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.evaluation.evaluator import LEGACY_PRE_PROTOCOLS
from bluepyemodel.evaluation.evaluator import PRE_PROTOCOLS
from bluepyemodel.evaluation.fitness_calculator_configuration import FitnessCalculatorConfiguration
from bluepyemodel.model.neuron_model_configuration import NeuronModelConfiguration
from bluepyemodel.tools import search_pdfs

logger = logging.getLogger(__name__)

seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}


class LocalAccessPoint(DataAccessPoint):
    """Access point to configuration files organized as project 38."""

    def __init__(
        self,
        emodel,
        emodel_dir,
        ttype=None,
        iteration_tag=None,
        final_path=None,
        recipes_path=None,
        legacy_dir_structure=False,
        with_seeds=False,
    ):
        """Init

        Args:
            emodel (str): name of the emodel.
            emodel_dir (str): path to the working directory.
            ttype (str): name of the t-type
            iteration_tag (str): iteration tag
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
            with_seed (bool): allows for emodel_seed type of emodel names in final.json
        """

        self.emodel_dir = Path(emodel_dir)
        self.recipes_path = recipes_path
        self.legacy_dir_structure = legacy_dir_structure
        self.emodel = emodel
        self.with_seeds = with_seeds

        super().__init__(emodel, ttype, iteration_tag)

        self.morph_path = None

        if final_path is None:
            self.final_path = self.emodel_dir / "final.json"
        else:
            self.final_path = Path(final_path)

        # variable to set if one wants to save a new final path after optimisation
        self.new_final_path = None

        Path(".tmp/").mkdir(exist_ok=True)
        self.rw_lock_final = fasteners.InterProcessReaderWriterLock(".tmp/final.lock")
        self.rw_lock_final_tmp = fasteners.InterProcessReaderWriterLock(".tmp/final_tmp.lock")

        self.pipeline_settings = self.load_pipeline_settings()
        self.unfrozen_params = None

    def set_emodel(self, emodel):
        """Setter for the name of the emodel, check it exists (with or without seed) in recipe."""
        _emodel = "_".join(emodel.split("_")[:2]) if self.with_seeds else emodel
        if _emodel not in self.get_all_recipes():
            raise Exception(
                f"Cannot set the emodel name to {_emodel} which does not exist in the recipes."
            )

        self.unfrozen_params = None
        self.emodel = emodel

    def load_pipeline_settings(self):
        """ """

        settings = self.get_recipes().get("pipeline_settings", {})

        if "morph_modifiers" not in settings:
            settings["morph_modifiers"] = self.get_recipes().get("morph_modifiers", None)

        return EModelPipelineSettings(**settings)

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

    def save_final(self, final, final_path, lock_file=True):
        """Save final emodels in json"""

        if final_path is None:
            raise Exception("Final_path is None")

        if lock_file:
            self.rw_lock_final.acquire_write_lock()

        with open(final_path, "w+") as fp:
            json.dump(final, fp, indent=2)

        if lock_file:
            self.rw_lock_final.release_write_lock()
            self.rw_lock_final_tmp.acquire_write_lock()

        with open(final_path.with_name(final_path.stem + "_tmp.json"), "w+") as fp:
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
        if self.with_seeds:
            emodel = "_".join(self.emodel.split("_")[:2])
        else:
            emodel = self.emodel
        return self.get_all_recipes()[emodel]

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

    def get_model_name_for_final(self, seed):
        """Return model name used as key in final.json."""

        if self.iteration_tag:
            return f"{self.emodel}__{self.iteration_tag}__{seed}"

        logger.warning(
            "The iteration_tag is %s. It is strongly advised to use "
            "an iteration tag in the future.",
            self.iteration_tag,
        )

        return f"{self.emodel}__{seed}"

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
        """Store an emodel obtained from BluePyOpt in the final.json. Note that if a model in the
        final.json has the same key (emodel__iteration_tag__seed), it will be overwritten.

        Args:
            scores (dict): scores of the efeatures. Of the format {"objective_name": score}.
            params (dict): values of the parameters. Of the format {"param_name": param_value}.
            optimizer_name (str): name of the optimizer (IBEA, CMA, ...).
            seed (int): seed used by the optimizer.
            validated (bool): None indicate that the model did not go through validation.
                False indicates that it failed validation. True indicates that it
                passed validation.
            scores_validation (dict): scores of the validation efeatures. Of the format
                {"objective_name": score}.
            features (dict): value of the efeatures. Of the format {"objective_name": value}.
        """

        if scores_validation is None:
            scores_validation = {}

        if features is None:
            features = {}

        with self.rw_lock_final.write_lock():
            with self.rw_lock_final_tmp.write_lock():

                final = self.get_final(lock_file=False)
                model_name = self.get_model_name_for_final(seed)

                if model_name in final:
                    logger.warning(
                        "Entry %s was already in the final.json and will be overwritten", model_name
                    )

                pdf_dependencies = self._build_pdf_dependencies(int(seed))

                if self.new_final_path is None:
                    final_path = self.final_path
                    _params = params
                else:
                    _params = final[self.emodel]["params"]
                    _params.update(params)
                    final_path = self.new_final_path

                final[model_name] = {
                    "emodel": self.emodel,
                    "score": sum(list(scores.values())),
                    "params": _params,
                    "fitness": scores,
                    "features": features,
                    "validation_fitness": scores_validation,
                    "validated": validated,
                    "optimizer": str(optimizer_name),
                    "seed": int(seed),
                    "ttype": self.ttype,
                    "iteration_tag": str(self.iteration_tag),
                    "pdfs": pdf_dependencies,
                }
                self.save_final(final, Path(final_path), lock_file=False)

    def _build_pdf_dependencies(self, seed):
        """Find all the pdfs associated to an emodel"""

        pdfs = {}

        pdfs["optimisation"] = search_pdfs.search_figure_emodel_optimisation(
            self.emodel, seed, self.iteration_tag
        )
        pdfs["traces"] = search_pdfs.search_figure_emodel_traces(
            self.emodel, seed, self.iteration_tag
        )
        pdfs["scores"] = search_pdfs.search_figure_emodel_score(
            self.emodel, seed, self.iteration_tag
        )
        pdfs["parameters"] = search_pdfs.search_figure_emodel_parameters(self.emodel)

        return pdfs

    def set_unfrozen_params(self, params):
        """Freeze parameters for partial optimisation."""
        # todo: check if params exists in final.json
        self.unfrozen_params = params

    def _freeze_params(self, params):
        """Freeze parameters to final except these in self.unfrozen_params."""
        emodel_params = self.get_emodel()["parameters"]
        for loc in params:
            for param in params[loc]:
                name = ".".join([param["name"], loc])
                if name not in self.unfrozen_params and isinstance(param["val"], list):
                    param["val"] = emodel_params[name]

    def store_model_configuration(self, configuration, path=None):
        """Store a model configuration as a json"""

        if path is None:
            path = Path(self.get_recipes()["params"])

        path.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(configuration.as_dict(), f, indent=2)

    def get_mechanisms_directory(self):
        """Return the path to the directory containing the mechanisms for the current emodel"""

        if self.emodel_dir:
            mechanisms_directory = self.emodel_dir / "mechanisms"
        else:
            mechanisms_directory = Path("./") / "mechanisms"

        if mechanisms_directory.is_dir():
            return mechanisms_directory.resolve()

        return None

    def get_available_mechanisms(self):
        """Get the list of names of the available mechanisms"""

        mechs = []

        mech_dir = self.get_mechanisms_directory()

        if mech_dir is None:
            return None

        for mech_file in glob.glob(str(Path(mech_dir) / "*.mod")):
            mechs.append(Path(mech_file).stem)

        return [{"name": m, "version": None} for m in set(mechs)]

    def get_available_morphologies(self):
        """Get the list of names of available morphologies"""

        names = []

        if self.emodel_dir:
            morph_dir = self.emodel_dir / "morphology"
        else:
            morph_dir = Path("./") / "morphology"

        if not morph_dir.is_dir():
            return None

        for morph_file in glob.glob(str(morph_dir / "*.asc")) + glob.glob(str(morph_dir / "*.swc")):
            names.append(Path(morph_file).stem)

        return set(names)

    def get_model_configuration(self):
        """Get the configuration of the model, including parameters, mechanisms and distributions"""

        configuration = NeuronModelConfiguration(
            configuration_name=None,
            available_mechanisms=self.get_available_mechanisms(),
            available_morphologies=self.get_available_morphologies(),
        )

        parameters = self._get_json("params")

        parameters["parameters"].pop("__comment", None)

        if self.unfrozen_params is not None:
            self._freeze_params(parameters["parameters"])

        if isinstance(parameters["mechanisms"], dict):
            configuration.init_from_legacy_dict(parameters, self.get_morphologies())
        else:
            configuration.init_from_dict(parameters)

        configuration.mapping_multilocation = self.get_recipes().get("multiloc_map", None)

        return configuration

    def store_targets_configuration(self, configuration):
        """Store the configuration of the targets (targets and ephys files used)"""

        config_dict = {
            "emodel": self.emodel,
            "ttype": self.ttype,
        }

        config_dict.update(configuration.as_dict())

        path_extract_config = self.pipeline_settings.path_extract_config
        path_extract_config.parent.mkdir(parents=True, exist_ok=True)

        with open(str(path_extract_config), "w") as f:
            f.write(json.dumps(config_dict, indent=2, cls=NumpyEncoder))

    def get_targets_configuration(self):
        """Get the configuration of the targets (targets and ephys files used)"""

        path_extract_config = self.pipeline_settings.path_extract_config

        with open(path_extract_config, "r") as f:
            config_dict = json.load(f)

        configuration = TargetsConfiguration(
            files=config_dict["files_metadata"],
            targets=config_dict["targets"],
            protocols_rheobase=config_dict["protocols_threshold"],
        )

        return configuration

    def store_fitness_calculator_configuration(self, configuration):
        """Store a fitness calculator configuration"""

        config_path = self.emodel_dir / self.get_recipes()["features"]
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(config_path), "w") as f:
            f.write(json.dumps(configuration.as_dict(), indent=2, cls=NumpyEncoder))

    def get_fitness_calculator_configuration(self):
        """Get the configuration of the fitness calculator (efeatures and protocols)"""

        config_dict = self._get_json("features")

        legacy = "efeatures" not in config_dict and "protocols" not in config_dict

        if legacy:

            efeatures = self._get_json("features")
            protocols = self._get_json("protocol")

            from_bpe = False
            for protocol_name, protocol in protocols.items():
                if protocol_name in PRE_PROTOCOLS + LEGACY_PRE_PROTOCOLS:
                    continue
                if "stimuli" not in protocol:
                    from_bpe = True
                    break

            configuration = FitnessCalculatorConfiguration(
                name_rmp_protocol=self.pipeline_settings.name_rmp_protocol,
                name_rin_protocol=self.pipeline_settings.name_Rin_protocol,
                threshold_efeature_std=self.pipeline_settings.threshold_efeature_std,
                validation_protocols=self.pipeline_settings.validation_protocols,
            )

            if from_bpe:
                configuration.init_from_bluepyefe(efeatures, protocols, {})
            else:
                configuration.init_from_legacy_dict(efeatures, protocols)

        else:
            configuration = FitnessCalculatorConfiguration(
                efeatures=config_dict["efeatures"],
                protocols=config_dict["protocols"],
                name_rmp_protocol=self.pipeline_settings.name_rmp_protocol,
                name_rin_protocol=self.pipeline_settings.name_Rin_protocol,
                threshold_efeature_std=self.pipeline_settings.threshold_efeature_std,
                validation_protocols=self.pipeline_settings.validation_protocols,
            )

        return configuration

    def get_morphologies(self):
        """Get the name and path to the morphologies from the recipes.

        Returns:
            morphology_definition (list): [{'name': morph_name, 'path': 'morph_path'}]. Might
            contain the additional entries "seclist_names" and "secarray_names" if they are
            present in the recipes.
        """

        recipes = self.get_recipes()

        morph_def = recipes["morphology"][0]

        if self.morph_path is None:
            self.morph_path = Path(recipes["morph_path"]) / morph_def[1]
            if not self.morph_path.is_absolute():
                self.morph_path = Path(self.emodel_dir) / self.morph_path
        else:
            self.morph_path = Path(self.morph_path)

        morphology_definition = {
            "name": self.morph_path.stem,
            "path": str(self.morph_path),
        }
        if "seclist_names" in recipes:
            morphology_definition["seclist_names"] = recipes["seclist_names"]

        if "secarray_names" in recipes:
            morphology_definition["secarray_names"] = recipes["secarray_names"]

        return morphology_definition

    def format_emodel_data(self, model_data):
        """Format emodel data."""

        out_data = {
            "emodel": self.emodel,
            "fitness": model_data.get("score", None),
            "parameters": model_data["params"],
            "scores": model_data.get("fitness", None),
        }

        for key in [
            "seed",
            "iteration_tag",
            "ttype",
            "branch",
            "rank",
            "optimizer",
            "validated",
        ]:
            if key in model_data:
                out_data[key] = model_data[key]

        if "githash" in model_data:
            out_data["iteration_tag"] = model_data["githash"]
        if "iteration_tag" not in out_data:
            out_data["iteration_tag"] = None

        if "validated" not in out_data:
            out_data["validated"] = None
        if "validation_fitness" in model_data:
            out_data["scores_validation"] = model_data["validation_fitness"]

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

    def has_fitness_calculator_configuration(self):
        """Check if the fitness calculator configuration exists"""

        return (
            Path(self.emodel_dir, "config", "features", self.emodel).with_suffix(".json").is_file()
        )

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

    def has_best_model(self, seed):
        """Check if the best model has been stored."""

        final = self.get_final()

        model_name = self.get_model_name_for_final(seed)

        return model_name in final

    def is_checked_by_validation(self, seed):
        """Check if the emodel with a given seed has been checked by Validation task."""

        final = self.get_final()

        model_name = self.get_model_name_for_final(seed)

        model = final.get(model_name, {})
        if "validated" in model and model["validated"] is not None:
            return True

        return False

    def is_validated(self):
        """Check if enough models have been validated."""

        n_validated = 0
        final = self.get_final()

        for _, entry in final.items():

            if (
                entry["iteration_tag"] == self.iteration_tag
                and entry["emodel"] == self.emodel
                and entry["validated"]
            ):
                n_validated += 1

        return n_validated >= self.pipeline_settings.n_model
