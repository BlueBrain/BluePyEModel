"""LocalAccessPoint class."""

"""
Copyright 2023, EPFL/Blue Brain Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import glob
import json
import logging
from functools import cached_property
from itertools import chain
from pathlib import Path

import fasteners
from bluepyefe.tools import NumpyEncoder

from bluepyemodel.access_point.access_point import DataAccessPoint
from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration
from bluepyemodel.emodel_pipeline.emodel import EModel
from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.emodel_pipeline.emodel_workflow import EModelWorkflow
from bluepyemodel.evaluation.evaluator import LEGACY_PRE_PROTOCOLS
from bluepyemodel.evaluation.evaluator import PRE_PROTOCOLS
from bluepyemodel.evaluation.fitness_calculator_configuration import FitnessCalculatorConfiguration
from bluepyemodel.export_emodel.utils import get_hoc_file_path
from bluepyemodel.export_emodel.utils import get_output_path_from_metadata
from bluepyemodel.model.mechanism_configuration import MechanismConfiguration
from bluepyemodel.model.neuron_model_configuration import NeuronModelConfiguration
from bluepyemodel.tools.mechanisms import get_mechanism_currents
from bluepyemodel.tools.mechanisms import get_mechanism_name

logger = logging.getLogger(__name__)

seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}

SUPPORTED_MORPHOLOGY_EXTENSIONS = (".asc", ".swc", ".ASC", ".SWC")


class LocalAccessPoint(DataAccessPoint):
    """Access point to access configuration files and e-models when stored locally."""

    def __init__(
        self,
        emodel,
        emodel_dir=None,
        etype=None,
        ttype=None,
        mtype=None,
        species=None,
        brain_region=None,
        iteration_tag=None,
        synapse_class=None,
        final_path=None,
        recipes_path=None,
        legacy_dir_structure=False,
        with_seeds=False,
    ):
        """Init

        Args:
            emodel (str): name of the emodel.
            emodel_dir (str): path to the working directory.
                Default to current working directory.
                If iteration_tag is not None, it will be ./run/iteration_tag
            ttype (str): name of the t-type
            iteration_tag (str): iteration tag
            final_path (str): path to final.json which will contain the optimised models.
            recipes_path (str): path to the json file which should contain the path to the
                configuration files used for each etype. The content of this file should follow the
                format:

                .. code-block::

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

        # pylint: disable=too-many-arguments

        super().__init__(
            emodel,
            etype,
            ttype,
            mtype,
            species,
            brain_region,
            iteration_tag,
            synapse_class,
        )

        if emodel_dir is None:
            self.emodel_dir = Path.cwd()
            if iteration_tag:
                self.emodel_dir = self.emodel_dir / "run" / iteration_tag
        else:
            self.emodel_dir = Path(emodel_dir)

        self.recipes_path = recipes_path
        self.legacy_dir_structure = legacy_dir_structure
        self.with_seeds = with_seeds

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

    @cached_property
    def morph_dir(self):
        """Return the morphology directory as read from the recipes, or fallback to 'morphology'."""
        recipes = self.get_recipes()
        return Path(self.emodel_dir, recipes.get("morph_path", "morphology"))

    @cached_property
    def morph_path(self):
        """Return the path to the morphology file as read from the recipes."""
        recipes = self.get_recipes()

        morph_file = None
        if isinstance(recipes["morphology"], str):
            morph_file = recipes["morphology"]
        else:
            for _, morph_file in recipes["morphology"]:
                if morph_file.endswith(SUPPORTED_MORPHOLOGY_EXTENSIONS):
                    break

        if not morph_file or not morph_file.endswith(SUPPORTED_MORPHOLOGY_EXTENSIONS):
            raise FileNotFoundError(f"Morphology file not defined or not supported: {morph_file}")

        morph_path = self.morph_dir / morph_file

        if not morph_path.is_file():
            raise FileNotFoundError(f"Morphology file not found: {morph_path}")
        if str(Path.cwd()) not in str(morph_path.resolve()) and self.emodel_metadata.iteration:
            raise FileNotFoundError(
                "When using a githash or iteration tag, the path to the morphology must be local"
                " otherwise it cannot be archived during the creation of the githash. To solve"
                " this issue, you can copy the morphology from "
                f"{morph_path.resolve()} to {Path.cwd() / 'morphologies'} and update your "
                "recipes."
            )

        return morph_path

    def set_emodel(self, emodel):
        """Setter for the name of the emodel, check it exists (with or without seed) in recipe."""
        _emodel = "_".join(emodel.split("_")[:2]) if self.with_seeds else emodel
        if _emodel not in self.get_all_recipes():
            raise ValueError(
                f"Cannot set the emodel name to {_emodel} which does not exist in the recipes."
            )

        self.unfrozen_params = None
        self.emodel_metadata.emodel = _emodel

    def load_pipeline_settings(self):
        """ """
        recipes = self.get_recipes()
        settings = recipes.get("pipeline_settings", {})
        if isinstance(settings, str):
            # read the pipeline settings from file
            settings = self.get_json("pipeline_settings")
        if "morph_modifiers" not in settings:
            settings["morph_modifiers"] = recipes.get("morph_modifiers", None)
        return EModelPipelineSettings(**settings)

    def _config_to_final(self, config):
        """Convert the configuration stored in EM_*.json to the format used for final.json."""
        return {
            self.emodel_metadata.emodel: {
                **vars(self.emodel_metadata),
                "score": config["fitness"],  # float
                "parameters": config["parameter"],  # list[dict]
                "fitness": config["score"],  # list[dict]
                "features": config["features"],  # list[dict]
                "validation_fitness": config["scoreValidation"],  # list[dict]
                "validated": config["passedValidation"],  # bool
                "seed": config["seed"],  # int
            }
        }

    def get_final_content(self, lock_file=True):
        """Return the final content from recipes if available, or fallback to final.json"""
        recipes = self.get_recipes()
        if "final" in recipes:
            if self.final_path and self.final_path.is_file():
                logger.warning("Ignored %s, using file from recipes", self.final_path)
            data = self.get_json("final")
            return self._config_to_final(data)
        return self.get_final(lock_file=lock_file)

    def get_final(self, lock_file=True):
        """Get emodel dictionary from final.json."""
        if self.final_path is None:
            raise TypeError("Final_path is None")

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
            raise TypeError("Final_path is None")

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
            emodel = "_".join(self.emodel_metadata.emodel.split("_")[:2])
            recipes_path = self.emodel_dir / emodel / "config" / "recipes" / "recipes.json"
        else:
            recipes_path = self.emodel_dir / self.recipes_path

        with open(recipes_path, "r") as f:
            return json.load(f)

    def get_recipes(self):
        """Load the recipes from a json file for an emodel.

        See docstring of __init__ for the format of the file of recipes.
        """
        if self.with_seeds:
            emodel = "_".join(self.emodel_metadata.emodel.split("_")[:2])
        else:
            emodel = self.emodel_metadata.emodel
        return self.get_all_recipes()[emodel]

    def get_json_path(self, recipe_entry):
        """Helper function to get a json path in recipe."""

        json_path = Path(self.get_recipes()[recipe_entry])

        if self.legacy_dir_structure:
            emodel = "_".join(self.emodel_metadata.emodel.split("_")[:2])
            json_path = self.emodel_dir / emodel / json_path
        elif not json_path.is_absolute():
            json_path = self.emodel_dir / json_path
        return json_path

    def get_json(self, recipe_entry):
        """Helper function to load a json using path in recipe."""
        json_path = self.get_json_path(recipe_entry)
        with open(json_path, "r") as f:
            return json.load(f)

    def get_model_name_for_final(self, seed):
        """Return model name used as key in final.json."""

        if self.emodel_metadata.iteration:
            return f"{self.emodel_metadata.emodel}__{self.emodel_metadata.iteration}__{seed}"

        logger.warning(
            "The iteration is %s. It is strongly advised to use an iteration tag in the future.",
            self.emodel_metadata.iteration,
        )

        return f"{self.emodel_metadata.emodel}__{seed}"

    def store_emodel(self, emodel):
        """Store an emodel obtained from BluePyOpt in the final.json. Note that if a model in the
        final.json has the same key (emodel__iteration_tag__seed), it will be overwritten.
        """

        with self.rw_lock_final.write_lock():
            with self.rw_lock_final_tmp.write_lock():
                final = self.get_final(lock_file=False)
                model_name = self.get_model_name_for_final(emodel.seed)

                if model_name in final:
                    logger.warning(
                        "Entry %s was already in the final.json and will be overwritten",
                        model_name,
                    )

                pdf_dependencies = emodel.build_pdf_dependencies(int(emodel.seed))

                if self.new_final_path is None:
                    final_path = self.final_path
                    _params = emodel.parameters
                else:
                    if "params" in final[self.emodel_metadata.emodel]:
                        _params = final[self.emodel_metadata.emodel]["params"]
                    else:
                        _params = final[self.emodel_metadata.emodel]["parameters"]
                    _params.update(emodel.parameters)
                    final_path = self.new_final_path

                final[model_name] = vars(self.emodel_metadata)
                final[model_name].update(
                    {
                        "score": sum(list(emodel.scores.values())),
                        "parameters": _params,
                        "fitness": emodel.scores,
                        "features": emodel.features,
                        "validation_fitness": emodel.scores_validation,
                        "validated": emodel.passed_validation,
                        "seed": int(emodel.seed),
                        "pdfs": pdf_dependencies,
                    }
                )

                self.save_final(final, Path(final_path), lock_file=False)

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

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(configuration.as_dict(), f, indent=2)

    def get_mechanisms_directory(self):
        """Return the path to the directory containing the mechanisms for the current emodel"""

        mechanisms_directory = self.emodel_dir / "mechanisms"

        if mechanisms_directory.is_dir():
            return mechanisms_directory.resolve()

        return None

    def get_available_mechanisms(self):
        """Get the list of names of the available mechanisms"""

        mech_dir = self.get_mechanisms_directory()
        if mech_dir is None:
            return None

        available_mechanisms = []
        for mech_file in glob.glob(str(Path(mech_dir) / "*.mod")):
            ion_currents, nonspecific_currents, ion_conc = get_mechanism_currents(mech_file)
            name = get_mechanism_name(mech_file)
            available_mechanisms.append(
                MechanismConfiguration(
                    name=name,
                    location=None,
                    ion_currents=ion_currents,
                    nonspecific_currents=nonspecific_currents,
                    ionic_concentrations=ion_conc,
                )
            )

        return available_mechanisms

    def get_available_morphologies(self):
        """Get the list of names of available morphologies"""
        morph_dir = self.morph_dir

        if not morph_dir.is_dir():
            return None

        patterns = ["*" + ext for ext in SUPPORTED_MORPHOLOGY_EXTENSIONS]
        return {morph_file.stem for pattern in patterns for morph_file in morph_dir.glob(pattern)}

    def get_model_configuration(self):
        """Get the configuration of the model, including parameters, mechanisms and distributions"""

        configuration = NeuronModelConfiguration(
            available_mechanisms=self.get_available_mechanisms(),
            available_morphologies=self.get_available_morphologies(),
        )

        try:
            parameters = self.get_json("parameters")
        except KeyError:
            parameters = self.get_json("params")

        if isinstance(parameters["parameters"], dict):
            parameters["parameters"].pop("__comment", None)

        if self.unfrozen_params is not None:
            self._freeze_params(parameters["parameters"])

        if isinstance(parameters["mechanisms"], dict):
            configuration.init_from_legacy_dict(parameters, self.get_morphologies())
        else:
            configuration.init_from_dict(parameters, self.get_morphologies())

        configuration.mapping_multilocation = self.get_recipes().get("multiloc_map", None)

        return configuration

    def store_targets_configuration(self, configuration):
        """Store the configuration of the targets (targets and ephys files used)"""

        config_dict = {
            "emodel": self.emodel_metadata.emodel,
            "ttype": self.emodel_metadata.ttype,
        }

        config_dict.update(configuration.as_dict())

        path_extract_config = self.pipeline_settings.path_extract_config
        Path(path_extract_config).parent.mkdir(parents=True, exist_ok=True)

        with open(str(path_extract_config), "w") as f:
            f.write(json.dumps(config_dict, indent=2, cls=NumpyEncoder))

    def get_targets_configuration(self):
        """Get the configuration of the targets (targets and ephys files used)"""

        path_extract_config = self.emodel_dir / self.pipeline_settings.path_extract_config

        with open(path_extract_config, "r") as f:
            config_dict = json.load(f)

        configuration = TargetsConfiguration(
            files=config_dict["files"],
            targets=config_dict["targets"],
            protocols_rheobase=config_dict["protocols_rheobase"],
            additional_fitness_efeatures=config_dict.get("additional_fitness_efeatures", None),
            additional_fitness_protocols=config_dict.get("additional_fitness_protocols", None),
            protocols_mapping=config_dict.get("protocols_mapping", None),
        )

        return configuration

    def store_fitness_calculator_configuration(self, configuration):
        """Store a fitness calculator configuration"""

        config_path = self.emodel_dir / self.get_recipes()["features"]
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(config_path), "w") as f:
            f.write(json.dumps(configuration.as_dict(), indent=2, cls=NumpyEncoder))

    def get_fitness_calculator_configuration(self, record_ions_and_currents=False):
        """Get the configuration of the fitness calculator (efeatures and protocols)"""

        config_dict = self.get_json("features")

        legacy = "efeatures" not in config_dict and "protocols" not in config_dict

        # contains ion currents and ionic concentrations to be recorded
        ion_variables = None
        if record_ions_and_currents:
            ion_currents, ionic_concentrations = self.get_ion_currents_concentrations()
            if ion_currents is not None and ionic_concentrations is not None:
                ion_variables = list(chain.from_iterable((ion_currents, ionic_concentrations)))

        if legacy:
            efeatures = self.get_json("features")
            protocols = self.get_json("protocol")

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
                validation_protocols=self.pipeline_settings.validation_protocols,
                stochasticity=self.pipeline_settings.stochasticity,
                ion_variables=ion_variables,
            )

            if from_bpe:
                configuration.init_from_bluepyefe(
                    efeatures,
                    protocols,
                    {},
                    self.pipeline_settings.threshold_efeature_std,
                )
            else:
                configuration.init_from_legacy_dict(
                    efeatures, protocols, self.pipeline_settings.threshold_efeature_std
                )

        else:
            configuration = FitnessCalculatorConfiguration(
                efeatures=config_dict["efeatures"],
                protocols=config_dict["protocols"],
                name_rmp_protocol=self.pipeline_settings.name_rmp_protocol,
                name_rin_protocol=self.pipeline_settings.name_Rin_protocol,
                validation_protocols=self.pipeline_settings.validation_protocols,
                stochasticity=self.pipeline_settings.stochasticity,
                ion_variables=ion_variables,
            )

        return configuration

    def create_emodel_workflow(self, state="not launched"):
        """Create an empty EModelWorkflow instance. EModel workflow should not be used in local"""
        return EModelWorkflow(
            None,
            None,
            None,
            state=state,
        )

    def get_emodel_workflow(self):
        """Emodel workflow is not used in local, so return None here"""
        return None, None

    def check_emodel_workflow_configurations(self, emodel_workflow):
        """Emodel workflow is not used in local, so always return True to let the workflow run"""
        # pylint: disable=unused-argument
        return True

    def store_or_update_emodel_workflow(self, emodel_workflow):
        """Emodel workflow is not used in local, so pass"""

    def get_morphologies(self):
        """Get the name and path to the morphologies from the recipes.

        Returns:
            morphology_definition (list): [{'name': morph_name, 'path': 'morph_path'}]. Might
            contain the additional entries "seclist_names" and "secarray_names" if they are
            present in the recipes.
        """

        recipes = self.get_recipes()
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

        if "githash" in model_data:
            iteration_tag = model_data["githash"]
        elif "iteration_tag" in model_data:
            iteration_tag = model_data["iteration_tag"]
        elif "iteration" in model_data:
            iteration_tag = model_data["iteration"]
        else:
            iteration_tag = None

        emodel_metadata = EModelMetadata(
            emodel=model_data.get("emodel", self.emodel_metadata.emodel),
            etype=model_data.get("etype", None),
            ttype=model_data.get("ttype", None),
            mtype=model_data.get("mtype", None),
            species=model_data.get("species", None),
            brain_region=model_data.get("brain_region", None),
            iteration_tag=iteration_tag,
            synapse_class=model_data.get("synapse_class", None),
        )

        emodel = EModel(
            fitness=model_data.get("score", None),
            parameter=model_data.get("params", model_data.get("parameters", None)),
            score=model_data.get("fitness", None),
            features=model_data.get("features", None),
            scoreValidation=model_data.get("validation_fitness", None),
            passedValidation=model_data.get("validated", None),
            seed=model_data.get("seed", None),
            emodel_metadata=emodel_metadata,
        )

        return emodel

    def get_emodel(self, lock_file=True):
        """Get dict with parameter of single emodel (including seed if any)"""

        final = self.get_final_content(lock_file=lock_file)

        if self.emodel_metadata.emodel in final:
            return self.format_emodel_data(final[self.emodel_metadata.emodel])

        logger.warning("Could not find models for emodel %s", self.emodel_metadata.emodel)

        return None

    def get_emodels(self, emodels=None):
        """Get a list of emodels

        Args:
            emodels (list): list of names of the emodels.
        """

        if emodels is None:
            emodels = [self.emodel_metadata.emodel]

        models = []
        for mod_data in self.get_final_content().values():
            if mod_data["emodel"] in emodels:
                models.append(self.format_emodel_data(mod_data))

        filtered_models = []
        api_metadata = self.emodel_metadata.for_resource()
        for m in models:
            model_metadata = m.emodel_metadata.for_resource()
            for f, v in api_metadata.items():
                if f in model_metadata and v != model_metadata[f]:
                    break
            else:
                filtered_models.append(m)

        return filtered_models

    def has_pipeline_settings(self):
        """Returns True if pipeline settings are present in the recipes"""

        return "pipeline_settings" in self.get_recipes()

    def has_fitness_calculator_configuration(self):
        """Check if the fitness calculator configuration exists"""

        recipes = self.get_recipes()

        return Path(recipes["features"]).is_file()

    def has_targets_configuration(self):
        """Check if the target configuration exists"""

        return (
            self.pipeline_settings.path_extract_config
            and Path(self.pipeline_settings.path_extract_config).is_file()
        )

    def has_model_configuration(self):
        """Check if the model configuration exists"""

        recipes = self.get_recipes()

        return Path(recipes["params"]).is_file()

    def get_emodel_etype_map(self):
        final = self.get_final_content()
        return {emodel: emodel.split("_")[0] for emodel in final}

    def get_emodel_names(self):
        """Get the list of all the names of emodels

        Returns:
            dict: keys are emodel names with seed, values are names without seed.
        """

        final = self.get_final_content()

        return {mod_name: mod.get("emodel", mod_name) for mod_name, mod in final.items()}

    def has_best_model(self, seed):
        """Check if the best model has been stored."""

        final = self.get_final_content()

        model_name = self.get_model_name_for_final(seed)

        return model_name in final

    def is_checked_by_validation(self, seed):
        """Check if the emodel with a given seed has been checked by Validation task."""

        final = self.get_final_content()

        model_name = self.get_model_name_for_final(seed)

        model = final.get(model_name, {})
        if "validated" in model and model["validated"] is not None:
            return True

        return False

    def is_validated(self):
        """Check if enough models have been validated."""

        n_validated = 0
        final = self.get_final_content()

        for _, entry in final.items():
            if (
                entry["iteration"] == self.emodel_metadata.iteration
                and entry["emodel"] == self.emodel_metadata.emodel
                and entry["validated"]
            ):
                n_validated += 1

        return n_validated >= self.pipeline_settings.n_model

    @classmethod
    def add_entry_recipes(
        cls,
        recipes_path,
        emodel,
        morph_path,
        morphology,
        parameters_path,
        protocols_path,
        features_path,
        pipeline_settings=None,
    ):
        """Append an entry to the recipes, create the file if it does not exist

        Args:
            recipes_path (str): path to the json containig the recipes.
            emodel (str): name of the emodel.
            morph_path (str): path the directory containing the morphologies.
            morphology (list of str): name of the morphology and name of the file.
            parameters_path (str): path to the json that contains the parameters.
            protocols_path (str): path to the json that contains the protocols.
            features_path (str): path to the json that contains the features.
            pipeline_settings (dict): pipeline settings.
        """

        recipes_path = Path(recipes_path)
        recipes_path.parent.mkdir(parents=True, exist_ok=True)

        if recipes_path.is_file():
            with recipes_path.open("r") as f:
                recipes = json.load(f)
        else:
            recipes = {}

        if pipeline_settings is None:
            settings = EModelPipelineSettings().as_dict()
        else:
            settings = pipeline_settings

        recipes[emodel] = {
            "morph_path": morph_path,
            "morphology": [morphology],
            "params": parameters_path,
            "protocol": protocols_path,
            "features": features_path,
            "pipeline_settings": settings,
        }

        with recipes_path.open("w") as f:
            json.dump(recipes, f, indent=2)

    def store_hocs(
        self,
        only_validated=False,
        only_best=True,
        seeds=None,
        map_function=map,
        new_emodel_name=None,
        description=None,
        output_base_dir="export_emodels_hoc",
    ):
        """Not Implemented"""
        raise NotImplementedError

    def store_emodels_hoc(
        self,
        only_validated=False,
        only_best=True,
        seeds=None,
        map_function=map,
        new_emodel_name=None,
        description=None,
    ):
        """Not Implemented"""
        raise NotImplementedError

    def store_emodels_sonata(
        self,
        only_validated=False,
        only_best=True,
        seeds=None,
        map_function=map,
        new_emodel_name=None,
        description=None,
    ):
        """Not Implemented"""
        raise NotImplementedError

    def sonata_exists(self, seed):
        """Returns True if the sonata hoc file has been exported"""
        output_path = get_output_path_from_metadata(
            "export_emodels_sonata", self.emodel_metadata, seed
        )
        hoc_file_path = get_hoc_file_path(output_path)
        return Path(hoc_file_path).is_file()
