"""
Copyright 2024, EPFL/Blue Brain Project

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

import argparse
import json
import logging

from bluepyemodel.model.model_configurator import ModelConfigurator
from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.efeatures_extraction.targets_configurator import TargetsConfigurator
from targets import filenames, ecodes_metadata, targets, protocols_rheobase

logger = logging.getLogger()


def get_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Pipeline for Nexus",
    )

    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=[
            "configure_nexus",
            "configure_model_from_gene",
            "configure_model_from_json",
            "extract",
            "test_optimise",
            "test_analyze",
        ],
    )
    parser.add_argument("--emodel", type=str, required=True)
    parser.add_argument("--ttype", type=str, required=False)
    parser.add_argument("--etype", type=str, required=True)
    parser.add_argument("--mtype", type=str, required=False)
    parser.add_argument("--iteration_tag", type=str, default="test")
    parser.add_argument("--use_ipyparallel", action="store_true", default=True)
    parser.add_argument("--use_multiprocessing", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="count", dest="verbosity", default=1)
    parser.add_argument("--seed", type=int, required=False, default=1)

    return parser


def configure_targets(access_point):

    files_metadata = []
    for filename in filenames:
        files_metadata.append({"cell_name": filename, "ecodes": ecodes_metadata})

    targets_formated = []
    for ecode in targets:
        for amplitude in targets[ecode]["amplitudes"]:
            for efeature in targets[ecode]["efeatures"]:
                targets_formated.append(
                    {
                        "efeature": efeature,
                        "protocol": ecode,
                        "amplitude": amplitude,
                        "tolerance": 10.0,
                    }
                )

    configurator = TargetsConfigurator(access_point)
    configurator.new_configuration(files_metadata, targets_formated, protocols_rheobase)
    configurator.save_configuration()
    print(configurator.configuration.as_dict())


def store_pipeline_settings(access_point):

    pipeline_settings = EModelPipelineSettings(
        extraction_threshold_value_save=1,
        stochasticity=False,
        optimiser="SO-CMA",
        optimisation_params={"offspring_size": 20},
        optimisation_timeout=600.0,
        threshold_efeature_std=0.05,
        max_ngen=250,
        validation_threshold=10.0,
        optimisation_batch_size=10,
        max_n_batch=10,
        n_model=1,
        name_gene_map="Mouse_met_types_ion_channel_expression",
        plot_extraction=False,
        plot_optimisation=True,
        compile_mechanisms=True,
        name_Rin_protocol="IV_-40",
        name_rmp_protocol="IV_0",
        validation_protocols=["IDhyperpol_150"],
        morph_modifiers=[],
    )

    access_point.store_pipeline_settings(pipeline_settings)


def configure(pipeline):
    pipeline.access_point.access_point.deprecate_all(
        {"iteration": pipeline.access_point.emodel_metadata.iteration,
         "eModel": pipeline.access_point.emodel_metadata.emodel}
    )
    configure_targets(pipeline.access_point)
    store_pipeline_settings(pipeline.access_point)


def configure_model(pipeline, legacy_conf_json_file, morphology_name, morphology_format=None):
    configurator = ModelConfigurator(pipeline.access_point)
    configurator.new_configuration()

    with open(legacy_conf_json_file, "r") as f:
        parameters = json.load(f)

    configurator.configuration.init_from_legacy_dict(parameters, {"name": morphology_name, "format": morphology_format})
    configurator.save_configuration()
    print(configurator.configuration)


if __name__ == "__main__":

    args = get_parser().parse_args()

    species = "mouse"

    brain_region = "SSCX"

    morphology = "C060114A5"
    morphology_format = "asc" # specify the format of the morphology file, if not specified, the first availabe format will be used

    nexus_project = "" # specify the project name in nexus
    nexus_organisation = "" # specify the organisation name in nexus
    nexus_endpoint = "prod"
    forge_path = "./forge.yml"
    forge_ontology_path = "./nsg.yml"

    legacy_conf_json_file = "./../L5PC/config/params/pyr.json" # specify the path to the json file containing the model configuration if the model is not configured from gene data

    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[1],
        handlers=[logging.StreamHandler()],
    )

    pipeline = EModel_pipeline(
        emodel=args.emodel,
        ttype=args.ttype,
        iteration_tag=args.iteration_tag,
        species=species,
        brain_region=brain_region,
        forge_path=forge_path,
        forge_ontology_path=forge_ontology_path,
        nexus_organisation=nexus_organisation,
        nexus_project=nexus_project,
        nexus_endpoint=nexus_endpoint,
        use_ipyparallel=args.use_ipyparallel,
        use_multiprocessing=args.use_multiprocessing,
        mtype=args.mtype,
        etype=args.etype,
        data_access_point="nexus",
    )

    if args.step == "configure_nexus":
        configure(pipeline)
    elif args.step == "configure_model_from_gene":
        pipeline.configure_model(morphology_name=morphology, morphology_format=morphology_format, use_gene_data=True)
    elif args.step == "configure_model_from_json":
        configure_model(pipeline, legacy_conf_json_file, morphology, morphology_format)
    elif args.step == "extract":
        pipeline.extract_efeatures()
    elif args.step == "test_optimise":
        logger.warning(
            "test_optimise is only to check that the optimisation works. To "
            "optimise the models, please use launch_luigi.sh"
        )
        pipeline.optimise(seed=args.seed)
    elif args.step == "test_analyze":
        logger.warning(
            "test_analyze is only to check that the validation and storing "
            "of the models work. For real validation and storing, please use launch_luigi.sh"
        )
        pipeline.store_optimisation_results()
        pipeline.validation()
        pipeline.plot()
