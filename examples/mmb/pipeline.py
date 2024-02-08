import argparse
import json
import logging

from bluepyemodel.model.model_configurator import ModelConfigurator
from bluepyemodelnexus.emodel_pipeline import EModel_pipeline_nexus
from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.efeatures_extraction.targets_configurator import TargetsConfigurator
from targets import filenames, ecodes_metadata, targets, protocols_rheobase

logger = logging.getLogger()


ECODES = [
    "IDthresh", "IDhyperpol", "IV", "IDrest", "IDRest",
    "IDThresh", "IDThres", "APWaveform", "APwaveform"
]

def get_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Pipeline for MMB'
    )

    parser.add_argument(
        "--step", type=str, required=True,
        choices=[
            "configure_nexus",
            "configure_model_from_gene",
            "configure_model_from_json",
            "extract",
            "test_optimise",
            "test_analyze"]
    )
    parser.add_argument('--emodel', type=str, required=True)
    parser.add_argument('--ttype', type=str, required=False)
    parser.add_argument('--etype', type=str, required=True)
    parser.add_argument('--mtype', type=str, required=False)
    parser.add_argument('--iteration_tag', type=str, default="test")
    parser.add_argument("-v", "--verbose", action="count", dest="verbosity", default=1)
    parser.add_argument('--use_ipyparallel', action="store_true", default=True)
    parser.add_argument('--use_multiprocessing', action="store_true", default=False)
    parser.add_argument('--seed', type=int, required=False, default=1)

    return parser


def configure_targets(access_point):

    files_metadata = []
    for filename in filenames:
        files_metadata.append({"cell_name": filename, "ecodes": ecodes_metadata})

    targets_formated = []
    for ecode in targets:
        for amplitude in targets[ecode]['amplitudes']:
            for efeature in targets[ecode]["efeatures"]:
                protocol_name = ecode
                if protocol_name in ECODES:
                    if protocol_name in ["IDThresh", "IDThres"]:
                        protocol_name = "IDthresh"
                    if protocol_name == "IDRest":
                        protocol_name = "IDrest"
                    if protocol_name == "APwaveform":
                        protocol_name = "APWaveform"
                targets_formated.append({
                    "efeature": efeature,
                    "protocol": protocol_name,
                    "amplitude": amplitude,
                    "tolerance": 20.
                })

    configurator = TargetsConfigurator(access_point)
    configurator.new_configuration(files_metadata, targets_formated, protocols_rheobase)
    configurator.save_configuration()
    print(configurator.configuration.as_dict())


def store_pipeline_settings(access_point):

    pipeline_settings = EModelPipelineSettings(
        stochasticity=False,
        optimiser="SO-CMA",
        optimisation_params={"offspring_size": 20},
        optimisation_timeout=100.,
        threshold_efeature_std=0.1,
        max_ngen=20,
        validation_threshold=15.,
        plot_extraction=True,
        plot_optimisation=True,
        compile_mechanisms=True,
        name_Rin_protocol="IV_-40",
        name_rmp_protocol="IV_0",
        efel_settings={"strict_stiminterval": True, "Threshold": -20, "interp_step": 0.025},
        strict_holding_bounds=True,
        validation_protocols= ["IDhyperpol_150"],
        morph_modifiers=[],
        n_model=1,
        optimisation_batch_size=3,
        max_n_batch=1,
    )

    access_point.store_pipeline_settings(pipeline_settings)


def configure(pipeline):
    pipeline.access_point.access_point.deprecate_all(
        {"iteration": pipeline.access_point.emodel_metadata.iteration}
    )
    configure_targets(pipeline.access_point)
    store_pipeline_settings(pipeline.access_point)


def configure_model(pipeline, morphology_name):
    configurator = ModelConfigurator(pipeline.access_point)
    configurator.new_configuration()

    filename = "/gpfs/bbp.cscs.ch/project/proj136/placeholder_singlecell/optimisation_local_luigi/config/params/pyr.json"

    with open(filename,'r') as f:
        parameters = json.load(f)

    configurator.configuration.init_from_legacy_dict(parameters, {"name": morphology_name})
    configurator.save_configuration()
    print(configurator.configuration)


if __name__ == "__main__":

    args = get_parser().parse_args()

    species = "mouse"

    brain_region = "SSCX"

    morphology = "cylindrical_morphology_19.1399"

    nexus_project = "mmb-emodels-for-synthesized-neurons"
    nexus_organisation = "bbp"
    nexus_endpoint = "prod"
    forge_path = "./forge.yml"
    forge_ontology_path = "./nsg.yml"

    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[args.verbosity],
        handlers=[logging.StreamHandler()],
    )

    pipeline = EModel_pipeline_nexus(
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
    )

    if args.step == "configure_nexus":
        configure(pipeline)
    elif args.step == "configure_model_from_gene":
        pipeline.configure_model(morphology, use_gene_data=True)
    elif args.step == "configure_model_from_json":
        configure_model(pipeline, morphology)
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
