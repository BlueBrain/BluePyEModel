import logging
import argparse
import pathlib

from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.recordings import FixedDtRecordingCustom
from bluepyemodel.emodel_pipeline.plotting import currentscape
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point

logger = logging.getLogger()


def get_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Pipeline for emodel creation'
    )

    parser.add_argument("--step", type=str, required=True, choices=[
        "extract", "optimize", "analyze", "validate", "currentscape"])
    parser.add_argument('--emodel', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--githash', type=str, required=False, default=None)
    parser.add_argument('--use_ipyparallel', action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="count", dest="verbosity", default=0)

    return parser


def currentscape_for_general_currents(pipeline, args, vars):
    """Plot currentscape for general currents for all protocols.
    
    Args:
        pipeline (EModel_Pipeline): the pipeline
        args (argparse.Namespace): arguments from the parser
        vars (list of str): the variables names of the currents.
            Should be present in Neuron.
            e.g. ["ik", "ina", "ica"]
    """
    # protocols to skip. Are not present for this particular example.
    preprots = ["RMPProtocol", "SearchHoldingCurrent", "RinProtocol", "SearchThresholdCurrent"]

    figures_dir = pathlib.Path("./figures") / args.emodel

    cell_evaluator = get_evaluator_from_access_point(
        pipeline.access_point,
        include_validation_protocols=True,
        use_fixed_dt_recordings=True,
        record_ions_and_currents=True,
    )

    # add recording for each protocol x new variable combination
    for prot in cell_evaluator.fitness_protocols["main_protocol"].protocols.values():
        if prot.name not in preprots:
            base_rec = prot.recordings[0]
            for var in vars:
                location = base_rec.location

                split_name = base_rec.name.split(".")
                split_name[-1] = var
                name = ".".join(split_name)

                # FixedDtRecordingCustom for fixed time steps.
                # Use LooseDtRecordingCustom for variable time steps
                new_rec = FixedDtRecordingCustom(name=name, location=location, variable=var)

                prot.recordings.append(new_rec)

    emodels = compute_responses(
        pipeline.access_point,
        cell_evaluator,
        pipeline.mapper,
        [args.seed],
        store_responses=True,
        load_from_local=False,
    )

    dest_leaf = "all"
    if not emodels:
        logger.warning("In plot_models, no emodel for %s", pipeline.access_point.emodel_metadata.emodel)
        return []

    # plot the currentscape figures and save them
    for mo in emodels:
        config = pipeline.access_point.pipeline_settings.currentscape_config
        figures_dir_currentscape = figures_dir / "currentscape" / dest_leaf
        currentscape(
            emodel=args.emodel,
            iteration_tag=args.githash,
            responses = mo.responses,
            config=config,
            metadata_str=mo.emodel_metadata.as_string(mo.seed),
            figures_dir=figures_dir_currentscape,
        )


def main():

    args = get_parser().parse_args()

    data_access_point = "local"
    recipes_path = './config/recipes.json'

    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[args.verbosity],
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[args.verbosity])

    pipeline = EModel_pipeline(
        emodel=args.emodel,
        data_access_point=data_access_point,
        recipes_path=recipes_path,
        iteration_tag=args.githash,
        use_ipyparallel=args.use_ipyparallel,
    )

    if args.step == "extract":
        pipeline.extract_efeatures()

    elif args.step == "optimize":
        pipeline.optimise(seed=args.seed)

    elif args.step == "analyze":
        pipeline.store_optimisation_results()
        pipeline.plot(only_validated=False)

    elif args.step == "validate":
        pipeline.validation()
        pipeline.plot(only_validated=False)

    elif args.step == "currentscape":
        logger.info(
            f"plot currentscape is {pipeline.access_point.pipeline_settings.plot_currentscape}"
        )

        vars = ["ik", "ina", "ica"]
        currentscape_for_general_currents(pipeline, args, vars)


if __name__ == "__main__":
    main()
