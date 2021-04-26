import glob
import pathlib
import argparse
import logging

from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from bluepyemodel.emodel_pipeline.emodel_pipeline import plotting


def get_logger():

    logger = logging.getLogger()

    logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])

    return logger


def get_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="Pipeline for proj72"
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["optimize", "analyse", "extraction", "validation"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1)

    return parser


def optimize(emodel, pipeline, optimizer, opt_params, max_ngen):

    chkp_path = "./checkpoints/checkpoint__{}__{}.pkl".format(emodel, opt_params["seed"])

    pipeline.optimize(
        max_ngen=max_ngen,
        stochasticity=False,
        opt_params=opt_params,
        optimizer=optimizer,
        checkpoint_path=chkp_path,
        timeout=600,
    )


def analyse(emodel, pipeline, optimizer, opt_params):

    figure_dir = "./figures/{}/".format(emodel)

    for chkp_path in glob.glob("./checkpoints/checkpoint__{}*.pkl".format(emodel)):

        splitted_path = pathlib.Path(chkp_path).stem.split("__")
        opt_params["seed"] = int(splitted_path[-1])

        pipeline.store_best_model(
            stochasticity=False,
            opt_params=opt_params,
            optimizer=optimizer,
            checkpoint_path=chkp_path,
        )

        plotting.optimization(chkp_path, figure_dir, emodel=emodel, githash=None)

    pipeline.plot_models(figures_dir=figure_dir)


def main():

    logger = get_logger()
    args = get_parser().parse_args()

    emodel = "L5PC"
    species = "mouse"
    db_api = "nexus"
    working_dir = pathlib.Path("./")

    optimizer = "MO-CMA"
    max_ngen = 5
    opt_params = {"offspring_size": 5, "weight_hv": 0.4, "seed": args.seed}

    pipeline = EModel_pipeline(emodel=emodel, species=species, db_api=db_api)

    if args.stage == "extraction":
        pipeline.extract_efeatures(plot=True)

    elif args.stage == "optimize":
        optimize(emodel, pipeline, optimizer, opt_params, max_ngen)

    elif args.stage == "analyse":
        analyse(emodel, pipeline, optimizer, opt_params)

    elif args.stage == "validation":
        pipeline.validate()


if __name__ == "__main__":

    main()
