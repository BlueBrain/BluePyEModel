"""Allows to execute the steps of the e-model building pipeline using python or CLI"""

import argparse
import datetime
import glob
import logging
import os
import pathlib

from bluepyemodel.access_point import get_db
from bluepyemodel.efeatures_extraction.efeatures_extraction import extract_save_features_protocols
from bluepyemodel.emodel_pipeline import plotting
from bluepyemodel.optimisation import setup_and_run_optimisation
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.validation.validation import validate

logger = logging.getLogger()


class EModel_pipeline:

    """EModel pipeline"""

    def __init__(
        self,
        emodel,
        species,
        brain_region,
        data_access_point,
        recipes_path=None,
        forge_path=None,
        githash=None,
        nexus_organisation=None,
        nexus_project=None,
        nexus_endpoint="staging",
        ttype=None,
        nexus_iteration_tag=None,
        use_ipyparallel=None,
    ):
        """Initialize the emodel_pipeline.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel under which the
                configuration data are stored.
            species (str): name of the species.
            brain_region (str): name of the brain region.
            data_access_point (str): name of the access_point used to access the data,
                can be "nexus" or "local".
                "local" expect the configuration to be  defined in a "config" directory
                containing recipes as in proj38. "nexus" expect the configuration to be defined
                on Nexus using NexusForge, see bluepyemodel/api/nexus.py.
            recipes_path (str): path of the recipes.json, only needed if the data access point is
                "local".
            forge_path (str): path to the .yml used to connect to Nexus Forge, only needed if
                db_api="nexus".
            githash (str): if provided, the pipeline will work in the directory
                working_dir/run/githash. Needed when continuing work or resuming optimisations.
            nexus_organisation (str): name of the Nexus organisation in which the project is
                located.
            nexus_project (str): name of the Nexus project to which the forge will connect to
                retrieve the data
            nexus_endpoint (str): Nexus endpoint ("prod" or "staging")
            ttype (str): name of the t-type. Required if using the gene expression or IC selector.
            nexus_iteration_tag (str): tag associated to the current run. Used to tag the
                Resources generated during the different run
            : should the parallelization map be base on ipyparallel.
        """

        self.emodel = emodel
        self.species = species
        self.brain_region = brain_region
        self.githash = githash

        self.mapper = instantiate_map_function(
            use_ipyparallel=use_ipyparallel, ipython_profil="IPYTHON_PROFILE"
        )

        self.access_point = self.init_access_point(
            data_access_point,
            recipes_path,
            forge_path,
            nexus_organisation,
            nexus_endpoint,
            nexus_project,
            ttype,
            nexus_iteration_tag,
        )

    def init_access_point(
        self,
        data_access_point,
        recipes_path,
        forge_path,
        nexus_organisation,
        nexus_endpoint,
        nexus_project,
        ttype,
        nexus_iteration_tag,
    ):
        """Instantiate a data access point, either to Nexus or GPFS"""

        endpoint = None
        if nexus_endpoint == "prod":
            endpoint = "https://bbp.epfl.ch/nexus/v1"
        elif nexus_endpoint == "staging":
            endpoint = "https://staging.nexus.ocp.bbp.epfl.ch/v1"

        emodel_dir = "./"
        if self.githash:
            emodel_dir = str(pathlib.Path("./") / "run" / self.githash)

        return get_db(
            access_point=data_access_point,
            emodel=self.emodel,
            emodel_dir=emodel_dir,
            recipes_path=recipes_path,
            final_path="final.json",
            species=self.species,
            brain_region=self.brain_region,
            organisation=nexus_organisation,
            project=nexus_project,
            endpoint=endpoint,
            forge_path=forge_path,
            ttype=ttype,
            iteration_tag=nexus_iteration_tag,
        )

    def extract_efeatures(self):
        """"""

        return extract_save_features_protocols(
            emodel=self.emodel, access_point=self.access_point, mapper=self.mapper
        )

    def optimize(self, seed=1, continue_opt=False):
        """"""

        setup_and_run_optimisation(
            self.access_point,
            emodel=self.emodel,
            seed=seed,
            mapper=self.mapper,
            continue_opt=continue_opt,
            githash=self.githash,
            terminator=None,
        )

    def store_optimisation_results(self):
        """"""

        for chkp_path in glob.glob("./checkpoints/*.pkl"):

            if self.emodel not in chkp_path:
                continue

            if self.githash and self.githash not in chkp_path:
                continue

            splitted_path = pathlib.Path(chkp_path).stem.split("__")
            seed = int(splitted_path[-1])

            store_best_model(
                access_point=self.access_point,
                emodel=self.emodel,
                seed=seed,
                checkpoint_path=chkp_path,
                githash=self.githash,
            )

    def validation(self):
        """"""

        validate(
            access_point=self.access_point,
            emodel=self.emodel,
            mapper=self.mapper,
        )

    def plot(self):

        for chkp_path in glob.glob("./checkpoints/*.pkl"):

            if self.emodel not in chkp_path:
                continue

            if self.githash and self.githash not in chkp_path:
                continue

            plotting.optimization(
                checkpoint_path=chkp_path,
                figures_dir=pathlib.Path("./figures") / self.emodel / "optimisation",
            )

        githashs = None
        if self.githash:
            githashs = [self.githash]

        return plotting.plot_models(
            self.access_point,
            self.emodel,
            mapper=self.mapper,
            seeds=None,
            githashs=githashs,
            figures_dir=pathlib.Path("./figures") / self.emodel,
            plot_distributions=True,
            plot_scores=True,
            plot_traces=True,
        )


def get_parser():
    """Instantiate a parser that can configure the steps of the E-Model pipeline"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="E-Model pipeline"
    )

    # Arguments that define the E-Model:
    parser.add_argument("--emodel", type=str, help="Name of the e-model")
    parser.add_argument("--species", type=str, help="Name of the species")
    parser.add_argument("--brain_region", type=str, help="Name of the brain region")

    # Arguments used in the instantiation of the data access point:
    parser.add_argument(
        "--data_access_point",
        type=str,
        required=True,
        choices=["local", "nexus"],
        help="Which data access point to use.",
    )
    parser.add_argument("--recipes_path", type=str, help="Relative path to the recipes.json.")
    parser.add_argument(
        "--nexus_organisation",
        type=str,
        help="Name of the organisation to which the Nexus project belong.",
    )
    parser.add_argument("--nexus_project", type=str, help="Name of the Nexus project.")
    parser.add_argument(
        "--nexus_endpoint",
        type=str,
        choices=["prod", "staging"],
        help="Name of the Nexus endpoint.",
    )
    parser.add_argument(
        "--nexus_iteration_tag",
        type=str,
        default=None,
        help="Tag associated to the current run, used to tag the Resources "
        "generated during the different run",
    )
    parser.add_argument(
        "--forge_path",
        type=str,
        default=None,
        help="Path to the .yml used to configure Nexus Forge.",
    )
    parser.add_argument("--ttype", type=str, help="Name of the t-type.")

    # Argument used for the steps of the pipeline:
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["extract", "optimize", "store", "plot", "validate"],
        help="Stage of the E-Model building pipeline to execute",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed to use for optimization")
    parser.add_argument(
        "--githash",
        type=str,
        required=False,
        default=None,
        help="Githash associated to the current E-Model " "building iteration.",
    )
    parser.add_argument(
        "--use_ipyparallel", action="store_true", default=False, help="Use ipyparallel"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="-v for INFO, -vv for DEBUG",
    )

    return parser


def get_arguments():
    """Get the arguments and check their validity"""

    args = get_parser().parse_args()

    if args.verbosity > 2:
        raise Exception("Verbosity cannot be more verbose than -vv")

    if args.data_access_point == "local":
        required_args = ["recipes_path"]
    elif args.data_access_point == "nexus":
        required_args = [
            "nexus_organisation",
            "nexus_project",
            "nexus_endpoint",
            "nexus_iteration_tag",
        ]

    for arg in required_args:
        if getattr(args, arg) is None:
            raise Exception(
                "When using %s as a data access point. The argument "
                "%s has to be informed." % (args.data_access_point, arg)
            )

    return args


def instantiate_map_function(use_ipyparallel=False, ipython_profil="IPYTHON_PROFILE"):
    """Instantiate a map function"""

    if use_ipyparallel:
        from ipyparallel import Client

        rc = Client(profile=os.getenv(ipython_profil))
        lview = rc.load_balanced_view()

        def mapper(func, it):
            start_time = datetime.datetime.now()
            ret = lview.map_sync(func, it)
            logger.debug("Took %s", datetime.datetime.now() - start_time)
            return ret

    else:
        mapper = map

    return mapper


def sanitize_gitignore():
    """In order to avoid git issue when archiving the current working directory,
    adds the following lines to .gitignore: 'run/', 'checkpoints/', 'figures/',
    'logs/', '.ipython/', '.ipynb_checkpoints/'"""

    path_gitignore = pathlib.Path("./.gitignore")

    if not (path_gitignore.is_file()):
        raise Exception("Could not update .gitignore as it does not exist.")

    with open(str(path_gitignore), "r") as f:
        lines = f.readlines()

    lines = " ".join(line for line in lines)

    to_add = ["run/", "checkpoints/", "figures/", "logs/", ".ipython/", ".ipynb_checkpoints/"]

    with open(str(path_gitignore), "a") as f:
        for a in to_add:
            if a not in lines:
                f.write(f"{a}\n")


def main():
    """This function can be used to run the different steps of an e-model building pipeline.
    It can also be used as an starting point to build your own."""

    args = get_arguments()

    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[args.verbosity],
        handlers=[logging.StreamHandler()],
    )

    pipeline = EModel_pipeline(
        emodel=args.emodel,
        species=args.species,
        brain_region=args.brain_region,
        data_access_point=args.data_access_point,
        recipes_path=args.recipes_path,
        forge_path=args.forge_path,
        githash=args.githash,
        nexus_organisation=args.nexus_organisation,
        nexus_project=args.nexus_project,
        nexus_endpoint=args.nexus_endpoint,
        ttype=args.ttype,
        nexus_iteration_tag=args.nexus_iteration_tag,
        use_ipyparallel=args.use_ipyparallel,
    )

    if args.step == "extract":
        pipeline.extract_efeatures()
    elif args.step == "optimize":
        pipeline.optimize(seed=args.seed)
    elif args.step == "store":
        pipeline.store_optimisation_results()
    elif args.step == "validate":
        pipeline.validation()
    elif args.step == "plot":
        pipeline.plot()


if __name__ == "__main__":
    main()
