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

import logging
import argparse

from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline

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


def main():

    args = get_parser().parse_args()

    recipes_path = './config/recipes.json'

    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[args.verbosity],
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[args.verbosity])

    pipeline = EModel_pipeline(
        emodel=args.emodel,
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


if __name__ == "__main__":
    main()
