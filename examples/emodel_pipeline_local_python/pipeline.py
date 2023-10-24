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
import glob

from bluepyemodel.export_emodel.export_emodel import export_emodels_hoc
from bluepyemodel.export_emodel.export_emodel import export_emodels_sonata
from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from bluepyemodel.efeatures_extraction.targets_configurator import TargetsConfigurator
from targets import file_type, filenames, ecodes_metadata, targets, protocols_rheobase

logger = logging.getLogger()

def configure_targets(access_point):
    files_metadata = []

    if file_type == "ibw":
        fs = glob.glob(f"{filenames[0]}/*ch0*.ibw")
        for filename in fs:
            fn = filename.split("/")[-1]
            for ecode in ecodes_metadata:
                if ecode in fn:
                    files_metadata.append(
                            {
                                "cell_name": "cell1",
                                "filename": filename.split("/")[-1].split(".")[0],
                                "ecodes": {ecode: ecodes_metadata[ecode]},
                                "other_metadata": {
                                    "v_file": filename,
                                    # IMPORTANT: modify "ch1" with your number of channel in your ibw files
                                    "i_file": filename.replace("ch0", "ch1"),
                                    "i_unit": "A",
                                    "v_unit": "V",
                                    "t_unit": "s",
                                }
                            }
                        )
    elif file_type == "nwb":
        files_metadata = []
        for filename in filenames:
            files_metadata.append(
                {
                    "cell_name": filename.split("/")[-1].split(".")[0],
                    "filepath":filename,
                    "ecodes": ecodes_metadata
                }
            )
    else:
        raise ValueError(f"file type {file_type} is not supported in this current pipeline: "
                          "Expected 'ibw' or 'nwb'. "
                          "For other types of format please modify configure_targets "
                          "in pipeline.py.")

    targets_formated = []
    for ecode in targets:
        for amplitude in targets[ecode]['amplitudes']:
            for efeature in targets[ecode]["efeatures"]:
                protocol_name = ecode
                # remove ohmic_input_resistance_vb_ssse for IV_0 protocol (RMPProtocol). we only need voltage_base
                if not (protocol_name == "IV" and amplitude == 0 and efeature == "ohmic_input_resistance_vb_ssse"):
                    targets_formated.append({
                        "efeature": efeature,
                        "protocol": protocol_name,
                        "amplitude": amplitude,
                        "tolerance": 20
                    })

    if not files_metadata:
        raise FileNotFoundError("Cannot find electrophysiological experimental data. "
                        "Please provide them, with the correct path to them in targets.py.")

    configurator = TargetsConfigurator(access_point)
    configurator.new_configuration(files_metadata, targets_formated, protocols_rheobase)
    configurator.save_configuration()

def get_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Pipeline for emodel creation'
    )

    parser.add_argument("--step", type=str, required=True, choices=[
        "extract", "optimise", "analyse", "export_hoc", "export_sonata"])
    parser.add_argument('--emodel', type=str, required=True)
    parser.add_argument('--etype', type=str, required=False, default=None)
    parser.add_argument('--mtype', type=str, required=False, default=None)
    parser.add_argument('--ttype', type=str, required=False, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--githash', type=str, required=False, default=None)
    parser.add_argument('--use_ipyparallel', action="store_true", default=False)
    parser.add_argument('--use_multiprocessing', action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="count", dest="verbosity", default=0)

    return parser


def main():

    args = get_parser().parse_args()

    species = "rat"
    brain_region = "SSCX"

    recipes_path = './config/recipes.json'

    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[args.verbosity],
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[args.verbosity])

    pipeline = EModel_pipeline(
        emodel=args.emodel,
        etype=args.etype,
        mtype=args.mtype,
        ttype=args.ttype,
        species=species,
        brain_region=brain_region,
        recipes_path=recipes_path,
        iteration_tag=args.githash,
        use_ipyparallel=args.use_ipyparallel,
        use_multiprocessing=args.use_multiprocessing,
    )

    if args.step == "extract":
        configure_targets(pipeline.access_point)
        pipeline.extract_efeatures()

    elif args.step == "optimise":
        pipeline.optimise(seed=args.seed)

    elif args.step == "analyse":
        pipeline.store_optimisation_results()
        pipeline.validation()
        pipeline.plot(only_validated=False)

    elif args.step == "export_hoc":
        export_emodels_hoc(pipeline.access_point,
                           only_validated=False,
                           only_best=False,
                           seeds=[args.seed])

    elif args.step == "export_sonata":
        export_emodels_sonata(pipeline.access_point,
                           only_validated=False,
                           only_best=False,
                           seeds=[args.seed],
                           map_function=pipeline.mapper)

if __name__ == "__main__":
    main()
