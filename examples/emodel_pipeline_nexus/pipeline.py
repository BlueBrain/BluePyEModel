import glob
import pathlib
import argparse
import json
import sys
import logging
import time

from bluepyemodel.access_point.nexus import NexusAccessPoint
from bluepyemodel.access_point.forge_access_point import NexusForgeAccessPoint
from kgforge.core import Resource

from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from bluepyemodel.emodel_pipeline.emodel_pipeline import plotting

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)


def get_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="Example pipeline"
    )

    parser.add_argument(
        "--step",
        type=str,
        choices=["upload_data", "configure", "optimize", "store", "plot", "extract", "validate"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1)

    return parser


targets = {
    "IDrest": {
        "protocol_type": ["StepThresholdProtocol"],
        "used_for_rheobase": False,
        "used_for_optimization": True,
        "amplitudes": [200.0],
        "efeatures": [
            "Spikecount",
            "mean_frequency",
            "voltage_base",
            "burst_number",
            "inv_time_to_first_spike",
            "AP_amplitude",
            "APlast_amp",
            "AP_begin_voltage",
            "AHP_depth",
            "ISI_CV",
            "ISI_log_slope",
            "fast_AHP",
            "adaptation_index2",
            "AHP_slow_time",
            "doublet_ISI",
            "decay_time_constant_after_stim",
        ],
    },
    "IDthresh": {
        "used_for_rheobase": True,
        "used_for_optimization": False,
        "protocol_type": [""],
        "amplitudes": [0.0],
        "efeatures": ["Spikecount"],
    },
    "IV": {
        "protocol_type": ["StepThresholdProtocol", "RinProtocol", "RMPProtocol"],
        "used_for_rheobase": False,
        "used_for_optimization": True,
        "amplitudes": [-100.0, -40.0, 0.],
        "efeatures": [
            "Spikecount",
            "voltage_base",
            "decay_time_constant_after_stim",
            "steady_state_voltage_stimend",
            "voltage_deflection",
            "voltage_deflection_begin",
            "ohmic_input_resistance_vb_ssse",
        ],
    },
}


def store_morphology(access_point):
    access_point.store_morphology("C060114A5")


def store_distribution(access_point):

    distributions = {
        "exp": {"function": "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}"},
        "decay": {
            "function": "math.exp({distance}*{constant})*{value}",
            "parameters": ["constant"],
        },
    }

    for distribution in distributions:

        access_point.store_channel_distribution(
            name=distribution,
            function=distributions[distribution]["function"],
            parameters=distributions[distribution].get("parameters", []),
        )


def store_parameters(access_point):

    model_parameters = {
        "global": [{"name": "v_init", "val": -80}, {"name": "celsius", "val": 34}],
        "distribution_decay": [{"name": "constant", "val": [-0.1, 0.0]}],
        "myelinated": [{"name": "cm", "val": 0.02}],
        "all": [
            {"name": "Ra", "val": 100},
            {"name": "g_pas", "val": [1e-5, 6e-5], "mech": "pas"},
            {"name": "e_pas", "val": [-95, -60], "mech": "pas"},
        ],
        "somadend": [{"name": "gIhbar_Ih", "val": [0, 2e-4], "dist": "exp", "mech": "Ih"}],
        "axonal": [
            {"name": "cm", "val": 1},
            {"name": "ena", "val": 50},
            {"name": "ek", "val": -90},
            {"name": "vshifth_NaTg", "val": 10, "mech": "NaTg"},
            {"name": "slopem_NaTg", "val": 9, "mech": "NaTg"},
            {"name": "gNaTgbar_NaTg", "val": [0, 1.5], "mech": "NaTg"},
            {"name": "gNap_Et2bar_Nap_Et2", "val": [0, 0.02], "mech": "Nap_Et2"},
            {"name": "gK_Pstbar_K_Pst", "val": [0, 1], "mech": "K_Pst"},
            {"name": "gK_Tstbar_K_Tst", "val": [0, 0.2], "mech": "K_Tst"},
            {"name": "gSKv3_1bar_SKv3_1", "val": [0, 1], "mech": "SKv3_1"},
            {"name": "gCa_HVAbar_Ca_HVA2", "val": [0, 0.001], "mech": "Ca_HVA2"},
            {"name": "gCa_LVAstbar_Ca_LVAst", "val": [0, 0.01], "mech": "Ca_LVAst"},
            {"name": "gSK_E2bar_SK_E2", "val": [0, 0.1], "mech": "SK_E2"},
            {"name": "decay_CaDynamics_DC0", "val": [20, 300], "mech": "CaDynamics_DC0"},
            {"name": "gamma_CaDynamics_DC0", "val": [0.005, 0.05], "mech": "CaDynamics_DC0"},
        ],
        "somatic": [
            {"name": "cm", "val": 1},
            {"name": "ena", "val": 50},
            {"name": "ek", "val": -90},
            {"name": "vshiftm_NaTg", "val": 13, "mech": "NaTg"},
            {"name": "vshifth_NaTg", "val": 15, "mech": "NaTg"},
            {"name": "slopem_NaTg", "val": 7, "mech": "NaTg"},
            {"name": "gNaTgbar_NaTg", "val": [0, 0.3], "mech": "NaTg"},
            {"name": "gK_Pstbar_K_Pst", "val": [0, 0.2], "mech": "K_Pst"},
            {"name": "gK_Tstbar_K_Tst", "val": [0, 0.1], "mech": "K_Tst"},
            {"name": "gSKv3_1bar_SKv3_1", "val": [0, 1], "mech": "SKv3_1"},
            {"name": "gCa_HVAbar_Ca_HVA2", "val": [0, 0.001], "mech": "Ca_HVA2"},
            {"name": "gCa_LVAstbar_Ca_LVAst", "val": [0, 0.01], "mech": "Ca_LVAst"},
            {"name": "gSK_E2bar_SK_E2", "val": [0, 0.1], "mech": "SK_E2"},
            {"name": "decay_CaDynamics_DC0", "val": [20, 300], "mech": "CaDynamics_DC0"},
            {"name": "gamma_CaDynamics_DC0", "val": [0.005, 0.05], "mech": "CaDynamics_DC0"},
        ],
        "apical": [
            {"name": "cm", "val": 2},
            {"name": "ena", "val": 50},
            {"name": "ek", "val": -90},
            {"name": "gamma_CaDynamics_DC0", "val": [0.005, 0.05], "mech": "CaDynamics_DC0"},
            {"name": "vshiftm_NaTg", "val": 6, "mech": "NaTg"},
            {"name": "vshifth_NaTg", "val": 6, "mech": "NaTg"},
            {"name": "gNaTgbar_NaTg", "val": [0, 0.1], "dist": "decay", "mech": "NaTg"},
            {"name": "gSKv3_1bar_SKv3_1", "val": [0, 0.003], "mech": "SKv3_1"},
            {"name": "gCa_HVAbar_Ca_HVA2", "val": [0, 0.0001], "mech": "Ca_HVA2"},
            {"name": "gCa_LVAstbar_Ca_LVAst", "val": [0, 0.001], "mech": "Ca_LVAst"},
        ],
        "basal": [
            {"name": "cm", "val": 2},
            {"name": "gamma_CaDynamics_DC0", "val": [0.005, 0.05], "mech": "CaDynamics_DC0"},
            {"name": "gCa_HVAbar_Ca_HVA2", "val": [0, 0.0001], "mech": "Ca_HVA2"},
            {"name": "gCa_LVAstbar_Ca_LVAst", "val": [0, 0.001], "mech": "Ca_LVAst"},
        ],
    }

    for location in model_parameters:

        for param in model_parameters[location]:

            access_point.store_optimisation_parameter(
                name=param["name"],
                value=param["val"],
                mechanism_name=param.get("mech", None),
                location=location,
                distribution=param.get("dist", "constant"),
                auto_handle_mechanism=True,
            )


def extraction_metadata(access_point):

    ton_toff = {
        "IV": {"ton": 20, "toff": 1020},
        "IDthresh": {"ton": 700, "toff": 2700},
        "IDrest": {"ton": 700, "toff": 2700},
    }

    for trace_path in glob.glob("./sscx_ephys_data/*.nwb"):

        for ecode in targets:

            file_metadata = {
                "ljp": 14.0,
                "protocol_name": ecode,
                "filepath": trace_path,
                "i_unit": "A",
                "v_unit": "V",
                "t_unit": "s",
            }

            file_metadata.update(ton_toff[ecode])

            access_point.store_trace_metadata(
                name=pathlib.Path(trace_path).stem,
                ecode=ecode,
                recording_metadata=file_metadata,
            )


def store_extraction_targets(access_point):

    for ecode, target in targets.items():
        for amp, type_ in zip(target["amplitudes"], target["protocol_type"]):
            access_point.store_emodel_targets(
                ecode=ecode,
                efeatures=target["efeatures"],
                amplitude=amp,
                extraction_tolerance=20,
                protocol_type=type_,
                used_for_extraction_rheobase=target["used_for_rheobase"],
                used_for_optimization=target["used_for_optimization"],
                used_for_validation=False,
            )


def download_traces():

    file_names = [
        "C060109A1-SR-C1",
        "C060109A2-SR-C1",
        "C060109A3-SR-C1",
        "C060110A2-SR-C1",
        "C060110A3-SR-C1",
        "C060110A5-SR-C1",
        "C060112A1-SR-C1",
        "C060112A3-SR-C1",
        "C060112A4-SR-C1",
        "C060112A6-SR-C1",
        "C060112A7-SR-C1",
        "C060114A2-SR-C1",
        "C060114A4-SR-C1",
        "C060114A5-SR-C1",
        "C060114A6-SR-C1",
        "C060114A7-SR-C1",
        "C060116A1-SR-C1",
        "C060116A3-SR-C1",
        "C060116A4-SR-C1",
        "C060116A5-SR-C1",
        "C060116A6-SR-C1",
        "C060116A7-SR-C1",
        "C060202A1-SR-C1",
        "C060202A2-SR-C1",
        "C060202A4-SR-C1",
        "C060202A5-SR-C1",
        "C060202A6-SR-C1",
        "C060209A3-SR-C1",
    ]

    nexus_access_point = NexusForgeAccessPoint(
        project="sscx",
        organisation="public",
        endpoint="https://bbp.epfl.ch/nexus/v1",
        forge_path="https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml",
        cross_bucket=False,
    )

    p = nexus_access_point.forge.paths("Dataset")

    for f in file_names:

        resources = nexus_access_point.forge.search(
            p.type.id == "Trace",
            p.distribution.encodingFormat == "application/nwb",
            p.name == f,
            limit=1000,
        )

        nexus_access_point.forge.download(
            resources, "distribution.contentUrl", "./sscx_ephys_data", overwrite=True
        )


def upload_data(pipeline):

    for nexus_type in [
        "SubCellularModelScript",
        "NeuronMorphology",
        "Trace",
    ]:
        print(f"Deprecating {nexus_type}...")
        pipeline.access_point.access_point.deprecate({"type": nexus_type}, False)

    print(f"Downloading traces from Nexus public/sscx...")
    download_traces()

    forge = pipeline.access_point.access_point.forge

    print(f"Registering Traces...")
    for trace_file in glob.glob("./sscx_ephys_data/*.nwb"):
        distribution = forge.attach(trace_file)
        name = pathlib.Path(trace_file).stem
        resource = Resource(type="Trace", name=name, distribution=distribution)
        forge.register(resource)

    print(f"Registering mechanisms...")
    for mod_files in glob.glob("../emodel_pipeline_local_python/mechanisms/*.mod"):
        distribution = forge.attach(mod_files)
        name = pathlib.Path(mod_files).stem
        resource = Resource(type="SubCellularModelScript", name=name, distribution=distribution)
        forge.register(resource)

    print(f"Registering morphology...")
    morphology_path = "../emodel_pipeline_local_python/morphologies/C060114A5.asc"
    name = pathlib.Path(morphology_path).stem
    distribution = forge.attach(morphology_path)
    resource = Resource(type="NeuronMorphology", name=name, distribution=distribution)
    forge.register(resource)


def store_pipeline_settings(access_point):

    access_point.store_pipeline_settings(
        extraction_threshold_value_save=1,
        efel_settings=None,
        stochasticity=False,
        morph_modifiers=None,
        optimizer="MO-CMA",
        optimisation_params={"offspring_size": 4},
        optimisation_timeout=300.0,
        threshold_efeature_std=0.05,
        max_ngen=4,
        validation_threshold=5.0,
        optimization_batch_size=3,
        max_n_batch=3,
        n_model=1,
        plot_extraction=True,
        plot_optimisation=True,
        additional_protocols=None,
        compile_mechanisms=True,
    )


def configure(pipeline):

    print(f"Deprecating project...")
    pipeline.access_point.deprecate_project(use_version=False)
    print(f"Storing morphology...")
    store_morphology(pipeline.access_point)
    print(f"Storing parameter distribution...")
    store_distribution(pipeline.access_point)
    print(f"Storing parameters...")
    store_parameters(pipeline.access_point)
    print(f"Storing targets...")
    store_extraction_targets(pipeline.access_point)
    print(f"Storing ephys files metadata...")
    extraction_metadata(pipeline.access_point)
    print(f"Storing pipeline settings...")
    store_pipeline_settings(pipeline.access_point)


if __name__ == "__main__":

    args = get_parser().parse_args()

    emodel = "L5PC"
    species = "mouse"
    brain_region = "SSCX"

    data_access_point = "nexus"
    nexus_project = "emodel_pipeline"
    nexus_organisation = "Cells"
    nexus_endpoint = "staging"
    forge_path = "forge.yml"
    iteration_tag = "v0"

    pipeline = EModel_pipeline(
        emodel=emodel,
        species=species,
        brain_region=brain_region,
        data_access_point=data_access_point,
        forge_path=forge_path,
        nexus_organisation=nexus_organisation,
        nexus_project=nexus_project,
        nexus_endpoint=nexus_endpoint,
        iteration_tag=iteration_tag,
    )

    if args.step == "upload_data":
        upload_data(pipeline)

    elif args.step == "configure":
        configure(pipeline)

    elif args.step == "extract":
        pipeline.extract_efeatures()

    elif args.step == "optimize":
        pipeline.optimize()

    elif args.step == "store":
        pipeline.store_optimisation_results()

    elif args.step == "validate":
        pipeline.validation()

    elif args.step == "plot":
        pipeline.plot()
