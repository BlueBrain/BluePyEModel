import glob
import numpy
import json
import pathlib
from bluepyemodel.api.nexus import NexusAPI


def download_test_ephys():

    from kgforge.core import KnowledgeGraphForge
    import getpass

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

    token = getpass.getpass()

    nexus_forge = KnowledgeGraphForge(
        "https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/"
        "examples/notebooks/use-cases/prod-forge-nexus.yml",
        token=token,
        bucket="public/sscx",
    )

    p = nexus_forge.paths("Dataset")

    for f in file_names:

        resources = nexus_forge.search(
            p.type.id == "Trace",
            p.distribution.encodingFormat == "application/nwb",
            p.name == f,
            limit=1000,
        )

        nexus_forge.download(
            resources, "distribution.contentUrl", "./sscx_ephys_data", overwrite=True
        )


# download_test_ephys()

emodel = "L5PC"
species = "mouse"

access_point = NexusAPI(
    emodel=emodel,
    species=species,
    project="emodel_pipeline",
    organisation="Cells",
    endpoint="https://staging.nexus.ocp.bbp.epfl.ch/v1",
    forge_path=None,
)

# Deprecate all resources
for nexus_type in [
    "ElectrophysiologyFeatureOptimisationNeuronMorphology",
    "ElectrophysiologyFeatureExtractionTrace",
    "ElectrophysiologyFeatureExtractionTarget",
    "ElectrophysiologyFeatureOptimisationTarget",
    "ElectrophysiologyFeatureValidationTarget",
    "ElectrophysiologyFeatureOptimisationParameter",
    "ElectrophysiologyFeatureOptimisationChannelDistribution",
    "SubCellularModel",
    "ElectrophysiologyFeature",
    "ElectrophysiologyFeatureExtractionProtocol",
    "EModel",
]:
    access_point.deprecate({"type": nexus_type})

extraction_targets = {
    "IDrest": {
        "protocol_type": ["StepThresholdProtocol"],
        "used_for_rheobase": False,
        "used_for_optimization": True,
        "targets": [200],
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
        "targets": [0],
        "efeatures": ["Spikecount"],
    },
    "IV": {
        "protocol_type": ["StepThresholdProtocol", "RinProtocol", "RMPProtocol"],
        "used_for_rheobase": False,
        "used_for_optimization": True,
        "targets": [-100.0, -40.0, 0.0],
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

for ecode in extraction_targets:

    for target_amplitude, protocol_type in zip(
        extraction_targets[ecode]["targets"], extraction_targets[ecode]["protocol_type"]
    ):

        access_point.store_emodel_targets(
            ecode=ecode,
            efeatures=extraction_targets[ecode]["efeatures"],
            amplitude=target_amplitude,
            extraction_tolerance=20,
            protocol_type=protocol_type,
            used_for_extraction_rheobase=extraction_targets[ecode]["used_for_rheobase"],
            used_for_optimization=extraction_targets[ecode]["used_for_optimization"],
            used_for_validation=False,
        )

ton_toff = {
    "IV": {"ton": 20, "toff": 1020},
    "IDthresh": {"ton": 700, "toff": 2700},
    "IDrest": {"ton": 700, "toff": 2700},
}

for trace_path in glob.glob("./sscx_ephys_data/*.nwb"):

    for ecode in extraction_targets:

        file_metadata = {
            "ljp": 14.0,
            "protocol_name": ecode,
            "filepath": trace_path,
            "i_unit": "A",
            "v_unit": "V",
            "t_unit": "s",
        }

        file_metadata.update(ton_toff[ecode])

        access_point.store_recordings_metadata(
            cell_id=pathlib.Path(trace_path).stem,
            ecode=ecode,
            ephys_file_path=trace_path,
            recording_metadata=file_metadata,
        )

access_point.store_morphology(
    name="C060114A5",
    morphology_path="/gpfs/bbp.cscs.ch/project/proj38/home/damart/demo_BPEM/morphologies/C060114A5.asc",
)

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
            parameter_name=param["name"],
            value=param["val"],
            mechanism_name=param.get("mech", None),
            location=[location],
            distribution=param.get("dist", "constant"),
        )

distributions = {
    "exp": {"function": "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}"},
    "decay": {"function": "math.exp({distance}*{constant})*{value}", "parameters": ["constant"]},
}

for distribution in distributions:

    access_point.store_channel_distribution(
        name=distribution,
        function=distributions[distribution]["function"],
        parameters=distributions[distribution].get("parameters", []),
    )

for mech_path in glob.glob(
    "/gpfs/bbp.cscs.ch/project/proj38/home/damart/demo_BPEM/mechanisms/*.mod"
):

    name = pathlib.Path(mech_path).stem

    if "Stoch" in name:
        stochastic = True
    else:
        stochastic = False

    access_point.store_mechanism(name=name, mechanism_script_path=mech_path, stochastic=stochastic)
