# BluePyEModel: Blue Brain Python E-Model Building Library


## Introduction

The Blue Brain Python E-Model Building Library (BluePyEModel) is a Python package dedicated to simplify the configuration and execution of E-Model building tasks. It covers tasks such as feature extraction, model optimisation, validation, model management and AIS synthesis. As such, it builds on top of BluePyEfe, BluePyOpt and BluePyMM.


## To get started with the E-Model building pipeline

This part will talk specificly about the E-Model building pipeline which for now encapsulate feature extraction, optimisation and model analysis. For model management and AIS synthesis, documentation is not available yet.

Despite the presence of the following explanation, E-Model building pipeline is not a trivial process, therefore, do not hesitate to contact the Cells team for help to get you setup (tanguy.damart@epfl.ch).


### Running the pipeline

The E-Model building pipeline can be executed either step by step using Python or all at once as a Luigi workflow.

#### Running the pipeline with Python/CLI

To run the pipeline using Python, you will need to use the class EModel_pipeline located in bluepyemodel/emodel_pipeline/emodel_pipeline.py. An example of the use of this class can be seen in the function `main` of the same file. To start, we recommend that you copy the inside of this function `main` or directly call the file bluepyemodel/emodel_pipeline/emodel_pipeline.py using the command line. In the latter case, please run the command `python emodel_pipeline.py --help` for a list of expected arguments.

The pipeline is divided in 4 steps:
- extraction: extracts efeatures from the ephys recordings and avergages the results along the requested targets.
- optimisation: builds a NEURON cell model and optimizes its parameters using as target the efeatures computed during efeature extraction.
- storage of the model: reads the results of the extraction and stores the models (best set of parameters) in local or on Nexus.
- validation: reads the models and runs the optimisation protocols and/or validation protocols on them. The efeature scores obtained on these protocols are then passed to a validation function that decides if the model is good enough.
- plotting: reads the models and runs the optimisation protocols and/or validation protocols on them. Then, plots the resulting traces along the efeature scores and parameter distributions.

These four steps are to be run in order as, for example, validation cannot be run if no models have been stored.

#### Running the pipeline with Luigi

To run the pipeline with luigi, you will need a luigi.cfg file. This file will contain all the arguments for the pipeline, similar to the ones that were provided when running with Python.

Here is a template that can be used as a starting point:
```
[DEFAULT]
account=proj38
virtual-env=PATH_TO_CURRENT_VENV
module_archive=archive/2020-11
workflows-sync=/gpfs/bbp.cscs.ch/home/${USER}/workflows
workers=10
chdir=PATH_TO_CURRENT_WORKING_DIRECTORY
time=24:00:00
enable-internet: True

[BB5-WORKER]
exclusive: True
mem: 0
nodes: 4
enable-internet: True

[core]
log_level=INFO

[parallel]
backend = ipyparallel

[EmodelAPIConfig]
api=nexus
species=mouse
brain_region=SSCX
forge_path=./forge.yml
nexus_poject=emodel_pipeline
nexus_organisation=Cells
nexus_endpoint=https://staging.nexus.ocp.bbp.epfl.ch/v1
nexus_iteration_tag=v0
ttype=_
```

The pipeline can then be run using the command:
`bbp-workflow launch-bb5 -f --config=luigi.cfg bluepyemodel.tasks.emodel_creation.optimisation EModelCreation emodel=L5PC species=mouse brain-region=SSCX`

### The data access points

To access the data needed during the E-Model building process such as settings, ephys data, morphology and more, the E-Model building pipeline relies on a data access point. This role of this access point is to retrieve the data from local storage or from Nexus and put it in the format used by the different steps.

There are two types of data access points: "local" and "nexus".

#### The local data access point

The local data access point retrieves and stores data from local storage and, for most operations, from the current working directory.

To use the local data access point, you first need to create a recipes.json, which will contain, for each e-model, a recipe describing how to build it. Here is an example of a recipe for a fictitious L5PC model:
```
{ 
    "L5PC": {
        "morph_path": "morphologies/",
        "morphology": [["L5TPC","FILENAME.asc"]],
        "params": "config/params/pyr.json",
        "protocol": "config/protocols/L5PC.json",
        "features": "config/features/L5PC.json",
        "pipeline_settings": {
            "optimisation_timeout": 300,
            "optimizer": "MO-CMA",
            "optimisation_params": {
                "offspring_size": 20
            }
        }
    }
}
```

TO FINISH

#### The Nexus data access point

TO COME


Pipeline settings:
-----------------

The settings of the pipeline are set as follows:
- When using the 'local' data access point, settings are to be informed in the field "pipeline_settings" of the recipes for each emodel independently.
- When using the 'nexus' data access point, settings have to be registered as a Resource of type PipelineSettings. This can be done using the function store_pipeline_settings from bluepyemodel.access_point.nexus.

List of settings:

extraction_threshold_value_save (int, default: 1): during extraction, minimum number of values needed for an efeatures to be returned in the output.
plot_extraction (bool, default: True): should the e-features and traces be plotted at the end of the e-features extraction. Can be lengthly if their is a lot of data.
efel_settings (dict, default: {'interp_step': 0.025, 'strict_stiminterval': True}): eFEl settings used during efeatures extraction and optimisation. If settings are also informed per e-feature, the latter will have priority.
stochasticity (bool, default: False): should stochasticity be enabled for the the channels that can be stochastic.
morph_modifiers (list, default: replace_axon_with_taper): List of morphology modifiers. Each modifier has to be informed by the path the file containing the modifier and the name of the function. E.g: morph_modifiers = [["path_to_module", "name_of_function"], ...].
optimizer (str, default: "IBEA"): algorithm used for optimization, can be "IBEA", "SO-CMA" or "MO-CMA". If the optimizer is "SO-CMA" or "MO-CMA", please pip install bluepyemodel with the 'cma' option.
optimisation_params (dict, default: None): parameters used by BluePyOpt during optimisation. The keys have to match the optimizer's call. E.g., for optimizer MO-CMA: {"offspring_size": 10, "weight_hv": 0.4}.
optimisation_timeout (float, default: 600.0): maximum time in second during which a protocol is allowed to run before being killed.
threshold_efeature_std (float, default: 0.05): lower bound used for the standard deviation of the e-features in the cell evaluator. If informed, the stds are computed as threshold_efeature_std * mean if std is < threshold_efeature_std * min.
max_ngen (int, default: 100): maximum number of generations of the evolutionary process of the optimization.
validation_threshold (float, default: 5.0): used by the default validation function. Threshold under which each score of the emodel has to be for it to pass validation.
plot_optimisation (bool, default: True, Luigi only): should the EModel scores and traces be plotted.

n_model (int, default: 3, nexus access point only, Luigi only): number of models optimized an validated to consider the EModel building task done. The Luigi pipeline will continue launching optimisation batch until this value is reached.
optimisation_batch_size (int, default: 5, nexus access point only, Luigi only): number of optimisation seeds to run in parallel for each batch of optimization launched by the Luigi pipeline.
max_n_batch (int, default: 3, nexus access point only, Luigi only): maximum number of optimisation batches that the Luigi pipeline will run if n_model is not reached.

path_extract_config (str, default: None, local access point only): path to the .json containing the extraction targets, files metadata and the name of the protocols used to compute the threshold of the cell.
name_Rin_protocol (str, default: None, local access point only): name of the protocol associated with the efeatures used for the computation of the input resistance scores during optimisation. This settings has to be set before efeature extraction if you wish to run a threshold based optimisation.
name_rmp_protocol (str, default: None, local access point only): name of the protocol associated with the efeatures used for the computation of the resting membrane potential scores during optimisation. This settings has to be set before efeature extraction if you wish to run a threshold based optimisation. 
validation_protocols (dict default: None, local access point only): names and targets of the protocol that will be used for validation only. This settings has to be set before efeature extraction if you wish to run validation.


## To get started with Model Management

TO COME


## To get started with AIS synthesis

TO COME
