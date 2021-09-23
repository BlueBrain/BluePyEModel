# BluePyEModel: Blue Brain Python E-Model Building Library


## Introduction

The Blue Brain Python E-Model Building Library (BluePyEModel) is a Python package facilitating the configuration and execution of E-Model building tasks. It covers tasks such as feature extraction, model optimisation, validation, model management and AIS synthesis. As such, it builds on top of BluePyEfe, BluePyOpt and BluePyMM.


## Installation

If you want to use BluePyEModel, you can either load the  `bluepyemodel` module if you are on BB5, or install BluePyEModel in a virtual environment.

For loading the module, type the following lines in your command line interface:

    module load unstable
    module load py-bluepyemodel

For installing BluePyEModel in a virtual environment, you first have to have cmake and GCC installed (the same your python has been compiled with), as some BluePyEModel dependencies need them. If you are on BB5, you can easily load the cmake and gcc modules using the following lines:

    module load unstable
    module load gcc/9.3.0 cmake

And replace 9.3.0 by the gcc version your python has been compiled with. You might have to load archived modules if gcc < 9.3.0 is needed.

Then, you can install BluePyEModel with the following line:

    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepyemodel[all]

If you want all the dependencies to be available. You can also select the dependencies you want by putting them into the brackets instead of 'all' (If you want multiple dependencies, you have to separate them by commas). The available dependencies are:

- luigi
- nexus
- generalisation
- cma
- all


## To get started with the E-Model building pipeline

This section will talk specificly about the E-Model building pipeline which for now encapsulate feature extraction, optimisation and model analysis. For model management and AIS synthesis, documentation is not available yet.

Despite the presence of the following explanation, E-Model building pipeline is not a trivial process, therefore, do not hesitate to contact the Cells team for help to get you setup (tanguy.damart@epfl.ch).


### Running the pipeline

The E-Model building pipeline can be executed either step by step using Python or all at once as a Luigi workflow.

For either, three information will always be required:
- A name for the emodel.
- The name of the species. As of now, it can be human, rat or mouse.
- The name of the brain region.

#### Running the pipeline with Python/CLI

To run the pipeline using Python, you will need to use the class EModel_pipeline located in bluepyemodel/emodel_pipeline/emodel_pipeline.py. An example of the use of this class can be seen in the function `main` of the same file. To start, we recommend that you copy the inside of this function `main` or directly call the file bluepyemodel/emodel_pipeline/emodel_pipeline.py using the command line. In the latter case, please run the command `python emodel_pipeline.py --help` for a list of expected arguments.

The pipeline is divided in 4 steps:
- extraction: extracts efeatures from the ephys recordings and averages the results along the requested targets.
- optimisation: builds a NEURON cell model and optimizes its parameters using as targets the efeatures computed during efeature extraction.
- storage of the model: reads the results of the extraction and stores the models (best set of parameters) in local or on Nexus.
- validation: reads the models and runs the optimisation protocols and/or validation protocols on them. The efeature scores obtained on these protocols are then passed to a validation function that decides if the model is good enough.
- plotting: reads the models and runs the optimisation protocols and/or validation protocols on them. Then, plots the resulting traces along the efeature scores and parameter distributions.

These four steps are to be run in order as, for example, validation cannot be run if no models have been stored.

#### Running the pipeline with Luigi

To run the pipeline with luigi, you will need a luigi.cfg file. This file will contain all the arguments for the pipeline, similar to the ones that were provided when running with Python, plus a few luigi specific settings.

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
enable-internet=True

[BB5-WORKER]
exclusive=True
mem=0
nodes=4
enable-internet=True

[core]
log_level=INFO

[parallel]
backend=ipyparallel

[EmodelAPIConfig]
api=local
recipes_path=./recipes.json
```

The pipeline can then be run using the command:
`bbp-workflow launch-bb5 -f --config=luigi.cfg bluepyemodel.tasks.emodel_creation.optimisation EModelCreation emodel=L5PC species=mouse brain-region=SSCX`
Where the emodel, species and brain-region have to be replace by the ones at hand.


### The data access points

To access the data needed by the E-Model building process such as settings, ephys, morphology and more, the E-Model building pipeline relies on a data access point. The role of the access point is to retrieve the data from local storage or from Nexus and shape it in the format expected by the different tasks.

There are two types of data access points: "local" and "nexus".

#### The local data access point

The local data access point retrieves and stores data from local storage and, for most operations, from the current working directory. An example of an E-Model building pipeline using local storage can be seen in example/emodel_pipeline_local_python.

To use the local data access point, you first need to create a recipes.json, which will contain, for each e-model, a recipe describing how to build it. Here is an example of a recipe for a fictitious L5PC model:
```
{ 
    "L5PC": {
        "morph_path": "morphologies/",
        "morphology": [["L5TPC","FILENAME.asc"]],
        "params": "params_pyr.json",
        "protocol": "protocol_L5PC.json",
        "features": "features_L5PC.json",
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
The path to this recipes.json will need to be provided in the argument `recipes_path` (to the Python script or in the luigi.cfg).

The recipes.json contains two types of information:
- It provides paths to the json files containing the morphology, parameters, features and protocols used to build the models. The format of the parameter file can be seen in example/emodel_pipeline_local_python, while the protocol and efeatures files will be generated by BluePyEModel (they can also be created by hand).
- It contains settings used to configure the pipeline. A complete list of the settings available can be seen below.

The final models generated using the local access point are stored in the file final.json.

#### The Nexus data access point

The Nexus data access point retrieves and stores data from a Nexus project where the data is stored as Resources. We highly recommend that you read the Nexus documentation and configure your Nexus project with the help of the DKE team before proceeding with E-Model building using the Nexus access point. Additionally, contact tanguy.damart@epfl.ch to help you get started.
An example of an E-Model building pipeline using Nexus can be seen in example/emodel_pipeline_nexus_luigi.

As the Nexus access point expects all data and settings to be obtained through Nexus, the first step is to register all the related required information in your Nexus project. To do so you will need to instantiate a Nexus access point (bluepyemodel/access_point/nexus.py) and register the data as Resources using the following methods of this class: `store_trace_metadata`, `store_emodel_targets`, `store_optimisation_parameter`, `store_channel_distribution`, `store_morphology`.
For a more detailed description of these functions, please refer to the file example/emodel_pipeline_nexus_luigi/pipeline.py and to the docstring of the function in bluepyemodel/access_point/nexus.py.

The final models generated using the Nexus access point are stored in the Nexus project in Resources of `type EModel`.

Pipeline settings:
-----------------

The settings of the pipeline are set as follows:
- When using the 'local' data access point, settings are to be informed in the field "pipeline_settings" of the recipes for each emodel independently.
- When using the 'nexus' data access point, settings have to be registered as a Resource of type PipelineSettings. This can be done using the function store_pipeline_settings from bluepyemodel.access_point.nexus.

The list of settings is accessible in the docstring of the class bluepyemodel.emodel_pipeline.emodel_settings.


## To get started with Model Management

TO COME


## To get started with AIS synthesis

TO COME
