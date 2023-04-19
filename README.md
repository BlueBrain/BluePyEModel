# BluePyEModel: Blue Brain Python E-Model Building Library


## Introduction

The Blue Brain Python E-Model Building Library (BluePyEModel) is a Python package facilitating the configuration and execution of E-Model building tasks. It covers tasks such as feature extraction, model optimisation and validation. As such, it builds on top of eFEL, BluePyEfe and BluePyOpt.


## Installation

If you want to use BluePyEModel, you can install it in a virtual environment.

To do so on BB5, first create a virtual environment using the most recent python version on spack:

    module load unstable python
    python -m venv myvenv
    module purge all
    source myvenv/bin/activate

Then, you can install BluePyEModel with the following line:

    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepyemodel[all]

You can also select the specific dependencies you want by putting them into the brackets instead of 'all' (If you want multiple dependencies, you have to separate them by commas). The available dependencies are:

- luigi
- nexus
- all
- currentscape


## To get started with the E-Model building pipeline

This section will talk about the E-Model building pipeline which for now contains e-features extraction, optimisation and model analysis. If you only wish to export a model  that was built using the pipeline to hoc, you can jump to the subsection "Exporting the models".

Note that despite the present explanation, building an e-model is not a trivial process, therefore, do not hesitate to contact the Cells team for help to get you set up.

The E-Model building pipeline can be executed either step by step using Python or all at once as a Luigi workflow.

Both python and luigi execution can rely on either local storage or Nexus to store for the configuration files and results.

This gives 4 possible scenarios:
1) Running using python with local storage
2) Running using python with Nexus storage
3) Running using Luigi with local storage
4) Running using Luigi with Nexus storage

Unless you have a Nexus project already set up and are part of one of Blue Brain's major projects, you will be in case number 1. For that reason, the next section will focus on scenario 1.

The section after that will focus on scenario 4 as it is the most complex. Scenario 2 and 3 can be achieved by taking the relevant elements from these two sections and their related examples.

### 1) Running using python with local storage

#### Configuration

This section presents a general picture of how to create an e-model using python and local storage, it relies on the use of the class EModel_pipeline.

For a detailed picture, please refer to the example directory `./examples/emodel_pipeline_local_python` and its README which shows how to setup an optimisation directory and how to run it on BB5 using slurm.

The pipeline is divided in 6 steps:
- extraction: extracts efeatures from the ephys recordings and averages the results along the requested targets.
- optimisation: builds a NEURON cell model and optimises its parameters using as targets the efeatures computed during efeature extraction.
- storage of the model: reads the results of the extraction and stores the models (best set of parameters) in local or on Nexus.
- validation: reads the models and runs the optimisation protocols and/or validation protocols on them. The efeature scores obtained on these protocols are then passed to a validation function that decides if the model is good enough.
- plotting: reads the models and runs the optimisation protocols and/or validation protocols on them. Then, plots the resulting traces along the efeature scores and parameter distributions.
These four steps are to be run in order as, for example, validation cannot be run if no models have been stored.
- exporting: read the parameter of the best models and export them in files that can be used either in NEURON or for circuit building.

In the present case, we will use the local access point. The main configuration file needed by the local access point is a file referred to as "recipes" since it contains the recipe of how a model should be built.
Therefore, in an empty directory, you will need to create a file `recipes.json`. Here is an example of a recipe for a fictitious L5PC model:
```
{ 
    "L5PC": {
        "morph_path": "morphologies/",
        "morphology": [["L5TPC","FILENAME.asc"]],
        "params": "./params_pyr.json",
        "protocol": "./protocol_L5PC.json",
        "features": "./features_L5PC.json",
        "pipeline_settings": {
            "path_extract_config": "extraction_config.json",
            "optimisation_timeout": 300,
            "optimiser": "MO-CMA",
            "optimisation_params": {
                "offspring_size": 20
            }
        }
    }
}
```
Each entry of the recipe must contain the fields morph_path, morphology, params, protocol, features and these have to point toward the json files that will be used to configure the model and the optimisation.
The format of each of these files being strictly defined, please refer to the example for the exact format.

The recipes also contain settings used to configure the pipeline. There are many settings, that can each be important for the success of the model building procedure. The complete list of the settings available can be seen in the docstring of the class at `bluepyemodel/emodel_pipeline/emodel_settings.py`.

#### Building the models

Then, you will need to create a python file that can be used to instantiate the pipeline and run it. Here, the pipeline is a python object of the class EModel_pipeline. Here is a minimal example of how to instantiate it:
```
from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline

emodel = "L5PC"
recipes_path = "./recipes.json"
data_access_point = "local"

pipeline = EModel_pipeline(
    emodel=emodel,
    data_access_point=data_access_point,
    recipes_path=recipes_path,
)
```

Finally, the different steps of the pipeline can be run with the commands:
```
pipeline.extract_efeatures()
pipeline.optimise(seed=1)
pipeline.store_optimisation_results()
pipeline.plot(only_validated=False)
```

Note that the mechanisms used by the models need to be present in a local directory named "mechanisms" and compiled using the command:
```
nrnivmodl mechanisms
```

The final models generated using the local access point are stored in the file `final.json` and the traces of the models can be seen in `./figures/`.

#### Exporting the models

If you wish to use the models generated with BluePyEModel outside of Python, you will need to export them as hoc files.
Following the example above, you can do so with the command:
```
from bluepyemodel.export_emodel.export_emodel import export_emodels_hoc
access_point = pipeline.access_point
export_emodels_hoc(access_point, only_validated=False, map_function=map)
```
This will create a local directory containing the hoc files of the models.

Note that if you wish to use the models in a circuit, you will have to use `export_emodels_sonata` instead.
However, most of the time, for circuit building, you will want to generalize the models to the morphologies of the circuit. For that, you will need to perform model management (MM), which is out of the scope of the present package (see https://github.com/BlueBrain/BluePyMM)

### 4) Running using Luigi with Nexus storage

This section present the general picture of how to create an e-model using Luigi and Nexus storage. For a detailed example, please refer to the files in examples/emodel_pipeline_nexus_ncmv3

Warning: to run the emodel pipeline using Nexus as a backend you will first need a fully configured Nexus project and be able to perform cross-bucket in projects containing the morphologies, mechanisms and ephys data you wish to use. All of these have to be setup by the DKE team beforehand.

To run the pipeline with luigi, you will need:
- a virtual environment with BluePyEModel installed with the options nexus and luigi: (```pip install bluepyemodel[luigi,nexus]```)
- a `luigi.cfg` file containing Luigi specific settings.
- On Nexus, entities of the type "EModelConfiguration", "EModelPipelineSettings" and "ExtractionTargetsConfiguration" (or "FitnessCalculatorConfiguration" if you already know the targets that you wish to fit). Please refer to the files 'pipeline.py` and the notebooks for an example of how to create such resources.

The pipeline can then be run using the command:
`bbp-workflow launch-bb5 -f --config=luigi.cfg bluepyemodel.tasks.emodel_creation.optimisation EModelCreation emodel=EMODEL ttype=TTYPE species=SPECIES brain-region=BRAIN_REGION iteration-tag=ITERATION_TAG`

The final models generated using the Nexus access point are stored in the Nexus project in Resources of type `EModel`.
