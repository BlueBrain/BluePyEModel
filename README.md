# BluePyEModel: Blue Brain Python E-Model Building Library


## Introduction

The Blue Brain Python E-Model Building Library (BluePyEModel) is a Python package facilitating the configuration and execution of E-Model building tasks. It covers tasks such as feature extraction, model optimisation and validation. As such, it builds on top of [eFEL](https://github.com/BlueBrain/eFEL), [BluePyEfe](https://github.com/BlueBrain/BluePyEfe) and [BluePyOpt](https://github.com/BlueBrain/BluePyOpt).

For a general overview and example of electrical model building, please refer to the preprint: [A universal workflow for creation, validation and generalization of detailed neuronal models](https://www.biorxiv.org/content/10.1101/2022.12.13.520234v1.full.pdf).

## Installation

To use BluePyEModel on BB5, it is easier to do so in a virtual environment.
It is possible to create a virtual environment using the most recent python version present on spack:

    module load unstable python
    python -m venv myvenv
    module purge all
    source myvenv/bin/activate

Then, BluePyEModel can be installed with the command::

    pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepyemodel[all]

If you do not wish to install all dependencies, specific dependencies can be selected by indicating which ones to install between brackets in place of 'all' (If you want multiple dependencies, they have to be separated by commas). The available dependencies are:

- luigi
- nexus
- all
- currentscape


## To get started with the E-Model building pipeline

![E-Model building pipeline](./images/pipeline.png)

This section will talk about the E-Model building pipeline which for now contains e-features extraction, optimisation and model analysis. If you only wish to export a model that was built using the pipeline to hoc, you can jump to the subsection "Exporting the models".

Note that despite the present explanation, building an e-model is not a trivial process, therefore, do not hesitate to contact the Cells team for help to get you set up.

The E-Model building pipeline can be executed either step by step using Python or all at once as a Luigi workflow.

Both python and luigi execution can rely on either local storage or Nexus to store the configuration files and results.

This gives 4 possible scenarios:
1) Running using python with local storage
2) Running using python with Nexus storage
3) Running using Luigi with local storage
4) Running using Luigi with Nexus storage

Unless you have a Nexus project already set up and are part of one of Blue Brain's major projects, you will be in case number 1. For that reason, the next section will focus on scenario 1.

The section after that will focus on scenario 4 as it is the most complex. Scenario 2 and 3 can be achieved by taking the relevant elements from these two sections and their related examples.

### 1) Running using python with local storage

This section presents a general picture of how to create an e-model using python and local storage, it relies on the use of the class EModel_pipeline.

For a detailed picture, please refer to the example directory [`./examples/emodel_pipeline_local_python`](bluepyemodel/examples/emodel_pipeline_local_python) and its [README](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/blob/main/examples/emodel_pipeline_local_python/README.md) which shows how to setup an optimisation directory and how to run it on BB5 using slurm.

The pipeline is divided in 6 steps:
- extraction: extracts e-features from ephys recordings and averages the results e-feature values along the requested targets.
- optimisation: builds a NEURON cell model and optimises its parameters using as targets the efeatures computed during e-feature extraction.
- storage of the model: reads the results of the extraction and stores the models (best set of parameters) in a local json file or on Nexus.
- validation: reads the models and runs the optimisation protocols and/or validation protocols on them. The e-feature scores obtained on these protocols are then passed to a validation function that decides if the model is good enough.
- plotting: reads the models and runs the optimisation protocols and/or validation protocols on them. Then, plots the resulting traces along the e-feature scores and parameter distributions.
- exporting: read the parameter of the best models and export them in files that can be used either in NEURON or for circuit building.

These six steps are to be run in order as for example validation cannot be run if no models have been stored. Steps "validation", "plotting" and "exporting" are optional. Step "extraction" can also be optional in the case where the file containing the protocols and optimisation targets is created by hand or if it is obtained from an older project.

#### Configuration

The main configuration file is referred to as "recipes" since it contains the recipe of how models should be built.
Therefore, in an empty directory, usually named `config`, you will need to create a file `recipes.json`. Here is an example of a recipe for a fictitious L5PC model:
```
{ 
    "L5PC": {
        "morph_path": "morphologies/",
        "morphology": [["L5TPC","L5TPC.asc"]],
        "params": "./params_pyr.json",
        "features": "./features_L5PC.json",
        "pipeline_settings": {
            "path_extract_config": "config/extraction_config.json",
            "optimisation_timeout": 300,
            "optimiser": "MO-CMA",
            "optimisation_params": {
                "offspring_size": 20
            }
        }
    }
}
```

Let's go over the content of this file:
- The keys of the dictionary are the names of the models that will be built. Here, we only have one model named "L5PC". This name is important as it will be used in every following step to specify which model is to be acted upon.
- `morph_path` contains the path of the directory containing the morphologies. This directory has to be a subdirectory of the directory from which the pipeline will be run. Otherwise, the morphologies cannot be versioned.
- `morphology` contains the name of the morphology file. The first element of the list is an arbitrary name for the morphology and the second is the name of the file containing the morphology. The file containing the morphology has to be in the directory specified by `morph_path`.
- `params` and `features` contains the path to the file containing the configuration of the parameters of the model and optimisation targets of the model respectively. As for the morphology, this file has to be in a local subdirectory. By convention, these files are put in the directory `./config/` or in a subdirectory of it.  To see the specific format of these configuration files, please refer to the example [`./examples/emodel_pipeline_local_python`](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/tree/main/examples/emodel_pipeline_local_python). If the step "extraction" is done through the pipeline, the file containing the optimisation targets will be created programmatically by the pipeline.
- `pipeline_settings` contains settings used to configure the pipeline. There are many settings, that can each be important for the success of the model building procedure. The complete list of the settings available can be seen in the API documentation of the class `EModelPipelineSettings`. An important settings if you wish to run e-feature extraction through the pipeline is `path_extract_config` which points to the path of the json file containing the targets of the extraction process. Once again, for the format of this file, please refer to the example [`./examples/emodel_pipeline_local_python`](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/tree/main/examples/emodel_pipeline_local_python).

#### Building the models

To run the modeling pipeline, you will need to create a python script used to instantiate the pipeline and execute its different steps. The pipeline is a python object of the class [`EModel_pipeline`](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/blob/main/bluepyemodel/emodel_pipeline/emodel_pipeline.py#L23). Here is a minimal example of how to instantiate it:
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
This snippet will likely not be used as such as the different steps of the pipeline are computationally intensive and will be run separately.

Note that for the pipeline to work, the NEURON mechanisms used by the models need to be present in a local directory named "mechanisms" and compiled using the command:
```
nrnivmodl mechanisms
```

The final models generated using the local access point are stored in the file `final.json` and the traces of the models can be seen in `./figures/`.

#### Exporting the models

If you wish to use the models generated with BluePyEModel outside of Python, you will need to export them as hoc files.
Following the example above, it can be done with the command:
```
from bluepyemodel.export_emodel.export_emodel import export_emodels_hoc
access_point = pipeline.access_point
export_emodels_hoc(access_point, only_validated=False, map_function=map)
```
This will create a local directory containing the hoc files of the models.

Note that if you wish to use the models in a circuit, you will have to use [`export_emodels_sonata`](bluepyemodel/export_emodel/export_emodel.py#L130) instead.
However, most of the time, for circuit building, you will want to generalize the models to the morphologies of the circuit. For that, you will need to perform model management (MM), which is out of the scope of the present package (see https://github.com/BlueBrain/BluePyMM)

#### Summary of the local directory structure

The final structure of the local directory for this simpler case should be as follows:
```
.
├── pipeline.py
├── mechanisms
│   ├── mode_file1.mod
│   ├── mode_file1.mod
│   ├── mode_file3.mod
├── config
│    ├── features_L5PC.json
│    ├── params_pyr.json
│    ├── extraction_config.json
│    └── recipes.json
├── morphologies
│    └── L5TPC.asc
```

In the more complex case where githash versioning and slurm are used, refer to the structure of the example of [`./examples/emodel_pipeline_local_python`](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/tree/main/examples/emodel_pipeline_local_python).

### 4) Running using Luigi with Nexus storage

This section present the general picture of how to create an e-model using Luigi and Nexus storage. For a detailed example, please refer to the files in [`examples/emodel_pipeline_nexus_ncmv3`](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/tree/main/examples/emodel_pipeline_nexus_ncmv3).

Warning: to run the emodel pipeline using Nexus as a backend you will first need a fully configured Nexus project and be able to perform cross-bucket in projects containing the morphologies, mechanisms and ephys data you wish to use. All of these have to be setup by the DKE team beforehand.

To run the pipeline with luigi, you will need:
- A virtual environment with BluePyEModel installed with the options nexus and luigi: (```pip install bluepyemodel[luigi,nexus]```)
- A `luigi.cfg` file containing the Luigi specific settings (see example [`./examples/emodel_pipeline_nexus_ncmv3/luigi.cfg`](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/blob/main/examples/emodel_pipeline_nexus_ncmv3/luigi.cfg)).
- On Nexus, entities of the type "EModelConfiguration", "EModelPipelineSettings" and "ExtractionTargetsConfiguration" (or "FitnessCalculatorConfiguration" if you already have optimisation targets). Please refer to the file [`pipeline.py`](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/blob/main/examples/emodel_pipeline_nexus_ncmv3/pipeline.py) and the notebooks for an example of how to create such resources.

The pipeline can then be run using the command:

    bbp-workflow launch-bb5 -f --config=luigi.cfg bluepyemodel.tasks.emodel_creation.optimisation EModelCreation emodel=EMODEL ttype=TTYPE species=SPECIES brain-region=BRAIN_REGION iteration-tag=ITERATION_TAG`

The final models generated using the Nexus access point are stored in the Nexus project in Resources of type [`EModel`](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/blob/main/bluepyemodel/emodel_pipeline/emodel.py#L24).

### Schematics of BluePyEModel classes

![Schematics of BluePyEModel classes](./images/classes_schema.png)