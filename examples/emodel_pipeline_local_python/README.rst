To get started with the E-Model building pipeline
=================================================

This section will talk about the E-Model building pipeline which for now contains e-features extraction, optimisation and model analysis. If you only wish to export a model that was built using the pipeline to hoc, you can jump to the subsection `Exporting the models`_.

Note that despite the present explanation, building an e-model is not a trivial process, therefore, do not hesitate to contact this package authors for help to get you set up.

The E-Model building pipeline can be executed either step by step using Python or all at once as a Luigi workflow.

Running using python with local storage
---------------------------------------

This section presents a general picture of how to create an e-model using python and local storage, it relies on the use of the class ``EModel_pipeline``.

Configuration
~~~~~~~~~~~~~

The main configuration file is referred to as "recipes" since it contains the recipe of how models should be built.
Therefore, in an empty directory, usually named ``config``, you will need to create a file ``recipes.json``. Here is an example of a recipe for a fictitious L5PC model:

.. code-block:: python

    {
        "L5PC": {
            "morph_path": "./morphologies/",
            "morphology": [
                [
                    "L5TPCa",
                    "C060114A5.asc"
                ]
            ],
            "params": "config/params/pyr.json",
            "features": "config/features/L5PC.json",
            "pipeline_settings": {
                "path_extract_config": "config/extract_config/L5PC_config.json",
                "plot_extraction": true,
                "optimisation_timeout": 300,
                "optimiser": "MO-CMA",
                "max_ngen": 100,
                "optimisation_params": {
                    "offspring_size": 20
                },
                "validation_threshold": 5,
                "plot_currentscape": true,
                "currentscape_config": {
                    "title": "L5PC"
                },
                "validation_protocols": [
                    "APWaveform_300"
                ]
            }
        }
    }

Let's go over the content of this file:

* The keys of the dictionary are the names of the models that will be built. Here, we only have one model named ``L5PC``. This name is important as it will be used in every following step to specify which model is to be acted upon.
* ``morph_path`` contains the path of the directory containing the morphologies. This directory has to be a subdirectory of the directory from which the pipeline will be run. Otherwise, the morphologies cannot be versioned.
* ``morphology`` contains the name of the morphology file. The first element of the list is an arbitrary name for the morphology and the second is the name of the file containing the morphology. The file containing the morphology has to be in the directory specified by ``morph_path``.
* ``params`` contains the essential mechanisms specifying their locations (e.g., axonal, somatic) as well as their distributions and parameters, which can be either frozen or free.
* ``features`` contains the path to the file that includes the output of the extraction, which are the ``efeatures`` and ``protocols``. The ``efeatures`` is a list of dictionaries, where each entry contains a feature associated with a specific protocol. ``protocols`` is also a list of dictionaries; each entry in this list contains the protocol's name, amplitude, among other details.
* ``pipeline_settings`` contains settings used to configure the pipeline. There are many settings, that can each be important for the success of the model building procedure. The complete list of the settings available can be seen in the API documentation of the class `EModelPipelineSettings <../../bluepyemodel/emodel_pipeline/emodel_settings.py>`_. An important setting if you wish to run e-feature extraction through the pipeline is ``path_extract_config`` which points to the path of the json file containing the targets, features names, protocols and files (ephys traces) of the extraction process.

Building the models
~~~~~~~~~~~~~~~~~~~

To run the modeling pipeline, you will need to create a python script used to instantiate the pipeline and execute its different steps. The pipeline is a python object of the class `EModel_pipeline <../../bluepyemodel/emodel_pipeline/emodel_pipeline.py>`_. Here is a minimal example of how to instantiate it:

.. code-block:: python

    from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline

    emodel = "L5PC"
    recipes_path = "./config/recipes.json"

    pipeline = EModel_pipeline(
        emodel=emodel,
        recipes_path=recipes_path,
    )

Finally, the different steps of the pipeline can be run with the commands:

.. code-block:: python

    pipeline.extract_efeatures()
    pipeline.optimise(seed=1)
    pipeline.store_optimisation_results()
    pipeline.validation()
    pipeline.plot(only_validated=False)

This snippet will likely not be used as such as the different steps of the pipeline are computationally intensive and will be run separately.

Note that for the pipeline to work, the NEURON mechanisms used by the models need to be present in a local directory named "mechanisms" and compiled using the command:

.. code-block:: python

    nrnivmodl mechanisms

The final models generated using the local access point are stored in the file ``final.json`` and the traces of the models can be seen in ``./figures/``.

Exporting the models
~~~~~~~~~~~~~~~~~~~~

If you wish to use the models generated with BluePyEModel outside of Python, you will need to export them as hoc files.
Following the example above, it can be done with the command:

.. code-block:: python

    from bluepyemodel.export_emodel.export_emodel import export_emodels_hoc
    access_point = pipeline.access_point
    export_emodels_hoc(access_point, only_validated=False, map_function=map)

This will create a local directory containing the hoc files of the models.

Note that if you wish to use the models in a circuit, you will have to use `export_emodels_sonata <../../bluepyemodel/export_emodel/export_emodel.py#L130>`_ instead.
However, most of the time, for circuit building, you will want to generalize the models to the morphologies of the circuit. For that, you will need to perform model management (MM), which is out of the scope of the present package (see `https://github.com/BlueBrain/BluePyMM <https://github.com/BlueBrain/BluePyMM>`_)

Summary of the local directory structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final structure of the local directory for this simpler case should be as follows:

.. code-block::

    .
    ├── pipeline.py
    ├── mechanisms
    │   ├── mode_file1.mod
    │   ├── mode_file1.mod
    │   ├── mode_file3.mod
    ├── config
    │    ├── extract_config
    │    │   ├── L5PC_config.json
    │    ├── features
    │    │   ├── L5PC.json
    │    ├── params
    │    │   ├── pyr.json
    │    └── recipes.json
    ├── morphologies
    │    └── L5TPC.asc


Advanced usage
==============

This section will talk about the E-Model building pipeline using githash versioning and slurm.

Setting up the directory and git repo
-------------------------------------

First, we recommend that you copy the present directory and all of its content to the folder in which you will want to work.

Once that is done you can create the virtual environment in which BluePyEModel will be installed:
``./create_venv.sh``

Then rename the file gitignore_template to .gitignore. This will avoid versioning unwanted files in the future.
``mv gitignore_template .gitignore``

Finally, initialize a git repository in the present directory:
``git init .``

Versioning the runs
-------------------

As you are likely to perform several rounds of extraction, optimisation and analysis, each of the runs will be tracked using a string called ``iteration_tag`` or ``githash``.

At the beginning of each optimisation run, an archive of the present directory will be created and stored in ``./run/GITHASH/``. You can have a look at `./scripts/optimisation.sh <./scripts/optimisation.sh>`_ to see how this operation is performed.

This process will ensure that a copy of the code as used at the moment of the launch exists, and that it remains unchanged even if you change the current directory to perform different optimisations.

The ``githash`` provided by this operation will uniquely characterize the run, and we recommend that you keep a list of the githashes generated and the circumstances in which they were generated.

Configuring your models
-----------------------

The present directory contains template mechanisms, morphologies, recipes and parameters files.
In order to configure the models that you want, you will have to:

* Copy the morphology you wish to use in the ``morphologies`` folder
* Copy the mechanisms (mod files) you wish to use in the ``mechanisms`` folder
* Create a json file containing the parameters of your model and put it in ``./config/params/``.
* Create a json files containing the files_metadata, targets and protocols_rheobase used as targets for the extraction process in ``./config/extract_config/EMODEL_NAME_config.json`` (for the format of this file section `Extraction`_ below).
* Create a new recipe in ``./config/recipes.json`` which should contain the paths to all the files mentioned above as well as the settings you wish to use when running the pipeline. You can have a look at the docstring of the class `EModelPipelineSettings <../../bluepyemodel/emodel_pipeline/emodel_settings.py>`_ for a complete overview of all the settings available.

Running the different steps
---------------------------

The main script used to execute the different steps of model building is the file `pipeline.py <pipeline.py>`_. It contains the commands calling BluePyEModel to perform the operations related to extraction, optimisation, analysis and validation.

Extraction
~~~~~~~~~~

To perform extraction, you will need an extraction config file as mentioned above. This file should contain the metadata of the ephys files that should be considered as well as the targets (protocols and efeatures) that should be extracted from the recordings present in these files.
It is recommended that you generate this file programmatically. The notebook `./extraction_configuration.ipynb <./extraction_configuration.ipynb>`_ gives an example of how to do so.

Then, to run the extraction, inform the name of the emodel in ``scripts/extract.sh`` and execute the file. Please navigate to the scripts directory in your terminal and then execute the following command: ``./extract.sh``
The name of the emodel must match an entry of the file ``recipes.json``.

The results of the extraction (if all goes well), should appear at the path mentioned in the entry ``features`` of the recipe. By convention, this path is usually set to ``./config/features/EMODEL_NAME.json``.
If you asked for the extraction to be plotted in the settings, the plots will be in ``./figures/EMODEL_NAME/extraction/``.

For a complete description of the extraction process, its inner working and settings please refer the `README and examples of BluePyEfe on GitHub <https://github.com/BlueBrain/BluePyEfe/>`_.

Optimisation
~~~~~~~~~~~~

To perform optimisation, you will need to provide a morphology, mechanisms and a parameter configuration file in your recipe.

As optimisation is a costly operation, we will show here how to execute it in parallel using slurm.

First, you will need to compile the mechanisms, which can be done with the command:

.. code-block:: python

    nrnivmodl mechanisms

Configure the #SBATCH directives at the beginning of your SLURM sbatch file according to your job requirements. Then, inform your emodel name in ``./scripts/optimisation.sh`` and execute it. Please navigate to the scripts directory in your terminal and then execute the following command: ``./optimisation.sh``
This will create several slurm jobs for different optimisation seeds and the githash associated to the run (keep it preciously!).

The optimisation usually takes between 2 and 72 hours depending on the complexity of the model.
If the model is not finished after 24 hours, you will need to resume it manually by informing the githash of the run in ``./scripts/optimisation.sh`` and executing it again.
To monitor the state of the optimisation, please have a look at the notebook `./monitor_optimisations.ipynb <./monitor_optimisations.ipynb>`_.

For a more in depth overview of the optimisation process please have a look at the `documentation and examples of the package BluePyOpt on GitHub <https://github.com/BlueBrain/BluePyOpt>`_.

Analysis
~~~~~~~~

Once a round of optimisation is finished, you might want to extract the results from the checkpoint files generated by the optimisation process and plot the traces and scores of the best models.

To do so, inform your emodel name and githash in ``./script/analysis.sh`` and execute it.

It will create a slurm job that will store the results in a local file called ``final.json`` as well as plot figures for these models that you will find in ``./figures/EMODEL_NAME/``.

If you wish to interact with the models, please have a look at the notebook `./exploit_models.ipynb <./exploit_models.ipynb>`_.

Currentscape plots can also be plotted by BluePyEModel, along with the other analysis figures. To do so, you simply have to add ``"plot_currentscape": true,`` to the ``pipeline_settings`` dict of ``./config/recipes.json``. All currents are recorded in [pA]. The currentscape figures are created using the same recordings, and are saved under ``./figures/EMODEL_NAME/currentscape``. If you want to customise your currentscape plots, you can pass a currentscape config to the ``pipeline_settings`` dict of ``./config/recipes.json`` under the key ``currentscape_config``. You can find more information about currentscape and its config `here <https://github.com/BlueBrain/Currentscape>`_.

The recordings of the voltage, as well as every available ionic currents and ionic concentration can be saved locally to ``./recordings`` when setting ``save_recordings`` to ``true`` in the ``pipeline_settings``.

If you don't want to have mechanism-specific currents in the currentscape plots, but have e.g. whole ionic currents plotted, it is possible by putting the names of the variables you want to plot under ``["current"]["names"]`` in the currentscape_config.

Validation
~~~~~~~~~~

If you wish to perform validation of your model (testing the model on protocols unseen during optimisation), you will have to mark these targets as such in your pipeline settings in the recipe file before efeature extraction.

Then, to run the validation, inform the emodel name and githash in ``./script/analysis.sh`` and execute it.
Once the validation is done, the models in your final.json will have a field ``passedValidation``.
This field can have 3 values:

* If it is None, that means the model did not go yet through validation.
* If it is False, it means the models did not pass validation successfully.
* If it is True, the model passed validation successfully.

As for the other steps, please have a look at the `docstring of the settings <../../bluepyemodel/emodel_pipeline/emodel_settings.py>`_ to configure the validation step.
