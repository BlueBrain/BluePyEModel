To get started with the E-Model building pipeline
=================================================
This guide will walk you through the process of setting up the E-Model building pipeline and running it on your local machine or on a cluster using Slurm. If you only wish to export an e-model that was built using the pipeline to hoc, you can jump to the section `Exporting the models`_.

Note that despite the present explanation, building an e-model is not a trivial process, therefore, do not hesitate to contact this package authors for help to get you set up. The present folder have been designed to be used with Slurm `Running the example using Slurm`_. If you want to understand the code better we encourage you to read on how to run the example locally.

Running the example locally
---------------------------
This guide illustrates how to execute the example locally, focusing on the utilisation of the EModel_pipeline class. Herein, we will navigate through the various stages of the pipeline, demonstrated using the L5PC model as a practical example.

Configuration
~~~~~~~~~~~~~

The main configuration file is referred to as "recipes" since it contains the recipe of how models should be built.
Therefore, in an empty directory, usually named ``config``, you will need to create a file ``recipes.json``. Here is an example of a recipe for a L5PC model:

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
                "default_std_value": 0.01,
                "efel_settings":{
                    "strict_stiminterval": true,
                    "Threshold": -20.0,
                    "interp_step": 0.025
                },
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
                "validation_protocols": ["sAHP_220"],
                "name_Rin_protocol":"IV_-20",
                "name_rmp_protocol":"IV_0"
            }
        }
    }

To provide a comprehensive understanding, let's delve into the specifics of this JSON configuration:

The keys of the dictionary are the names of the models that will be built. Here, we only have one model named ``L5PC``. This name is important as it will be used in every following step to specify which model is to be acted upon.

* ``morph_path`` contains the path of the directory containing the morphologies. This directory has to be a subdirectory of the directory from which the pipeline will be run. Otherwise, the morphologies cannot be versioned.
* ``morphology`` contains the name of the morphology file. The first element of the list is an arbitrary name for the morphology and the second is the name of the file containing the morphology. The file containing the morphology has to be in the directory specified by ``morph_path``.
* ``params`` contains the essential mechanisms specifying their locations (e.g., axonal, somatic) as well as their distributions and parameters, which can be either frozen or free.
* ``features`` contains the path to the file that includes the output of the extraction step, see `Extraction`_ for more details.
* ``pipeline_settings`` contains settings used to configure the pipeline. There are many settings, that can each be important for the success of the model building procedure. The complete list of the settings available can be seen in the API documentation of the class `EModelPipelineSettings <../../bluepyemodel/emodel_pipeline/emodel_settings.py>`_. An important setting if you wish to run e-feature extraction through the pipeline is ``path_extract_config`` which points to the path of the json file containing the targets of the extraction process (e.g. ``L5PC_config.json``), features names, protocols and files (ephys data). More details on how to generate this file can be found in the section `Extraction`_.

In this example, the expected final structure of the local directory should be as follows:

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


Getting the ephys data
~~~~~~~~~~~~~~~~~~~~~~

Prior to initiating the extraction process, the electrphysiological data needs to be placed in ephys_data folder. In this example, the data used is for continuous adapting pyramidal cells (cADpyr) e-type model of rat somatosensory cortex. The data is accessible for download from this `repo <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/feature_extraction/input-traces/C060109A1-SR-C1>`_. You can conveniently retrieve it using the ``download_ephys_data.sh`` script. When using your own ephys data, it is crucial to specify the type of files you are working with. Please set the ``file_type`` variable to either "ibw" or "nwb" in the configuration file ``targets.py``, depending on your data format. Additionally, ensure you provide the correct path to your ephys data files in the ``filenames`` list within the same configuration file.

Extraction
~~~~~~~~~~

To perform the extraction, you will need an extraction config file `./config/extract_config/L5PC_config.json <./config/extract_config/L5PC_config.json>`_. This file will be automatically created before the extraction by the ``configure_targets`` function in ``./pipeline.py``, if you are using your own data, the function might need to be modified for your needs. This function relies on the parameters set in the ``./targets.py`` configuration file which contains:

* files_metadata: Path to the ephys data files. Please ensure to set your file type (ibw or nwb) in the ``file_type`` variable.
* ecodes_metadata: List of ecodes protocols (e.g. IDthresh) for which you want features to be extracted.
* protocols_rheobase: The protocol to use to find the rheobase of the cell.
* targets: List of dictionaries, where each entry contains the protocol within which the features are extracted at a specific amplitude.

Therefore, before proceeding, it is essential to edit ``./targets.py`` to accurately reflect your specific settings. Once ``./targets.py`` has been configured to your requirements, the ``configure_targets`` function will parse these settings and subsequently create the appropriate ``L5PC_config.json`` configuration file.

We provide a Python script, pipeline.py, designed to initialise and orchestrate the various stages of the pipeline. This pipeline operates as a Python object, specifically an instance of the EModel_pipeline class, which you can find more about here: `EModel_pipeline <../../bluepyemodel/emodel_pipeline/emodel_pipeline.py>`_.

Then, to create the extraction configuration file and run the extraction process execute the following command:

.. code-block:: shell

    python pipeline.py --step='extract' --emodel='L5PC'

Please make sure that the name of the e-model matches an entry of the file ``recipes.json``.

The results of the extraction (if all goes well), should appear at the path mentioned in the entry ``features`` of the recipe. By convention, this path is usually set to ``./config/features/EMODEL_NAME.json``. The features file contains the ``efeatures`` and ``protocols``. The ``efeatures`` is a list of dictionaries, where each entry contains a feature associated with a specific protocol. ``protocols`` is also a list of dictionaries; each entry in this list contains the protocol's name, amplitude, among other details.

If you asked for the extraction to be plotted in the settings, the plots will be in ``./figures/EMODEL_NAME/extraction/``. The folder contains figures for each cell that has been extracted. Each cell folder should have plots for:

* Individual features vs relative/absolute stimulus amplitude.
* Recordings plot for each protocol specified during extraction.

Note that our extraction process utilises traces from just one cell in this example, leading to limited sample sizes and occasionally, small or zero standard deviations (std) for certain features. This can inflate feature scores post-optimisation. To counteract this, any calculated std of zero during extraction is replaced by a default value specified in the ``default_std_deviation`` of the ``pipeline_settings`` as mentioned in the ``recipes.json``, please refer to the `Configuration`_ section.

For a complete description of the extraction process, its inner working and settings please refer the `README and examples of BluePyEfe on GitHub <https://github.com/BlueBrain/BluePyEfe/>`_.

Optimisation
~~~~~~~~~~~~

To perform optimisation, you will need to provide a morphology, mechanisms and a parameter configuration file in your recipe.

Note that for the optimisation to work, it is necessary to compile the NEURON mechanisms (.mod files) located  within the ``./mechanisms`` for this present example. This can be achieved using the following command:

.. code-block:: shell

   nrnivmodl ./mechanisms

This command should generate a folder containing compiled mechanisms, and the name of this folder will vary depending on your machine's architecture.

Then, to initiate the optimisation process on your local machine, just enter the command below:

.. code-block:: shell

    python pipeline.py --step='optimise' --emodel='L5PC'

However, since optimisation requires significant resources, see the `Running the example using Slurm`_ section for a more efficient approach, which explains how to carry out the task in parallel using Slurm.

To monitor the state of the optimisation, use the ``./monitor_optimisation.py``:

.. code-block:: shell

    python monitor_optimisation.py

Alternatvely, you can use the notebook `./monitor_optimisation.ipynb <./monitor_optimisation.ipynb>`_ for better visualisation of the optimisation process.

Analysis
~~~~~~~~

Once a round of optimisation is finished, you might want to get the results from the checkpoint files (within the `./checkpoints` directory) generated by the optimisation process and plot the traces and scores of the models. The final models generated are stored in the file ``final.json``.

To proceed with the analysis, execute the command provided below:

.. code-block:: shell

    python pipeline.py --step='analyse' --emodel='L5PC'

This particular command triggers a sequence of operations within the Python script, as it invokes the following methods:

.. code-block:: python

    pipeline.store_optimisation_results()
    pipeline.validation()
    pipeline.plot(only_validated=False)

These methods, called in succession, are responsible for storing the results of the optimisation, validating the e-models (testing the model on protocols unseen during optimisation), and then plotting the data.

The validation protocols are specified in the ``pipeline_settings`` dict of ``./config/recipes.json`` under the key ``validation_protocols``. Once the validation is done, the e-models in your ``final.json`` will have a field ``validated``.
This field can have 3 values:

* If it is None, that means the model did not go yet through validation.
* If it is False, it means the model did not pass validation successfully.
* If it is True, the model passed validation successfully.

The plots are stored in ``./figures/`` which contains the following subfolders:

* ``efeatures_extraction``: Contains separate figures for each e-feature, each drawn based on the specific protocol used for extraction.
* ``distributions``: Displays optimisation parameter distributions between the low and high optimisation bounds as specified in params.json. The figure depicts parameter variations of only the best individuals of each seed.
* ``optimisation``: Depicts the optimisation curve, highlighting optimisation progress over generations. It plots the minimum and average optimisation fitness scores versus the number of optimisation generations, alongside details such as the lowest score achieved, total generations completed, the specific evolutionary algorithm employed, and the final status of the optimisation procedure.
* ``parameter_evolution``: Illustrates the evolution of the parameters within the optimisation bounds over generations.
* ``scores``: Presents the feature scores of each optimised e-feature in terms of z-scores from the experimental e-feature mean value.
* ``traces``: Exhibits the traces derived from the resulting optimised e-model for each optimised and validated protocol.
* ``currentscape``: Currentscape plots (see section `Currentscape`_) for each optimisation protocol.
The folders, currentscape, distributions, scores and traces will contain figures within the ``all`` subfolder. If ``pipeline.plot(only_validated=True)``, only the validated models are plotted within the ``validated`` subfolder.

If you wish to interact with the e-models, please have a look at the notebook `./exploit_models.ipynb <./exploit_models.ipynb>`_.

Note that you may observe disproportionately large scores for some features. This phenomenon often originates from the relatively small standard deviations associated with the extraction of these particular features, which in turn, is frequently a consequence of utilising a smaller sample size. Smaller sample sizes tend to yield less diverse data, thereby restricting the variability and potentially skewing feature scores post-optimisation.


Currentscape
~~~~~~~~~~~~

Currentscape plots can also be plotted by BluePyEModel, along with the other analysis figures. To do so, you simply have to add ``"plot_currentscape": true,`` to the ``pipeline_settings`` dict of ``./config/recipes.json``. All currents are recorded in [pA]. The currentscape figures are created using the same recordings, and are saved under ``./figures/EMODEL_NAME/currentscape``. If you want to customise your currentscape plots, you can pass a currentscape config to the ``pipeline_settings`` dict of ``./config/recipes.json`` under the key ``currentscape_config``. You can find more information about currentscape and its config `here <https://github.com/BlueBrain/Currentscape>`_.

The recordings of the voltage, as well as every available ionic currents and ionic concentration can be saved locally to ``./recordings`` when setting ``save_recordings`` to ``true`` in the ``pipeline_settings``.

If you don't want to have mechanism-specific currents in the currentscape plots, but have e.g. whole ionic currents plotted, it is possible by putting the names of the variables you want to plot under ``["current"]["names"]`` in the currentscape_config.


Running the example using Slurm
-------------------------------

The Slurm version of the pipeline parallels its local counterpart, yet it requires preliminary configuration.

Setting up the directory and git repo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we recommend that you copy the present directory and all of its content to the folder in which you will want to work.

Once that is done you can create the virtual environment in which BluePyEModel will be installed:
``./create_venv.sh``

Then rename the file gitignore_template to .gitignore. This will avoid versioning unwanted files in the future.
``mv gitignore_template .gitignore``

Finally, initialise a git repository in the present directory:
``git init .``

Versioning the runs
~~~~~~~~~~~~~~~~~~~

As you are likely to perform several rounds of extraction, optimisation and analysis, each of the runs will be tracked using a string called ``iteration_tag`` or ``githash``.

At the beginning of each optimisation run, an archive of the present directory will be created and stored in ``./run/GITHASH/``. You can have a look at `./optimisation.sh <./optimisation.sh>`_ to see how this operation is performed.

This process will ensure that a copy of the code as used at the moment of the launch exists, and that it remains unchanged even if you change the current directory to perform different optimisations.

The ``githash`` provided by this operation will uniquely characterise the run, and we recommend that you keep a list of the githashes generated and the circumstances in which they were generated.

Running the different steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refer to the local counteract for the configuration of the recipes and targets files.

To facilitate the execution of the different steps of the pipeline on Slurm, we provide the following auxiliary scripts that can be executed in the following order:

.. code-block:: shell

    ./extract.sh
    ./optimisation.sh
    ./analysis.sh

Make sure to configure the necessary variables within these scripts, including setting the ``OPT_EMODEL`` value as well as configuring their sbatch counterpart by setting the ``#SBATCH`` directives according to your job requirements.

These scripts will also generates logs of the different steps for each run to track its progress and capture any issues that may arise during execution. These log files are stored in the ``./logs`` with a naming convention reflective of the operation and its corresponding job identifier (e.g., ``opt_jobid.log``). In addition to individual log files, each step maintains its own historical record (e.g., ``extract_list.log``, ``opt_list.log`` ``analyse_list.log``) . These files are also situated within the ``./logs`` directory, serving as cumulative logs that document the series of runs pertinent to that particular step. Please ensure to check these logs if you encounter issues during the pipeline execution.

When running the Optimisation, the script will create several slurm jobs for different optimisation seeds and a githash associated to the run (keep it preciously!), In case it goes missing, however, you can retrieve the githash from the ``opt_list.log`` file associated with each run. Note that the optimisation script  handles the compilation of mechanisms, assuming they are located within the ``./mechanisms`` directory. This is done to ensure that the mechanisms are compiled again if there are any changes.

The optimisation usually takes between 2 and 72 hours depending on the complexity of the model. If the model is not finished after 24 hours, you will need to resume it manually by informing the githash of the run in ``./optimisation.sh`` and executing it again.

Exporting the models
--------------------

If you wish to use the models generated with BluePyEModel outside of Python, you will need to export them as hoc files.
Following the example above, it can be done with the command:

.. code-block:: python

    from bluepyemodel.export_emodel.export_emodel import export_emodels_hoc
    access_point = pipeline.access_point
    export_emodels_hoc(access_point, only_validated=False, map_function=map)

This will create a local directory containing the hoc files of the models.

Note that if you wish to use the models in a circuit, you will have to use `export_emodels_sonata <../../bluepyemodel/export_emodel/export_emodel.py#L130>`_ instead.
However, most of the time, for circuit building, you will want to generalise the models to the morphologies of the circuit. For that, you will need to perform model management (MM), which is out of the scope of the present package (see `https://github.com/BlueBrain/BluePyMM <https://github.com/BlueBrain/BluePyMM>`_)
