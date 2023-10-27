Getting started with E-Model building pipeline
==============================================

This guide will walk you through the process of setting up the E-Model building pipeline and running it on your local machine or on a cluster using Slurm. The present folder has been designed to be used with Slurm (see `Running the example using Slurm`_). To understand the code better, we encourage you to read `Running the example locally`_.

Note that despite the present explanation, building an e-model is not a trivial process, therefore, do not hesitate to contact this package authors for help to get you set up.

If you encounter any issues during the execution of the pipeline, please refer to `Troubleshooting`_ for potential solutions.

Running the example locally
---------------------------

This part illustrates how to execute the example locally (on your PC), focusing on utilising the `EModel_pipeline <../../bluepyemodel/emodel_pipeline/emodel_pipeline.py>`_ class. To ease the execution of the pipeline, we provide a Python script, ``pipeline.py``, designed to initialise and orchestrate various stages of the pipeline. This pipeline operates as a Python object, specifically an instance of the ``EModel_pipeline`` class. Herein, we will navigate through the various stages of the pipeline: extract, optimise, analyse and export, demonstrated using the L5PC model as a practical example.

Configuration
~~~~~~~~~~~~~

The main configuration file is named “recipes” as it contains the ingredients to build the model. Therefore, in an empty directory, named ``config``, you need to create a file ``recipes.json``. Here is an example of a recipe for a L5PC model:

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
                "extract_absolute_amplitudes": false,
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

* ``morph_path`` contains the path of the directory containing the morphologies. This directory has to be a subdirectory of the directory from which the pipeline will be run. Otherwise, the morphologies cannot be versioned (see `Versioning the runs`_).
* ``morphology`` contains the name of the morphology file. The first element of the list is an arbitrary name for the morphology and the second is the name of the file containing the morphology. The file containing the morphology has to be in the directory specified by ``morph_path``.
* ``params`` contain the essential mechanisms specifying their locations (e.g., axonal, somatic) as well as their distributions and parameters, which can be either frozen or free.
* ``features`` contains the path to the file that includes the output of the extraction step, see `Extraction`_ for more details.
* ``pipeline_settings`` contains settings used to configure the pipeline. There are many settings, that can each be important for the success of the model building procedure. The complete list of the settings available can be seen in the API documentation of the class `EModelPipelineSettings <../../bluepyemodel/emodel_pipeline/emodel_settings.py>`_. An important setting if you wish to run e-feature extraction through the pipeline is ``path_extract_config`` which points to the path of the json file containing the targets of the extraction process (e.g. ``L5PC_config.json``), features names, protocols and files (ephys data). More details on how to generate this file can be found in the section `Extraction`_.

In this example, the expected final structure of the local directory should be as follows:

.. code-block::

    .
    ├── pipeline.py
    ├── mechanisms
    │   ├── mod_file1.mod
    │   ├── mod_file1.mod
    │   ├── mod_file3.mod
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

Prior to initiating the extraction process, the electrophysiological data needs to be placed in ephys_data folder. In this example, the data used is for continuous adapting pyramidal cells (cADpyr) e-type model of rat somatosensory cortex. The data is accessible for download from this `repository <https://github.com/BlueBrain/SSCxEModelExamples/tree/main/feature_extraction/input-traces/C060109A1-SR-C1>`_. You can conveniently retrieve it using the ``download_ephys_data.sh`` script. When using your own ephys data, it is crucial to specify the type of files you are working with.

The example works with Igor Binary Wave (ibw) files. You can also use Neurodata Without Borders (nwb) files. Please update the ``file_type`` variable to “ibw” or “nwb” in the configuration file ``targets.py``. Make the necessary changes in the file depending on your data. You can also use other file types, such as the Axon Binary File format (abf) and MATLAB binary (mat) files, which use BluePyEfe's `reader <https://github.com/BlueBrain/BluePyEfe/blob/master/bluepyefe/reader.py>`_ functions. It will require modifying the ``configure_targets`` function accordingly. If your ephys data format is of any other type, don't hesitate to contact the package authors to implement its reader in BluePyEfe.

Extraction
~~~~~~~~~~

To perform the extraction, you will need an extraction config file `./config/extract_config/L5PC_config.json <./config/extract_config/L5PC_config.json>`_. This file will be automatically created before the extraction by the ``configure_targets`` function in ``./pipeline.py``, if you are using your own data, the function might need to be modified for your needs. This function relies on the parameters set in the ``./targets.py`` configuration file which contains:

* ``files_metadata``: Path to the ephys data files. Please ensure to set your file type (ibw or nwb) in the ``file_type`` variable.
* ``ecodes_metadata``: List of ecodes protocols (e.g. IDthresh) for which you want features to be extracted.
* ``protocols_rheobase``: The protocol to use to find the rheobase of the cell.
* ``targets``: List of dictionaries, where each entry contains the protocol within which the features are extracted at a specific amplitude.

Therefore, before proceeding, it is essential to edit ``./targets.py`` to accurately reflect your specific settings. Once ``./targets.py`` has been configured to your requirements, the ``configure_targets`` function will parse these settings and subsequently create the appropriate ``L5PC_config.json`` configuration file.

If you wish to use non-threshold based optimisation that instead uses the absolute values of currents (e.g. using "IDRest_1.0" instead of "IDRest_100"), then you need to add the following to the ``pipeline_settings`` in ``./config/recipes.json``:

.. code-block:: python

    "extract_absolute_amplitudes": true,

and remove the ``name_Rin_protocol`` and ``name_rmp_protocol`` entries.

Then, to create the extraction configuration file and run the extraction process execute the following command:

.. code-block:: shell

    python pipeline.py --step='extract' --emodel='L5PC'

Please make sure that the name of the e-model matches an entry of the file ``recipes.json``.

The results of the extraction (if all goes well), should appear at the path mentioned in the entry ``features`` of the recipe. By convention, this path is usually set to ``./config/features/EMODEL_NAME.json``. The features file contains the ``efeatures`` and ``protocols``. The ``efeatures`` is a list of dictionaries, where each entry contains a feature associated with a specific protocol. ``protocols`` is also a list of dictionaries; each entry in this list contains the protocol's name, and amplitude, among other details.

If ``plot_extraction": true``, in ``pipeline_settings``, the plots will be in ``./figures/EMODEL_NAME/extraction/``. The folder contains figures for each cell that has been extracted. Each cell folder should have plots for:

* Individual features vs relative/absolute stimulus amplitude.
* Recordings plot for each protocol specified during extraction.

.. _default_std_deviation:

Note that our extraction process utilises traces from just one cell in this example, leading to limited sample sizes and occasionally, small or zero standard deviations (``original_std``) for certain features. This can inflate feature scores post-optimisation. To counteract this, a zero standard deviation during extraction is replaced by a default value specified in the ``default_std_deviation`` of the pipeline_settings as mentioned in the ``recipes.json``. Please refer to the `Configuration`_ section and ``pipeline_settings`` `pipeline_settings <https://github.com/BlueBrain/BluePyEModel/blob/977f206e1d0e17f4694890c03857beeb7df705d2/bluepyemodel/emodel_pipeline/emodel_settings.py#L117>`_ in BluePyEModel.

Each feature dictionary in the extracted features json file has another entry called threshold_efeature_std. This comes from the `threshold_efeature_std <https://github.com/BlueBrain/BluePyEModel/blob/977f206e1d0e17f4694890c03857beeb7df705d2/bluepyemodel/emodel_pipeline/emodel_settings.py#L173C13-L173C35>`_ in ``pipeline_settings`` (if not provided, it is ``null``). It can also be useful for small original_std .
For a complete description of the extraction process, its inner workings and settings please refer `README and examples of BluePyEfe on GitHub <https://github.com/BlueBrain/BluePyEfe/>`_.

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

Alternatively, you can use the notebook `./monitor_optimisation.ipynb <./monitor_optimisation.ipynb>`_ for better visualisation of the optimisation process.

Analysis
~~~~~~~~

Once a round of optimisation is finished, you might want to get the results from the checkpoint files (within the `./checkpoints` directory) generated by the optimisation process and plot the traces and scores of the models




To proceed with the analysis, execute the command provided below:

.. code-block:: shell

    python pipeline.py --step='analyse' --emodel='L5PC'

This particular command triggers a sequence of operations within the Python script, as it invokes the following methods:

.. code-block:: python

    pipeline.store_optimisation_results()
    pipeline.validation()
    pipeline.plot(only_validated=False)

These methods, called in succession, are responsible for storing the results of the optimisation, validating the e-models (testing the model on protocols unseen during optimisation), and then plotting the data.

The validation protocols are specified in the ``pipeline_settings`` dictionary of ``./config/recipes.json`` under the key ``validation_protocols``.

The analysis of each optimised model is stored in the file ``./final.json``. Here's a description of some of the entries of the ``final.json`` file:

* ``score``: global z-score of the optimised e-model. It is the sum of z-scores of all e-features used during optimisation. Validation e-feature scores are not added to this score.
* ``parameters``: best hall of fame parameters of the optimised e-model
* ``fitness``: z-score of each optimised e-feature
* ``features``: the numerical value of each e-feature
* ``validation_fitness``: z-scores of each validation e-feature
* ``validated``: whether the model has been validated, This field can have 3 values:

    - ``None``, the model has not yet been through validation
    - ``False``, the model did not pass validation successfully.
    - ``True``, the model passed validation successfully.

* ``pdfs``: path to the pdf file containing the plots of the traces, scores and parameters distributions of the optimised e-model

The plots are stored in ``./figures/`` which contains the following subfolders:

* ``efeatures_extraction``: Contains separate figures for each e-feature, each drawn based on the specific protocol used for extraction.
* ``distributions``: Displays optimisation parameter distributions between the low and high optimisation bounds as specified in params.json. The figure depicts parameter variations of only the best individuals of each seed.
* ``optimisation``: Depicts the optimisation curve, highlighting optimisation progress over generations. It plots the minimum and average optimisation fitness scores versus the number of optimisation generations, alongside details such as the lowest score achieved, total generations completed, the specific evolutionary algorithm employed, and the final status of the optimisation procedure.
* ``parameter_evolution``: Illustrates the evolution of the parameters within the optimisation bounds over generations.
* ``scores``: Presents the feature scores of each optimised e-feature in terms of z-scores from the experimental e-feature mean value.
* ``traces``: Exhibits the traces derived from the resulting optimised e-model for each optimised and validated protocol.
* ``currentscape``: Currentscape plots (see section `Currentscape`_) for each optimisation protocol.
The folders: currentscape, distributions, scores and traces will contain figures within the ``all`` subfolder. If ``pipeline.plot(only_validated=True)``, only the validated models are plotted within the ``validated`` subfolder.

If you wish to interact with the e-models, please have a look at the notebook `./exploit_models.ipynb <./exploit_models.ipynb>`_.

Note that you may observe disproportionately large scores for some features. This phenomenon often originates from the relatively small standard deviations associated with the extraction of these particular features, which in turn, is frequently a consequence of utilising a smaller sample size. Smaller sample sizes tend to yield less diverse data, thereby restricting the variability and potentially skewing feature scores post-optimisation (refer to this `section <default_std_deviation_>`_).

Currentscape
************

Currentscape plots can also be plotted by BluePyEModel, along with the other analysis figures. To do so, you simply have to add ``"plot_currentscape": true,`` to the ``pipeline_settings`` dictionary of ``./config/recipes.json``. All currents are recorded in [pA]. The currentscape figures are created using the same recordings and are saved under ``./figures/EMODEL_NAME/currentscape``. If you want to customise your currentscape plots, you can pass a currentscape config to the ``pipeline_settings`` dictionary of ``./config/recipes.json`` under the key ``currentscape_config``. You can find more information about currentscape and its config `here <https://github.com/BlueBrain/Currentscape>`_.

The recordings of the voltage, as well as every available ionic current and ionic concentration can be saved locally to ``./recordings`` when setting ``save_recordings`` to ``true`` in the ``pipeline_settings``.

If you do not want to have mechanism-specific currents in the currentscape plots, but have e.g. whole ionic currents plotted, it is possible by putting the names of the variables you want to plot under ``["current"]["names"]`` in the currentscape_config.

Exporting
~~~~~~~~~

If you wish to use the models generated with BluePyEModel outside of Python, you will need to export them as hoc files. To export the models generated with BluePyEModel, you can use the following commands:

.. code-block:: shell

    python pipeline.py --step='export_hoc' --emodel='L5PC'

or

.. code-block:: shell

    python pipeline.py --step='export_sonata' --emodel='L5PC'

The first command creates the hoc files to run with NEURON locally. The second step creates hoc files to be used in bbp circuit building pipeline. Ensure that the mechanisms are compiled before running the commands.

Once the exportation is done, the hoc files as well as the morphology of the model will be stored in local directory ``./export_emodels_hoc`` and ``./export_emodels_sonata`` respectively. Additionally, the sonata folder will contain a sonata nodes.h5 file. However, most of the time, for circuit building, you will want to generalise the models to the morphologies of the circuit. For that, you will need to perform model management (MM), which is out of the scope of the present package (see `https://github.com/BlueBrain/BluePyMM <https://github.com/BlueBrain/BluePyMM>`_ or `https://github.com/BlueBrain/emodel-generalisation <https://github.com/BlueBrain/emodel-generalisation>`_ )


Running the example using Slurm
-------------------------------

The Slurm version of the pipeline parallels its local counterpart, yet it requires preliminary configuration.

Setting up the directory and git repo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we recommend that you copy the present directory and all of its content to the folder in which you will want to work.

Then, initialise a git repository in the present directory:
``git init .``

Finally, you can set up the virtual environment necessary for running BluePyEModel by using the command:

.. code-block:: shell

        ./create_venv.sh

Executing this script initiates the creation of a virtual environment in the `./myvenv` directory and proceeds with the installation of BluePyEModel within this isolated space. This ensures that the package is installed in a clean environment, thereby avoiding any potential conflicts with other packages.

Versioning the runs
~~~~~~~~~~~~~~~~~~~

As you are likely to perform several rounds of extraction, optimisation and analysis, each of the runs will be tracked using a string called ``iteration_tag`` or ``githash``.

At the beginning of each optimisation run, an archive of the present directory will be created and stored in ``./run/GITHASH/``. You can have a look at `./optimisation.sh <./optimisation.sh>`_ to see how this operation is performed.

This process will ensure that a copy of the code as used at the moment of the launch exists and that it remains unchanged even if you change the current directory to perform different optimisations.

The ``githash`` provided by this operation will uniquely characterise the run, and it will be logged in the ``./logs/opt_list.log`` file. This file contains the list of all the runs that have been performed and their corresponding ``githash``.

Running the different steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refer to `Running the example locally`_ for the configuration of the recipes and target files.

To facilitate the execution of the different steps of the pipeline on Slurm, we provide the following auxiliary scripts that can be executed in the following order:

.. code-block:: shell

    ./extract.sh
    ./optimisation.sh
    ./analysis.sh
    ./export_hoc.sh

Don't forget to configure the necessary variables within these scripts, including setting the ``OPT_EMODEL`` value and configuring the ``#SBATCH`` directives in the corresponding .sbatch script according to your job requirements.

For more details about the different steps, please refer to the `Running the example locally`_ section.

These scripts will also generate logs of the different steps for each run to track its progress and capture any issues that may arise during execution. These log files are stored in the ``./logs`` with a naming convention reflective of the operation and its corresponding job identifier (e.g., ``opt_jobid.log``). In addition to individual log files, each step maintains its own historical record (e.g., ``extract_list.log``, ``opt_list.log`` ``analyse_list.log``) . These files are also situated within the ``./logs`` directory, serving as cumulative logs that document the series of runs pertinent to that particular step. Please ensure to check these logs if you encounter issues during the pipeline execution.

When running the optimisation, the script will create several slurm jobs for different optimisation seeds and a githash associated with the run (keep it preciously!), However, if you lose it, you can retrieve the githash from the ``opt_list.log`` file associated with each run. The optimisation script also compiles the mod files, assuming they are in the ``./mechanisms`` directory. Note that BluePyEmodel will delete any existing compiled files folder in the home directory before initiating a new optimisation. This is done to ensure that the mechanisms are compiled again if there are any changes.

The optimisation usually takes between 2 and 72 hours depending on the complexity of the model. If the model is not finished after 24 hours, you will need to set the githash of the run in the ``RESUME`` variable within ``./optimisation.sh`` and run the script again.

Troubleshooting
---------------
Here are some of the issues that you may encounter during the execution of the pipeline and their potential solutions.

nrnivmodl: bad interpreter
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter the following error:

.. code-block:: shell

    bash: /myvenv/bin/nrnivmodl: bad interpreter: No such file or directory

Ensure that you have activated your virtual environment before running the script. You can do this using the source or . command, depending on your shell:

.. code-block:: shell

    source /path/to/myvenv/bin/activate

In some cases, particularly on certain operating systems or file systems, the error message you encountered can also occur if the path to the script or the virtual environment directory is too long.

Long file paths can lead to issues with file system limitations, and the operating system may not be able to locate the necessary files correctly.
If you suspect that the path length is causing the problem, you can try the following:

* Shorten the Path: If possible, shorten the directory structure or move the script and the virtual environment to a location with a shorter path.
* Use Symbolic Links: Consider using symbolic links to create shorter aliases for directories or files. This can help reduce the effective path length.

X11 forwarding
~~~~~~~~~~~~~~
When running on a remote computer, please note that X11 forwarding may cause issues during optimisation, as multiple NEURON instances are launched during the optimisation of an E-model. If the X11 (GUI) is present, it can prevent the successful launch of NEURON instances.
To address this, you can include the following line in your sbatch files to set the NEURON_MODULE_OPTIONS environment variable:

.. code-block:: shell

    export NEURON_MODULE_OPTIONS="-nogui"

This line is intended to prevent NEURON from sending any GUI info. An alternative solution would be to disable X11 forwarding altogether in your SSH session.