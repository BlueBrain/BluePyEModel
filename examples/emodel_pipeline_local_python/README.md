The present directory illustrates how to setup an e-model creation directory as well as how to run the different steps of the creation process on BB5.

### Setting up the directory and git repo

First, we recommend that you copy the present directory and all of its content to the folder in which you will want to work.

Once that is done you can create the virtual environment in which BluePyEModel will be installed:
```./create_venv.sh```

Then rename the file gitignore_template to .gitignore. This will avoid versioning unwanted files in the future.
```mv gitignore_template .gitignore```

Finally, initialize a git repository in the present directory:
```git init .```

### Versioning the runs

As you are likely to perform several rounds of extraction, optimisation and analysis, each of the run will be tracked using a string called `iteration_tag` or `githash`.

At the beginning of each optimisation run, an archive of the present directory will be created and stored in `./run/GITHASH/`. You can have a look at `./scripts/optimisation.sh` to see how this operation is performed.

This process will ensure that a copy of the code as used at the moment of the launch exists, and that is remains unchanged even if you change the current directory to perform different optimisations.

The `githash` provided by this operation will uniquely characterize the run, and we recommend that you keep a list of the githashes generated and the circumstances in which they were generated.

### Configuring your models

The present directory contains template mechanisms, morphologies, recipes and parameters files.
In order to configure the models that you want, you will have to:
- Copy the morphology you wish to use in the `morphologies` folder
- Copy the mechanisms (mod files) you wish to use in the `mechanisms` folder
- Create a json file containing the parameters of your model and put it in `./config/parameters/`.
- Create a json files containing the files_metadata, targets and protocols_rheobase used as targets for the extraction process in `./config/features/EMODEL_NAME_config.json` (for the format of this file section Extraction below).
- Create a new recipe in `./config/recipes.json` which should contain the paths to all the files mentioned above as well as the settings you wish to use when running the pipeline. You can have a look at the docstring of the class EModelPipelineSettings for a complete overview of all the settings available.

### Running the different steps

The main script used to execute the different steps of model building is the file `pipeline.py`. It contains the commands calling BluePyEModel to perform the operations related to extraction, optimisation, analysis and validation.

#### Extraction

To perform extraction, you will need an extraction config file as mentioned above. This file should contain the metadata of the ephys files that should be considered as well as the targets (protocols and efeatures) that should be extracted from the recordings present in these files.
It is recommended that you generate this file programmatically. The notebook `./extraction_configuration.ipynb` gives an example of how to do so.

Then, to run the extraction, inform the name of the emodel in `scripts/extract.sh` and execute the file.
The name of the emodel must match an entry of the file `recipes.json`.

The results of the extraction (if all goes well), should appear at the path mentioned in the entry `efeatures` of the recipe. By convention, this path is usually set to `./config/features/EMODEL_NAME.json`.
If you asked for the extraction to be plotted in the settings, the plots will be in `./figures/EMODEL_NAME/extraction/`.

For a complete description of the extraction process, its inner working and settings please refer the [README and examples of the branch BPE2 of BluePyEfe on GitHub](https://github.com/BlueBrain/BluePyEfe/tree/BPE2).

#### Optimisation

To perform optimisation, you will need to provide a morphology, mechanisms and a parameter configuration file in your recipe.

As optimisation is a costly operation, we will show here how to execute it in parallel using slurm and BB5.

First, you will need to compile the mechanisms, which can be done with the command:
```nrnivmodl mechanisms```
Then, inform your emodel name in `./scripts/optimisation.sh` and execute it.
This will create several slurm jobs for different optimisation seeds and the githash associated to the run (keep it preciously!).

The optimisation usually takes between 2 and 72 hours depending on the complexity of the model.
If the model is not finished after 24 hours, you will need to resume it manually by informing the githash of the run in `./scripts/optimisation.sh` and executing it again.
To monitor the state of the optimisation, please have a look at the notebook `./monitor_optimisation.ipynb`.

For a more in depth overview of the optimisation process please have a look at the [documentation and examples of the package BluePyOpt on GitHub](https://github.com/BlueBrain/BluePyOpt).

#### Analysis

Once a round of optimisation is finished, you might want to extract the results from the checkpoint files generated by the optimisation process and plot the traces and scores of the best models.

To do so, inform you emodel name and githash in `./script/analysis.sh` and execute it.

It will create a slurm job that will store the results in a local file called `final.json` as well as plot figures for these models that you will find in `./figures/EMODEL_NAME/`.

If you wish to interact with the models, please have a look at the notebook `./exploit_models.ipynb`.

Currentscape plots can also be plotted by BluePyEModel, along with the other analysis figures. To do so, you simply have to add `"plot_currentscape": true,` to the `"pipeline_settings"` dict of `./config/recipes.json`. All currents are recorded in [pA]. The currentscape figures are created using the same recordings, and are saved under `./figures/EMODEL_NAME/currentscape`. If you want to customise your currentscape plots, you can pass a currentscape config to the `"pipeline_settings"` dict of `./config/recipes.json` under the key `"currentscape_config"`. You can find more information about currentscape and its config [here](https://bbpgitlab.epfl.ch/cells/currentscape#about-the-config).

The recordings of the voltage, as well as every available ionic currents and ionic concentration can be saved locally to `./recordings` when setting `save_recordings` to `true` in the `pipeline_settings`. 

If you don't want to have mechanism-specific currents in the currentscape plots, but have e.g. whole ionic currents plotted, it is possible by putting the names of the variables you want to plot under `["current"]["names"]` in the currentscape_config.

#### Validation

If you wish to perform validation on your model (testing the model on protocols unseen during optimisation), you will have to mark these targets as such in your pipeline settings in the recipe file before efeature extraction.

Then, to run the validation, inform the emodel name and githash in `./script/analysis.sh` and execute it.
Once the validation is done, the models in your final.json wil have a field `passedValidation`.
This field can have 3 values:
- If it is None, that means the model did not go yet through validation.
- If it is False, it means the models did not pass validation successfully.
- If it is True, the model passed validation successfully.

As for the other steps, please have a look at the [docstring of the settings](https://bbpgitlab.epfl.ch/cells/bluepyemodel/-/blob/main/bluepyemodel/emodel_pipeline/emodel_settings.py) to configure the validation step.
