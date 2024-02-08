# E-Model Building Pipeline with Luigi and Nexus for MMB using point neuron

This example is intended to test the Luigi workflow by using a simpler morphology model (point neuron) which reduces the optimisation time.

## 1. Setup the virtual environment
To setup the venv and install all the needed packages, run:
```
kinit
./create_venv.sh
```
This script might ask for your username and password several times if you do not have a connection token setup in gitlab.

Then activate the virtual environment:
```
source myvenv/bin/activate
```

## 2. Configure the Luigi pipeline:
The ``luigi.cfg`` file contains the configuration for the Luigi pipeline. Make sure to set the `account` to the project number of your personal OpenShift project as well as the ``virtual-env`` and ``chdir`` to the path of your virtual environment (created by the previous step) and the current working directory respectively. Other parameters can be modified depending on your needs. Below is a description of several variables you may wish to customize:

- [DEFAULT] : settings section used by bbp-workflow.
    - ``account``: project name (e.g. proj72).
    - ``virtual-env``: path to your virtual environment.
    - ``chdir``: absolute path to the current working directory.
    - ``workers``: number of workers to be used by the pipeline, at the moment only 1 worker is supported.
    - ``time``: maximum time allowed for the pipeline to run.
- [Optimise] : settings section used for the optimisation.
    - ``node``: number of nodes to be used on the HPC.
    - ``time``: maximum time allowed to run the optimisation.
- [parallel]
    - ``backend``: select the backend for processing, which can be either 'ipyparallel' or 'multiprocessing'. If left unspecified, no parallelisation will be used.
- [EmodelAPIConfig]
    - ``nexus_project``: a valid Nexus project name to which the emodel should be uploaded.

For a detailed description of the configuration file, please refer to the [Luigi documentation](https://luigi.readthedocs.io/en/stable/configuration.html).

## 3. Register the resources that will be used by the pipeline on the Nexus project (bbp/mmb-emodels-for-synthesized-neurons);

Before running the Luigi pipeline, we need to register the following resources that will be used by the pipeline on the Nexus project:

- ``EModelPipelineSettings`` (EMPS): the pipeline settings of the e-model.
- ``ExtractionTargetsConfiguration`` (ETC): The extraction target configuration of the e-model and the links to the ephys data. This resource is created by parsing the ``targets.py`` using the  ``configure_targets`` function in pipeline.py.
- ``EModelConfiguration`` (EMC): the configuration of the e-model, which links to the morphology and mechanisms and stores a reformatted version of the parameters file of the e-model.

Those are the only required resources that need to be created before running the pipeline. The rest of the resources will be created by the Luigi pipeline.

So, to create and register the ``EModelPipelineSettings`` (EMPS) and ``ExtractionTargetsConfiguration`` (ETC), run:

```
python pipeline.py --step=configure_nexus --emodel=EMODEL_NAME --iteration_tag=ITERATION_TAG --etype=ETYPE
```

The user will be asked to confirm that he wishes to deprecate resources present on Nexus, then he will be asked to enter his Nexus token (that can be obtained from https://bbp.epfl.ch/nexus/web/).

The EMODEL_NAME and ETYPE have to be valid names present in the gene map (e.g.: EMODEL_NAME=="cADpyr", ETYPE=="cADpyr").

The iteration_tag can be any string (no space allowed). This variable allows the user to run different tests or iterations in parallel on the same e-model. All Nexus resources related to BPEM will be tagged with it and when running the pipeline, it will only use the resources having the matching iteration_tag. Note that the `iteration_tag` specified here in luig.cfg has to match the `iteration_tag` informed when running pipeline.py. If a different `iteration_tag` was used in pipeline.py, the pipeline will crash as BPEM will not find the expected resources.

Also, note that the ephys trace files, as well as the targets (e-features and protocols) used for the present example, are hardcoded in the file targets.py.

To set up the EModelConfiguration (EMC), which includes the model's channels, parameters and parameter distributions. You can either create the configuration based on gene data, or through a legacy json file. For that example, we will choose the latter option for faster execution. Advanced users can use their own by changing the path of ``filename`` in the ``configure_model`` function in ``pipeline.py``.

```
python pipeline.py --step=configure_model_from_json --emodel=EMODEL_NAME --iteration_tag=ITERATION_TAG --etype=ETYPE
```

If you choose to create the configuration based on gene data, you will need to provide a ttype, thus, ensure that you have also specified the ``TTYPE`` when running the ``configure_nexus`` step:

```
python pipeline.py --step=configure_model_from_gene --emodel=EMODEL_NAME --iteration_tag=ITERATION_TAG --etype=ETYPE --ttype=TTYPE
```

Note that it is unlikely that the model created from gene would work out of the box. Therefore, if you wish to modify the gene-based configuration before proceeding with model optimisation, you can get the configuration from Nexus and modify it before proceeding further. The jupyter notebook [edit_neuron_model_configuration.ipynb](../ncmv3//edit_neuron_model_configuration.ipynb) explains how to do so.

The goal for the future will be for the pipeline and model to be configured through WebUI rather than using python script.

## 4. Run the Luigi pipeline:
Set the variables (``emodel``, ``etype`` and ``iteration``) in [launch_luigi.sh](launch_luigi.sh/) to the same values as above (EMODEL_NAME, ETYPE and ITERATION_TAG) as well as ``species``, ``brain_region`` to the same values set in pipeline.py.

Then, execute:

```
source myvenv/bin/activate
kinit
./launch_luigi.sh
```

The pipeline should start and a few minutes later, the following message will appear: "No more log messages received, connection timed out." After that, to monitor the progress of the pipeline, a webapp can be accessed by first executing:
```
bbp-workflow webui -o
```
And then opening in a browser the url returned by this command.

If an error happens during the execution of the workflow, the command ./launch_luigi.sh can be run again and the workflow will restart from the latest step. If the error persists, please refer to the following [Troubleshooting](../emodel_pipeline_nexus_mmb/README.md/#troubleshooting) section or contact Ilkan or Aurélien.

## 5. Results:

Once the Luigi pipeline ran successfully, the following resources will be saved in the Nexus project indicated within the ``nexus_project`` variable in pipeline.py (for this example, it is set to ``bbp/mmb-emodels-for-synthesized-neurons``) along with the hoc file of the e-model:

- ``FitnessCalculatorConfiguration`` (FCC): the fitness calculator configuration of the e-model, which stores the features and protocols file of the e-model.
- ``EmodelScript`` (ES): the hoc file of the e-model.
- ``EModelWorkflow`` (EMW): the resource to which all the above resources are linked to, including the workflow state.
- ``EModel`` (EM): all the information related to an optimised e-model. It contains the final parameters of the e-model from final.json, and pdfs of the e-model distribution plots, features scores and e-model response traces. It also links to EModelWorflow.

In conclusion, here is the graph structure that will be generated on Nexus upon completing the entire pipeline:

```
    EModelWorkflow --> EModel
                        |
                        ├──> EModelPipelineSettings
                        |
                        ├──> ExtractionTargetsConfiguration
                        |       |
                        |       ├──> Trace1
                        |       ├──> ...
                        |       └──> TraceN
                        |
                        ├──> EModelConfiguration
                        |       |
                        |       ├──> Mechanism1
                        |       ├──> ...
                        |       └──> MechanismN
                        |       └──> Morphology
                        |
                        ├──> FitnessCalculatorConfiguration
                        |
                        └──> EModelScript
```

You can also check the graph structure of the resources created on [Nexus](https://bbp.epfl.ch/nexus/web/). Here is the link to an example [EModel resource](https://bbp.epfl.ch/nexus/web/bbp/mmb-emodels-for-synthesized-neurons/resources/https%253A%252F%252Fbbp.epfl.ch%252Fneurosciencegraph%252Fdata%252F44b60143-3ac0-4ec8-b091-49af9a2601ec)

To check that the models were optimised successfully, you can refer to the figures created in ``./figures/EMODEL_NAME/``.
The `optimisation` subfolder contains plots of the fitness versus number of generations run by the optimiser, while the `traces` and `scores` subfolders contain the voltage traces and efeatures scores for the e-models.

## Troubleshooting
Refer to the [Troubleshooting of BluePyEModel](https://github.com/BlueBrain/BluePyEModel/tree/main/examples/L5PC#troubleshooting) for common issues.

### When running launch_luigi.sh, if you see the error:

```
  STDOUT:

  STDERR:
error: You are not a member of project "USERNAME".
Your projects are:
* bbp-mooc-sim-neuro
* bbp-ou-cells
* bbp-ou-coreservices
* bbp-ou-nexus
* bbp-ou-nse
```

Solution: open a ticket with the helpdesk and ask them to "setup your for personal OpenShift project".

### When running launch_luigi.sh, if you see the error:

```
  RAN: /usr/bin/ssh -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes -o StreamLocalBindUnlink=yes -NT -R/tmp/2609079-sch.sock:localhost:8082 -R/tmp/2609079-agt.sock:/tmp/ssh-IlIhsVxNsOVr/agent.17 r4i3n27

  STDOUT:

  STDERR:
```

Solution: run the command "kinit" and provide your password.

### There is a failed task in the Luigi webapp:

Click on the red little "bug" button next to the task, it will display the log.

If the last line of the log reads something like: "failed to find resource for filter {...}", there are two possible causes:
1. You launched the pipeline with an iteration_tag different from the one you specified when configuring the pipeline. If that's the case, edit your luigi.cfg and inform the correct iteration_tag.
2. It happens from time to time that nexus forge fails to get a result even when a matching resource exists. If that's the case, launch the pipeline again, it will restart from where it stopped.
