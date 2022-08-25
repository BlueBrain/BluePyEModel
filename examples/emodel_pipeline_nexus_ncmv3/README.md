# E-Model Building Pipeline with Luigi and Nexus for NCMv3

To create an e-model:

## 1. Setup the virtual environnement

To run this example, you will need to ask the HelpDesk to "setup your for personal OpenShift project" (see https://bbpteam.epfl.ch/project/spaces/display/BBPNSE/Workflow).

Then, to setup the venv and install all the needed packages, run:
```
kinit
./create_venv.sh
```
This script might ask for your username and password several times if you do not have a connection token setup in gitlab.

## 2. Configure the Luigi pipeline:
In luigi.cfg, modify the entries:

    - virtual-env: path to your virtual environnement
    - chdir: absolute path to the current working directory

## 3. Register the resources that will be used by the pipeline on the Nexus project (bbp/ncmv3);

To setup the resources needed by the pipeline such as efeature extraction targets and pipeline settings, run:
```
$VENV/bin/python pipeline.py --step=configure_nexus --emodel=EMODEL_NAME --iteration_tag=ITERATION_TAG --ttype=TTYPE
```
The user will be asked to confirm that he wished to deprecated resources present on Nexus, then he will be asked to enter his Nexus token (that can be obtained from https://bbp.epfl.ch/nexus/web/).

The EMODEL_NAME and TTYPE have to be valid names present in the gene map (e.g.: EMODEL_NAME=="L5_TPC:B_cAC", TTYPE="L4/5 IT_1").

The iteration_tag can be any string (no space allowed). This variable allows the user to run different tests or iterations in parrallel on the same emodel. All Nexus resources related to BPEM will be tagged with it and when running the pipeline, it will only use the resources having the matching iteration_tag. Note that the `iteration_tag` specified here in luig.cfg has to match the `iteration_tag` informed when running pipeline.py. If a different `iteration_tag` was used in pipeline.py, the pipeline will crash as BPEM will not find the expected resources.

Also note that the ephys trace files as well as the targets (e-features and protocols) used for the present   example are hardcoded in the file targets.py.

To setup the configuration of the model, that include the model's channels, parameters and parameter distributions, either you can create the configuration based on gene data:
```
$VENV/bin/python pipeline.py --step=configure_model_from_gene --emodel=EMODEL_NAME --iteration_tag=ITERATION_TAG --ttype=TTYPE
```
Or through a legacy json file:
```
$VENV/bin/python pipeline.py --step=configure_model_from_json --emodel=EMODEL_NAME --iteration_tag=ITERATION_TAG --ttype=TTYPE
```
In this case, you can choose which file to use to initialize the configuration in pipeline.py, line 235. As of now it uses Darshan's personal configuration files, which are known to work.

Note that it is unlikely that the model created from gene would work out of the box. Therefore, if you wish to modify the gene based configuration before proceeding with model optimisation, you can get the configuration from Nexus and modify it before proceeding further. The jupyter notebook edit_neuron_model_configuration.ipynb explains how to do so.

The goal for the future will be for the pipeline and model to be configured through WebUI rather than using python script.

## 4. Run the Luigi pipeline:

Execute:
```
source myvenv/bin/activate
kinit
./launch_luigi.sh "EMODEL_NAME" "TTYPE" "ITERATION_TAG"
```

The pipeline should start and a few minutes later, the following message will appear: "No more log messages received, connection timed out." After that, to monitor the progress of the pipeline, a webapp can be accessed by first executing:
```
bbp-workflow webui -o
```
And then opening in a browser the url returned by this command.

If an error happens during the execution of the workflow, the command ./launch_luigi.sh "EMODEL_NAME" "TTYPE" "ITERATION_TAG" can be run again and the workflow will restart from the latest step. If the error persists, please refer to the following Troubleshooting section or contact Tanguy.

## 5. Exploitation of the results:

Once the pipeline ran successfully, the final emodels are saved in the Nexus project (here: bbp/ncmv3) as resources of type `EModel`.

To check that the models were optimised successfully, you can refer to the figures created in ./figures/EMODEL_NAME/.
The `optimisation` subfolder contains plots of the fitness versus number of generations run by the optimiser, while the `trace` and `scores` subfolder contain the voltage traces and efeatures scores for the emodels.

If you wish to run other protocol on a model or investigate the final model yourself, an example of how to do so is available in the notebook exploit_model.ipynb.

## Troubleshooting

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
1. You launched the pipeline with an iteration_tag different from the one you  specified when configurating the pipeline. If that's the case, edit your luigi.cfg and inform the correct iteration_tag.
2. It happens from time to time that nexus forge fails to get a result even when a matching resource exists. If that's the case, launch the pipeline again (./launch_luigi.sh "EMODEL_NAME" "TTYPE" "ITERATION_TAG") that will restart from where it stopped. 
