#!/bin/bash

deactivate
module purge all
source myvenv_cli/bin/activate

ttype=$2
ttype_formatted=${ttype// /__}

bbp-workflow launch-bb5 -f --config=luigi.cfg bluepyemodel.tasks.emodel_creation.optimisation EModelCreation emodel="$1" ttype=$ttype_formatted species=mouse brain-region=SSCX iteration-tag="$3"
