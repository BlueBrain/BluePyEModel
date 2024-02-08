#!/bin/bash

deactivate
module purge all
source myvenv/bin/activate

emodel="cADpyr"
etype="cADpyr"
ttype="182_L4/5 IT CTX"
ttype_formatted=${ttype// /__}
species="mouse"
brain_region="SSCX"
iteration="XX-XX-XXXX"

bbp-workflow launch-bb5 -f --config=luigi.cfg bluepyemodel.tasks.emodel_creation.optimisation EModelCreation emodel=$emodel species=$species brain-region=$brain_region iteration-tag=$iteration etype=$etype ttype=$ttype_formatted