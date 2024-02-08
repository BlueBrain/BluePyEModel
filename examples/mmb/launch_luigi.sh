#!/bin/bash

deactivate
module purge all
source myvenv/bin/activate

emodel="cADpyr"
etype="cADpyr"
iteration="XX-XX-XXXX"
species="mouse"
brain_region="SSCX"

bbp-workflow launch-bb5 -f --config=luigi.cfg bluepyemodel.tasks.emodel_creation.optimisation EModelCreation emodel=$emodel species=$species brain-region=$brain_region iteration-tag=$iteration etype=$etype