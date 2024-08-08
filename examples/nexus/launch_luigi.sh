#!/bin/bash

#####################################################################
# Copyright 2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#####################################################################

deactivate
module purge all
source myvenv/bin/activate

emodel=EMODEL_NAME # e.g. "L5_TPC"
etype=ETYPE_NAME # e.g. "cAC"
mtype=MTYPE_NAME
ttype=TTYPE_NAME
species="mouse"
brain_region="SSCX"
iteration="XXXX-XX-XX"

mtype_formatted=${mtype// /__}
ttype_formatted=${ttype// /__}

bbp-workflow launch-bb5 -f --config=luigi.cfg bluepyemodel.tasks.emodel_creation.optimisation EModelCreation emodel=$emodel species=$species brain-region=$brain_region iteration-tag=$iteration etype=$etype mtype=$mtype_formatted ttype=$ttype_formatted