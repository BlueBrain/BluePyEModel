#!/bin/bash

#####################################################################
# Copyright 2023, EPFL/Blue Brain Project

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

#SBATCH --account=proj000
#SBATCH --error=./logs/export_hoc_%j.log
#SBATCH --output=./logs/export_hoc_%j.log
#SBATCH --ntasks=10
#SBATCH --time=03:00:00
#SBATCH --constraint=cpu
#SBATCH --partition=prod

set -e
set -x

export IPYTHON_PROFILE=export_hoc_${SLURM_JOB_ID}_$(hostname)
export USEIPYP=1
export IPYTHONDIR="`pwd`/.ipython"
export NEURON_MODULE_OPTIONS="-nogui"

ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun ipengine --profile=${IPYTHON_PROFILE} --location=$(hostname) &
sleep 20

python pipeline.py --use_ipyparallel --step='export_hoc' --emodel=${OPT_EMODEL} --seed=${OPT_SEED}
