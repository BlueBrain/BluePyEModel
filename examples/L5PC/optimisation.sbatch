#!/bin/bash

#####################################################################
# Copyright 2023-2024 Blue Brain Project / EPFL

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
#SBATCH --error=./logs/opt_%j.log
#SBATCH --output=./logs/opt_%j.log
#SBATCH --ntasks=20
#SBATCH --time=12:00:00
#SBATCH --constraint=cpu
#SBATCH --partition=prod

set -e
set -x

RUNTIME=`date +%Y.%m.%d.%H.%M.%S`
echo "${OPT_EMODEL},${OPT_SEED},${GITHASH},${SLURM_JOB_NODELIST},${SLURM_JOB_ID},${SLURM_NTASKS},${RUNTIME}" >> ${RUNLISTPATH}

export IPYTHON_PROFILE=optimisation__${SLURM_JOB_ID}_$(hostname)
export USEIPYP=1
export IPYTHONDIR="`pwd`/.ipython"
export NEURON_MODULE_OPTIONS="-nogui"

ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun ipengine --profile=${IPYTHON_PROFILE} --location=$(hostname) &
sleep 20

python pipeline.py --use_ipyparallel --step='optimise' --emodel=${OPT_EMODEL} --seed=${OPT_SEED} --githash=${GITHASH}
