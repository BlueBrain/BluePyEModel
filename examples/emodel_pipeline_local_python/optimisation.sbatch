#!/bin/bash

#SBATCH --account=proj000
#SBATCH --error=./logs/opt_%j.log
#SBATCH --output=./logs/opt_%j.log
#SBATCH --ntasks=72
#SBATCH --time=00:20:00
#SBATCH --constraint=cpu
#SBATCH --partition=prod

set -e
set -x

RUNTIME=`date +%Y.%m.%d.%H.%M.%S`
echo "${OPT_EMODEL},${OPT_SEED},${GITHASH},${SLURM_JOB_ID},${SLURM_NTASKS},${RUNTIME}" >> ${RUNLISTPATH}

export IPYTHON_PROFILE=optimisation__${SLURM_JOB_ID}_$(hostname)
export USEIPYP=1
export IPYTHONDIR="`pwd`/.ipython"

ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun ipengine --profile=${IPYTHON_PROFILE} --location=$(hostname) &
sleep 20

python pipeline.py --use_ipyparallel --step='optimise' --emodel=${OPT_EMODEL} --seed=${OPT_SEED} --githash=${GITHASH}
