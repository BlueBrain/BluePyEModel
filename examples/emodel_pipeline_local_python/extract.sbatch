#!/bin/bash

#SBATCH --account=proj000
#SBATCH --error=./logs/extract_%j.log
#SBATCH --output=./logs/extract_%j.log
#SBATCH --ntasks=10
#SBATCH --time=01:00:00
#SBATCH --constraint=cpu
#SBATCH --partition=prod

set -e
set -x

RUNTIME=`date +%Y.%m.%d.%H.%M.%S`
echo "${OPT_EMODEL},${SLURM_JOB_ID},${SLURM_NTASKS},${RUNTIME}" >> ${RUNLISTPATH}

export IPYTHON_PROFILE=extract_${SLURM_JOB_ID}_$(hostname)
export USEIPYP=1
export IPYTHONDIR="`pwd`/.ipython"

ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun ipengine --profile=${IPYTHON_PROFILE} --location=$(hostname) &
sleep 20


python pipeline.py --use_ipyparallel --step='extract' --emodel=${OPT_EMODEL}
