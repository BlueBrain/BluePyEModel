#!/bin/bash

#SBATCH --account=proj130
#SBATCH --error=./logs/opt_%j.log
#SBATCH --output=./logs/opt_%j.log
#SBATCH --ntasks=20
#SBATCH --time=24:00:00
#SBATCH --constraint=cpu
#SBATCH --partition=prod

set -e
set -x

export IPYTHON_PROFILE=opt_${SLURM_JOB_ID}_$(hostname)
export USEIPYP=1
export IPYTHONDIR="`pwd`/.ipython"

ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun ipengine --profile=${IPYTHON_PROFILE} --location=$(hostname) &
sleep 20

python pipeline.py --use_ipyparallel --step='optimize' --emodel=${OPT_ETYPE} --seed=${OPT_SEED} --githash=${GITHASH}
