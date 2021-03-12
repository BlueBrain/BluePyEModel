#!/bin/bash

#SBATCH --partition=prod
#SBATCH --time=0-00:15:00
#SBATCH --account=proj38
#SBATCH --output=logs/report_%j.log
#SBATCH --error=logs/report_%j.log
#SBATCH --ntasks=4
#SBATCH --signal=B:USR1@650

# luigi --module bluepyemodel.tasks.emodel_creation.optimisation ExtractEFeatures --local-scheduler

# export IPYTHONDIR=${PWD}/.ipython
# export IPYTHON_PROFILE=opt
# echo $IPYTHON_PROFILE
# rm -rf ${IPYTHONDIR}
# ipcontroller --init --ip='*' --sqlitedb --profile=${IPYTHON_PROFILE} --ping=30000 &
# sleep 5
# srun ipengine --profile=${IPYTHON_PROFILE} --timeout=1000  &

# catch signal here and send a SIGUSR1 to the python script
# because signals are usually not sent to other processes (here python) (except SIGINT)
# http://mywiki.wooledge.org/SignalTrap#When_is_the_signal_handled.3F
# trap 'echo signal recieved!; kill "${PID}"; wait "${PID}"; handler' USR1 SIGTERM
trap 'kill -SIGUSR1 "${PID}"; wait "${PID}"; handler' USR1

# execute python script in the background to be able to catch signals with trap
python -m bluepyemodel.tasks.luigi_custom &
PID="$!"
wait "${PID}"
