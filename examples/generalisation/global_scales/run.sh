#!/usr/bin/bash -l
#rm -r figures
#rm -r models
#rm -r tmp

export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

unset PMI_RANK
export USE_NEURODAMUS=True

if [[ $USE_NEURODAMUS ]]
then
    module load unstable neurodamus-neocortex
    module unload python
fi


if [[ $1 == ipyp ]]
then
    IPYP='ais'
    export IPYTHONDIR="`pwd`/.ipython"
    rm -rf ${IPYTHONDIR}
    ipcontroller --init --ip='*' --sqlitedb --profile=${IPYP} --ping=3000 &
    sleep 2
    srun ipengine --profile=${IPYP} --timeout=1000  &
    export IPYTHON_PROFILE=${IPYP}
elif [[ $1 == dask ]]
then
    . ~/base/bin/activate
    module purge
    module load unstable hpe-mpi
    module load unstable py-dask-mpi/2.21.0

    SCHEDULER_PATH="`pwd`/.scheduler.json"
    PWD=$(pwd)
    LOGS=$PWD/logs
    mkdir -p $LOGS

    export DASK_DISTRIBUTED__LOGGING__DISTRIBUTED="warning"
    export DASK_DISTRIBUTED__WORKER__USE_FILE_LOCKING=False
    export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False
    export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False
    export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80
    export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95
    export DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD=spawn
    export DASK_DISTRIBUTED__WORKER__DAEMON=False
    export PARALLEL_DASK_SCHEDULER_PATH=${SCHEDULER_PATH} 

    mpirun dask-mpi --scheduler-file ${SCHEDULER_PATH} &

    sleep 10
fi

luigi --module bluepyemodel.tasks.generalisation.run RunAll  --local-scheduler --log-level INFO  \
    #--emodels '["cADpyr_L5TPC"]' \
    #--rerun-emodels '["cADpyr_L3TPC"]' \
