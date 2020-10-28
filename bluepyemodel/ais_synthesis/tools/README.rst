Running using Dask
==================

This is an example of a sbatch script that can be adapted to execute the script using multiple nodes and workers.

Dask variables are not strictly required, but highly recommended, and they can be fine tuned.


.. code:: bash

    #!/bin/bash -l
    #SBATCH --nodes=2             # Number of nodes
    #SBATCH --time=24:00:00       # Time limit
    #SBATCH --partition=prod      # Submit to the production 'partition'
    #SBATCH --constraint=cpu      # Constraint the job to run on nodes with/without SSDs. If you want SSD, use only "nvme". If you want KNLs then "knl"
    #SBATCH --exclusive           # only if you need to allocate whole node
    #SBATCH --mem=0
    #SBATCH --ntasks-per-node=72  # no of mpi ranks to use per node
    #SBATCH --account=projXX      # your project number
    #SBATCH --job-name=myscript
    #SBATCH --output=myscript_out_%j
    #SBATCH --error=myscript_err_%j
    set -e
    
    module purge
    module load unstable hpe-mpi
    module unload unstable
    
    unset PMI_RANK  # for neuron
    
    # Dask configuration
    export DASK_DISTRIBUTED__LOGGING__DISTRIBUTED="info"
    export DASK_DISTRIBUTED__WORKER__USE_FILE_LOCKING=False
    export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False  # don't spill to disk
    export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False  # don't spill to disk
    export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80  # pause execution at 80% memory use
    export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95  # restart the worker at 95% use
    export DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD=spawn
    export DASK_DISTRIBUTED__WORKER__DAEMON=True
    # Reduce dask profile memory usage/leak (see https://github.com/dask/distributed/issues/4091)
    export DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL=10000ms  # Time between statistical profiling queries
    export DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE=1000000ms  # Time between starting new profile
    
    # Split tasks to avoid some dask errors (e.g. Event loop was unresponsive in Worker)
    export PARALLEL_BATCH_SIZE=1000
    
    # Script parameters
    OUTPUT="/path/to/mecombo_emodel.tsv"
    CIRCUIT_CONFIG="/gpfs/bbp.cscs.ch/project/proj68/circuits/Isocortex/20190307/CircuitConfig"
    MORPHOLOGY_PATH="/gpfs/bbp.cscs.ch/project/proj68/circuits/Isocortex/20190307/morphologies"
    RELEASE_PATH="emodel_release"
    N_CELLS=100
    MTYPE="L5_TPC:A"
    
    # load the virtual env (alternatively, load the required modules)
    source ~/venv/3.7.4-BluePyEModel/bin/activate
    
    srun -v \
    BluePyEModel -v get_me_combos_parameters \
    --circuit-config "$CIRCUIT_CONFIG" \
    --morphology-path "$MORPHOLOGY_PATH" \
    --release-path "$RELEASE_PATH" \
    --output "$OUTPUT" \
    --n-cells "$N_CELLS" \
    --mtype "$MTYPE" \
    --parallel-lib dask


