#!/bin/bash

cd ..
source ./myvenv/bin/activate

export OPT_ETYPE="L5PC"
export RESUME=0 # Put 0 to start a new run or "githash" too resume an optimisation

if [[ ${RESUME} == 0 ]]
then
    git add -A && git commit --allow-empty -a -m "Running optimization ${OPT_ETYPE}"
    export GITHASH=$(git rev-parse --short HEAD)
    git archive --format=tar --prefix=${GITHASH}/ HEAD | (if [ ! -d "./run" ]; then mkdir ./run; fi && cd ./run/ && tar xf -)

    JOBNAME=${GITHASH}
else
    export GITHASH=${RESUME}
    JOBNAME="Resume_${GITHASH}"
fi

export RUNDIR="./run/${GITHASH}"
echo "Githash: $GITHASH"
echo "E-model: $OPT_ETYPE"

if [ ! -f "${RUNDIR}/x86_64/special" ]; then
    cd ${RUNDIR}
    nrnivmodl mechanisms
    cd -
fi

for seed in {1..4}; do
    export OPT_SEED=${seed}
    sbatch -J "${JOBNAME}_${OPT_SEED}" ./scripts/optimisation.sbatch
done
