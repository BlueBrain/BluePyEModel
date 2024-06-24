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

source ./myvenv/bin/activate

export OPT_EMODEL="L5PC" # Your e-model name
export RESUME="0" # Put "0" to start a new run or a githash to resume an optimisation

if [[ ${RESUME} == "0" ]]
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
echo "E-model: $OPT_EMODEL"

export RUNLISTPATH="`pwd`/logs/opt_list.log"
export SUBMITTIME=`date +%Y.%m.%d.%H.%M.%S`

if [ ! -f "${RUNDIR}/x86_64/special" ]; then
    cd ${RUNDIR}
    nrnivmodl mechanisms
    cd -
fi

for seed in {1..5}; do
    export OPT_SEED=${seed}
    sbatch -J "${JOBNAME}_${OPT_SEED}" ./optimisation.sbatch
done