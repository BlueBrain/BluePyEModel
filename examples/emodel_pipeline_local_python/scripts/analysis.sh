#!/bin/bash

cd ..
source ./myvenv/bin/activate

export OPT_ETYPE="L5PC"
export GITHASH=YOUR_GITHASH

sbatch -J "analysis_${OPT_ETYPE}_${GITHASH}" ./scripts/analysis.sbatch
