#!/bin/bash

source ./myvenv/bin/activate

export OPT_EMODEL="L5PC"
export GITHASH="YOUR_GITHASH_HERE"

export RUNLISTPATH="`pwd`/logs/analyse_list.txt"

sbatch -J analysis_${OPT_ETYPE}_${GITHASH} ./analysis.sbatch
