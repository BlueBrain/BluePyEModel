#!/bin/bash

source ./myvenv/bin/activate

export OPT_EMODEL="L5PC"
export RUNLISTPATH="`pwd`/logs/extract_list.txt"

source ./download_ephys_data.sh

#python pipeline.py --step='extract' --emodel='L5PC'
sbatch -J "extract_${OPT_EMODEL}"  ./extract.sbatch