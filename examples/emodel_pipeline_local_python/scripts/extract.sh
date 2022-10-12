#!/bin/bash

source /gpfs/bbp.cscs.ch/project/proj130/singlecell/myvenv/bin/activate

cd ..

python pipeline.py --step='extract' --emodel=L5PC
