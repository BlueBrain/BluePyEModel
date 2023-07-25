#!/bin/bash

source ./myvenv/bin/activate

cd ..

python pipeline.py --step='extract' --emodel=L5PC
