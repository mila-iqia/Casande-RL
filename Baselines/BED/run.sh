#!/bin/bash

# Code is tested using Python 3.7
# Required package: Numpy

python main.py \
    --input_dir='../dataset/' \
    --dataset_name=SymCAT \
    --solver=BED \
    --test_size=10000 \
    --search_depth=1 \
    --param_search
