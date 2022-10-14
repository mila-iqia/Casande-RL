#!/bin/bash

# Code is tested using Python 3.7
# Required package: Numpy

python main.py \
    --input_dir='./dataset/' \
    --dataset_name=DDXPlus_test \
    --solver=BED_BATCH \
    --search_depth=1 \
    --max_episode_len=30 \
    --threshold=0.01
