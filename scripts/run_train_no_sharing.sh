#!/bin/bash

config_base='./run_configs'
output_base='./output'
config='config1.yaml'
output='config1'
data_base='.'
data='release_train_patients.zip'
eval_data="release_train_patients.zip"
validation_data="release_validate_patients.zip"
workers=4
cuda_idx=0
cpu_list="0 1 2 3"
prefix=""

if [ $# -ge 1 ]
then
        data_base="$1"
fi

if [ $# -ge 2 ]
then
        config="$2"
        output=$(basename -- "$config")
        output="${output%.*}"
fi

if [ $# -ge 3 ]
then
        cuda_idx=$3
fi

if [ $# -ge 4 ]
then
        workers=$4
fi

if [ $# -ge 5 ]
then
        prefix="$5"
fi

if [ $# -ge 6 ]
then
        data="$6"
        eval_data="$6"
fi

if [ $# -ge 7 ]
then
        eval_data="$7"
        validation_data="$7"
fi

if [ $# -ge 8 ]
then
        validation_data="$8"
fi

poetry run python ./chloe/main_rl.py  --data "$data_base/$data" --eval_data "$data_base/$eval_data"  --end_training_eval_data "$data_base/$validation_data" --output "$output_base/$output$prefix" --config "$config_base/$config" --cuda_idx $cuda_idx --n_workers $workers
