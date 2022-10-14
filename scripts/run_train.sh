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
prefix=""
plasma="/tmp/plasma"

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
        plasma="$3"
fi

if [ $# -ge 4 ]
then
        cuda_idx=$4
fi

if [ $# -ge 5 ]
then
        workers=$5
fi

if [ $# -ge 6 ]
then
        prefix="$6"
fi

if [ $# -ge 7 ]
then
        data="$7"
        eval_data="$7"
fi

if [ $# -ge 8 ]
then
        eval_data="$8"
        validation_data="$8"
fi

if [ $# -ge 9 ]
then
        validation_data="$9"
fi

poetry run python ./chloe/main_rl.py  --data "$data_base/$data" --eval_data "$data_base/$eval_data" --end_training_eval_data "$data_base/$validation_data" --output "$output_base/$output$prefix" --config "$config_base/$config" --cuda_idx $cuda_idx --n_workers $workers --shared_data_socket "$plasma"  --no_replace_if_present

