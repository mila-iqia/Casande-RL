#!/bin/bash

output_base='./output'
config='config1.yaml'
output='config1'
data_base='.'
eval_data="release_validate_patients.zip"
cuda_idx=0
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
        eval_data="$4"
fi

if [ $# -ge 5 ]
then
        output="$5"
fi

if [ $# -ge 6 ]
then
        prefix="$6"
fi

poetry run python ./chloe/eval_fly.py  --data "$data_base/$eval_data" --output "$output_base/$output$prefix" --config "$config" --cuda_idx $cuda_idx --batch_mode --batch_size 33112  --deterministic
