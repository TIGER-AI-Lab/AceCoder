#!/bin/bash

# this scripts creates inferences for prompts and test cases chat gpt came up with
# only do it 1 at a time, this is basically used for me to manually run functions
MAX_RETRY=1
models=(
    "llama3.1_instruct 8b"
    "qwen_coder_2.5 7b"
)
datasets=(
    "bigcode_python_fns"
    "oss"
    "evol"
)

for dataset_name in ${datasets[@]}
do
    python training_dataset/evaluate_inferenced_code.py --model_name=qwen_coder_2.5 --model_type=32b --dataset_name=$dataset_name --save_each_test_pass_status=True --sampling_method="greedy" &
    for model in "${models[@]}"
    do
        set -- ${model} # $1 = model_name, $2=model_type
        python training_dataset/evaluate_inferenced_code.py --model_name=${1} --model_type=${2} --dataset_name=$dataset_name &
    done
done

wait # wait for all process to finish
echo "Finish evaluating on ${dataset_name} with ${1} ${2}"