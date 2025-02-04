#!/bin/bash

# this scripts creates inferences for prompts and test cases chat gpt came up with
# only do it 1 at a time, this is basically used for me to manually run functions
MAX_RETRY=3
models=(
    "qwen_coder_2.5 7b"
    "llama3.1_instruct 8b"
)

datasets=(
    "bigcode_python_fns"
    "oss"
    "evol"
)

set -- ${models[$1]} 
for dataset_name in ${datasets[@]}
do
    # oracle model
    python training_dataset/inference_generated_prompts.py --dataset_name=$dataset_name --model_name="qwen_coder_2.5" --model_type="32b" --sampling_method="greedy"
    for model in ${models[@]}
        do
        # training model
        set -- ${model} # $1 = model_name, $2=model_type
        python training_dataset/inference_generated_prompts.py --dataset_name=$dataset_name --model_name=${1} --model_type=${2} --sampling_method="best_of_n_top_p_sampling"
        done
done

echo "Finish Inferences"