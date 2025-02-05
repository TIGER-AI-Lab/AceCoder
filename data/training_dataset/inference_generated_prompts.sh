#!/bin/bash

datasets=(
    "bigcode_python_fns"
    "oss"
    "evol"
)

for dataset_name in "${datasets[@]}"
do
    # oracle model
    python training_dataset/inference_generated_prompts.py --dataset_name=$dataset_name --model_name="qwen_coder_2.5" --model_type="32b" --sampling_method="greedy"
    sleep 5
    python training_dataset/inference_generated_prompts.py --dataset_name=$dataset_name --model_name="qwen_coder_2.5" --model_type="7b" --sampling_method="best_of_n_top_p_sampling"
done

echo "Finish Inferencing"