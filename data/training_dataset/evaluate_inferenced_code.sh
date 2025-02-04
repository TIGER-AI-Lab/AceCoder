#!/bin/bash

datasets=(
    "bigcode_python_fns"
    "oss"
    "evol"
)

for dataset_name in "${datasets[@]}"
do
    python training_dataset/evaluate_inferenced_code.py --model_name="qwen_coder_2.5" --model_type="32b" --dataset_name=$dataset_name --sampling_method="greedy" &
    python training_dataset/evaluate_inferenced_code.py --model_name="qwen_coder_2.5" --model_type="7b" --dataset_name=$dataset_name --sampling_method="best_of_n_top_p_sampling"&

done

wait # wait for all process to finish
echo "Finish evaluating inferenced code"