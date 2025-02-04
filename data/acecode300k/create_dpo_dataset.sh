#!/bin/bash

python training_dataset/create_dataset/dpo/create_dpo_dataset.py
python training_dataset/create_dataset/dpo/convert_dataset_for_llama_factory_dpo.py
