# This folder contains code used to create a DPO Dataset out of different training dataset

## To reproduce, follow the following steps very closely
IMPORTANT: you should run all scripts in the REPO ROOT directory

1. Run 1_preprocessing.py, this script does preprocessing by filtering out python code not suitable for a typical "leetcode" style question. We want code which is in function or class form so it is easier to come up with questions. The output of this program will be in training_dataset/evol/data/evol_v1.jsonl
2. Run 2_generate_test_cases.py, this script calls chatGPT to create questions and test cases for each code sample we have. (Note that we have a ct parameter used for controlling how much we want to inference, set it to -1 if you want to inference all). The output of this program will be in training_dataset/evol/data/evol_v2.jsonl
3. Run 3_inference_on_prompts.sh, this script calls local models to make inferences on the questions. Currently we will call deepseek, codellama, codellama_python and codellama_instruct. The output of this program will be in "Inferenced output/evol/".
    - IMPORTANT: before running this script make sure you follow the setup procedure in the main repo's README and have conda correctly configured.
    - IMPORTANT: you might want to delete the entire folder of: "Inferenced output/evol/", this is because the code will not make new generation until the old generation have been deleted.
4. Run 4_evaluate_inferenced_code, this script will evaluate all the inferenced answers as well as the original answers, and calculate their accuracy for each test. The output of this program will be in training_dataset/evol/data/evol_v3.jsonl.
    - For inspection purpose you can stop here
5. Run 5_create_dataset.py, this script creates a dataset where each entry contains info such as the 2 programs and which one is better. The output of this program will be in training_dataset/evol/data/evol_dataset.jsonl.
    - It will also generate a summary.csv containing useful statistics to monitor accuracies