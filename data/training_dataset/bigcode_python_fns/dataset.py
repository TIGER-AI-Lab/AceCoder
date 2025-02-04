from inference.GetDatasets import get_dataset_from_huggingface


def get_bigcode_python_fn_dataset():
    data = get_dataset_from_huggingface("bigcode/stack-dedup-python-fns")["train"]
    return data
