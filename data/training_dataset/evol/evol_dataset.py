from inference.GetDatasets import get_dataset_from_huggingface


# https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K?row=0
def get_evol_dataset():
    data = get_dataset_from_huggingface("ise-uiuc/Magicoder-Evol-Instruct-110K")[
        "train"
    ]
    return data
