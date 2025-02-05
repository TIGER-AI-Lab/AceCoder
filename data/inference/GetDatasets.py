from typing import Optional

import datasets


def get_dataset_from_huggingface(
    hugging_face_dataset_path: str,
    hugging_face_dataset_name: Optional[str] = None,
) -> datasets.dataset_dict.DatasetDict:
    """Get a hugging face dataset object. We will store this file locally so that we do not have to redownload this

    Parameter:
        hugging_face_dataset_path: a string that you cope from huggingface, ex: "gsm8k", "TIGER-Lab/MathInstruct"
        hugging_face_dataset_name: huggingface dataset name, for GSM8K we have "main" and "socratic"
    """
    data = datasets.load_dataset(hugging_face_dataset_path, hugging_face_dataset_name)
    return data
