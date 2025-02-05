import json
import os
import time
from typing import Any, Dict, List


def chunking(lst: List[Any], n: int) -> List[List[Any]]:
    """Split a list into a list of list where each sublist is of size n"""
    if n <= 0:
        raise Exception(f"Are you fucking kidding me with n = {n}?")
    if len(lst) <= n:
        return [lst]
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """load a .jsonl file. Return a List of dictionary, where each dictionary is a line in the file"""
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} Does not exist!!!!")
    with open(file_path, "r") as f:
        lst = f.readlines()
    output = [json.loads(i) for i in lst]
    return output


def save_jsonl(file_path: str, content: List[Dict[Any, Any]]) -> None:
    """save a .jsonl file."""
    with open(file_path, "w") as f:
        for i in content:
            f.write(json.dumps(i) + "\n")


def append_jsonl(file_path: str, content: List[Dict[Any, Any]]) -> None:
    """append to a .jsonl file."""
    with open(file_path, "a") as f:
        for i in content:
            f.write(json.dumps(i) + "\n")


class MyTimer:
    """A simple timer class where you initialize it, then just call print_runtime everytime you want to time yourself"""

    def __init__(self) -> None:
        self.start = time.time()

    def print_runtime(self, message: str, reset_timer: bool = True) -> None:
        """Print the runtime, the output will be in the form of f"{message} took ..."

        Parameter:
            message: a string indicating what you have done
            reset_timer: whether to reset timer so that next call to this function will show the time in between print_runtime
        """
        runtime = time.time() - self.start
        minute = int(runtime / 60)
        seconds = runtime % 60
        if minute > 0:
            print(f"{message} took {minute} minutes {seconds} seconds")
        else:
            print(f"{message} took {seconds} seconds")

        if reset_timer:
            self.start = time.time()
