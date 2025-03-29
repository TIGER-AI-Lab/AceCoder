import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
from termcolor import cprint
from tqdm import tqdm

from evalplus.sanitize import sanitize, code_extract

# from evalplus.config import *
from .evalplus_eval import (
    untrusted_check_assert,
)

DEFAULT_GT_TIME_LIMIT_FACTOR = 4.0
DEFAULT_MIN_TIME_LIMIT = 1.0

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]

def check_correctness_assert(
    task_id: int,
    completion_id: int,
    entry_point: str,
    solution: str,
    assert_tests: List[str],
    dataset: str=None,
    base_only=False,
    fast_check=False,
    identifier=None,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    extract_solution:bool=False,
    atol: int=1e-6,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)
    is_extracted = False
    if extract_solution:
        # base model may sometimes outputs too many "\n" and makes the code extraction too ** flow.
        # so we skip them if the number of lines > 500
        if not len(solution.split("\n")) > 500: 
            extracted_solution = code_extract(solution.encode('utf-8', 'ignore').decode('utf-8').replace('\x00', ''))    
            # if entry_point in _solution:
            #     solution = _solution
            is_extracted = True
        else:
            extracted_solution = solution
    ret = {
        "completion_id": completion_id,
        "task_id": task_id,
        "_identifier": identifier,
        "solution": extracted_solution,
        "n_tests": len(assert_tests),
    }
    eval_results = untrusted_check_assert(
        dataset,
        extracted_solution,
        entry_point,
        assert_tests,
        atol=atol,
        ref_time=[DEFAULT_MIN_TIME_LIMIT]*len(assert_tests),
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )
    if eval_results["status"] == "syntax_error" and is_extracted:
        # try to use the original solution
        ret['solution'] = solution
        eval_results = untrusted_check_assert(
            dataset,
            solution,
            entry_point,
            assert_tests,
            atol=atol,
            ref_time=[DEFAULT_MIN_TIME_LIMIT]*len(assert_tests),
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )
    ret["eval_results"] = eval_results
    
    return ret

def get_entry_point_from_test_case(test_case: str) -> str:
    """
    Get the entry point from the first test case.
    Args:
        test_case: a test case string, like "assert f(1) == 2"
    Returns:
        the entry point, like "f"
    """
    start_idx = test_case.find("assert ") + len("assert ")
    end_idx = test_case.find("(")
    return test_case[start_idx:end_idx]

def get_test_inputs_outputs_from_test_case(test_cases: List[str]) -> Tuple[List[str], List[str]]:
    """
    Get the inputs and outputs from the test cases.
    Args:
        test_cases: a list of test case strings
    Returns:
        a tuple of inputs and outputs
    """
    inputs = []
    outputs = []
    for test_case in test_cases:
        input_start_idx = test_case.find("(")
        assert input_start_idx != -1, f"Cannot find '(' in {test_case}"
        output_start_idx = test_case.find("==")
        if output_start_idx == -1:
            output_start_idx = test_case.rfind("is")
            assert output_start_idx != -1, f"Cannot find '==' or 'is' in {test_case}"
        output_start_idx += 2
        input_end_idx = test_case[:output_start_idx].rfind(")")
        assert input_end_idx != -1, f"Cannot find ')' in {test_case}"
        test_input = test_case[input_start_idx+1:input_end_idx].strip()
        try:
            if test_input:
                test_input = eval(test_input)
            else:
                test_input = []
        except:
            print(f"Cannot eval {test_input}")
            print(test_case)
            print(input_start_idx, input_end_idx)
            raise 
        inputs.append(test_input)
        assert output_start_idx != -1, f"Cannot find '==' in {test_case}"
        output = eval(test_case[output_start_idx:].strip())
        outputs.append(output)
    return inputs, outputs
    

def evaluate(
    samples: Union[str, List[Dict[str, Any]]],
    dataset: str = None,
    base_only: bool = False,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = True,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    output_file: Optional[str] = None,
    n_workers: Optional[int] = None,
    extract_solution: bool = True,
):
    if not n_workers:
        n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)
    
    if isinstance(samples, str) and os.path.exists(samples):
        result_path = samples.replace(".jsonl", ".eval_results.json")
    elif isinstance(samples, list):
        result_path = None
                
    if output_file is not None:
        result_path = output_file
        
    if result_path and os.path.isfile(result_path) and not i_just_wanna_run:
        print(f"Load from previous results from {result_path}")
        if result_path.endswith(".jsonl"):  
            with open(result_path, "r") as f:
                all_samples_results = [json.loads(line) for line in f]
        else:
            with open(result_path, "r") as f:
                all_samples_results = json.load(f)

    else:
        if isinstance(samples, str) and os.path.exists(samples):
            if samples.endswith(".jsonl"):
                with open(samples, "r") as f:
                    all_samples = [json.loads(line) for line in f]
            else:
                with open(samples, "r") as f:
                    all_samples = json.load(f)
        else:
            all_samples = samples
        
        dataset_hash = None

        _identifier_list = [x['_identifier'] for x in all_samples]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)  # task_id ->
            remainings = set()

            for sample in tqdm(all_samples, desc="Submitting samples"):
                task_id = sample["task_id"]
                # test_inputs, expected_output = get_test_inputs_outputs_from_test_case(sample["tests"])
                entry_point = get_entry_point_from_test_case(sample['tests'][0])
                solution = sample["output"]
                remainings.add(sample["_identifier"])
                args = (
                    task_id,
                    completion_id[task_id],
                    entry_point,
                    solution,
                    sample["tests"],
                    dataset,
                    base_only,
                    not test_details,  # fast_check
                    sample["_identifier"] if "_identifier" in sample else None,
                    min_time_limit,
                    gt_time_limit_factor,
                    extract_solution,
                )
                futures.append(executor.submit(check_correctness_assert, *args))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remainings), "Missing problems in unfinished"

            def stucking_checker():
                while remainings:
                    last_size = len(remainings)
                    time.sleep(20)
                    if last_size != len(remainings) or len(remainings) == 0:
                        continue
                    # Potential stucking
                    warn("No samples had finished testing in the last 20s")
                    warn(f"{len(remainings)} samples to be tested...")

            threading.Thread(target=stucking_checker).start()

            all_samples_results_identifier_map = {}
            for i, future in tqdm(enumerate(as_completed(futures)), total=n_samples):
                result = future.result()
                # except TimeoutError:
                #     print(f"Timeout for {i}th sample")
                #     result = {
                #         "completion_id": i,
                #         "task_id": task_id,
                #         "_identifier": sample["_identifier"],
                #         "solution": solution,
                #         "n_tests": len(sample["tests"]),
                #         "base": ["timeout", []]
                #     }
                remainings.remove(result["_identifier"])
                # result['pass_rate'] = result['eval_results']['pass_rate']
                all_samples_results_identifier_map[result["_identifier"]] = result
                eval_results[result["task_id"]].append(result)
            
            all_samples_results = [all_samples_results_identifier_map[x] for x in _identifier_list]
        # save the results
        if result_path:
            if result_path.endswith(".jsonl"):
                with open(result_path, "w") as f:
                    for result in all_samples_results:
                        f.write(json.dumps(result) + "\n")
            else:
                with open(result_path, "w") as f:
                    json.dump(all_samples_results, f, indent=4)
            print(f"Results saved to {result_path}")
            
    pass_rates = [x['eval_results']['pass_rate'] for x in all_samples_results]
    if __name__ == "__main__":
        print(f"Pass rate: {np.mean(pass_rates)}")
    else:
        return all_samples_results, pass_rates

def main():
    from fire import Fire

    Fire(evaluate)


if __name__ == "__main__":
    main()
