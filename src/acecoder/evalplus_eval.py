# The MIT License
#
# Copyright (c) OpenAI (https://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import itertools
import multiprocessing
import os
import time
from multiprocessing import Array, Value
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import threading

from ctypes import c_char, create_string_buffer
from evalplus.config import *
from evalplus.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
    TimeoutException
)

def compatible_eval_result(results: Dict) -> Dict:
    # compatibility
    for task_results in results["eval"].values():
        # update the "files" field to "nfiles"
        if "files" in task_results and "nfiles" not in task_results:
            task_results["nfiles"] = len(task_results.pop("files"))
    return results

# Example usage:
def read_string(arr, index, str_length=256):
    start = index * str_length
    # Read until null terminator or end of string slot
    raw = arr[start:start + str_length]
    return raw.split(b'\x00')[0].decode()

def write_string(arr, index, string, str_length=256):
    start = index * str_length
    buf = create_string_buffer(string[:str_length].encode(), str_length)
    arr[start:start + str_length] = buf.raw

# unbiased estimator from https://github.com/openai/human-eval
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"
MISSING_DEPENDENCY = "missing_dependency"
UNEXECUTED = "unexecuted"
SYNTAX_ERROR = "syntax_error"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3
_MISSING_DEPENDENCY = 4
_UNEXECUTED = 5
_SYNTAX_ERROR = 6

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _MISSING_DEPENDENCY: MISSING_DEPENDENCY, _UNEXECUTED: UNEXECUTED, _SYNTAX_ERROR: SYNTAX_ERROR, _UNKNOWN: None}


def query_maximum_memory_bytes() -> Optional[int]:
    # Disable functionalities that can make destructive changes to the test.
    # allow only 4GB memory usage
    maximum_memory_bytes = os.getenv(
        "EVALPLUS_MAX_MEMORY_BYTES", 4 * 1024 * 1024 * 1024
    )
    maximum_memory_bytes = min(int(maximum_memory_bytes), psutil.virtual_memory().total)
    if maximum_memory_bytes == -1:
        return None
    return maximum_memory_bytes


def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False

def unsafe_execute_assert(
    dataset: str,
    entry_point: str,
    code: str,
    assert_tests: List,
    time_limits,
    atol,
    fast_check,
    stat,  # Value
    details,  # Array
    code_error,
    tests_errors,  # Array
    progress,  # Value
):
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        reliability_guard(maximum_memory_bytes=query_maximum_memory_bytes())
        exec_globals = {}
        try:
            with swallow_io():
                exec(code, exec_globals)
                # fn = exec_globals[entry_point]

            for i, test_case in enumerate(assert_tests):
                with swallow_io():
                    try:
                        with time_limit(time_limits[i]):
                            exec(test_case, exec_globals)
                        details[i] = _SUCCESS
                    except ModuleNotFoundError as e:
                        details[i] = _MISSING_DEPENDENCY
                        write_string(tests_errors, i, str(e))
                    except SyntaxError as e:
                        details[i] = _SYNTAX_ERROR
                        write_string(tests_errors, i, str(e))
                    except TimeoutException as e:
                        details[i] = _TIMEOUT
                        write_string(tests_errors, i, str(e))
                    except Exception as e:
                        details[i] = _FAILED
                        write_string(tests_errors, i, str(e))
                        
                progress.value += 1 
                if details[i] != _SUCCESS and fast_check:
                    raise Exception("Fast check failed")

            stat.value = _SUCCESS
        except SyntaxError: 
            stat.value = _SYNTAX_ERROR
        except ModuleNotFoundError:
            stat.value = _MISSING_DEPENDENCY
        except BaseException as e:
            # if module not found error, pring it for debug.
            # if "No module named" in str(e):
            #     print(e)
            stat.value = _FAILED
            write_string(code_error, 0, str(e))
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

def untrusted_check_assert(
    dataset: str,
    code: str,
    entry_point: str,
    assert_tests: List[str],
    atol,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> Tuple[str, np.ndarray]:
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = min(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", 15), sum(time_limits)) + 1
    if not fast_check:
        timeout += 1  # extra time for data collection

    # shared memory objects
    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(assert_tests))])
    # errors is a list of strings
    # Method 2: Or if you need to initialize with spaces
    tests_errors = Array(c_char, b" " * (len(assert_tests) * 256))
    code_error = Array(c_char, b" " * 256)

    p = multiprocessing.Process(
        target=unsafe_execute_assert,
        args=(
            dataset,
            entry_point,
            code,
            assert_tests,
            time_limits,
            atol,
            fast_check,
            stat,
            details,
            code_error,
            tests_errors,
            progress,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    details = details[: progress.value] + [_UNEXECUTED] * (len(assert_tests) - progress.value)
    
    tests_errors = [read_string(tests_errors, i, 1024) for i in range(len(assert_tests))]
    tests_errors = [x if x.strip() else None for x in tests_errors]
    code_error = read_string(code_error, 0, 1024) if code_error[0] != 0 else None
    code_error = code_error if code_error.strip() else None
    
    details = [{"pass": x == _SUCCESS, "reason": _mapping[x], "error_message": tests_errors[i], "time_limit": time_limits[i]} for i, x in enumerate(details)]
    pass_rate = sum([x["pass"] for x in details]) / len(details) if details else 0

    if not stat:
        stat = TIMEOUT

    if stat == PASS:
        if len(details) != len(assert_tests) or not all([x["pass"] for x in details]):
            stat = FAIL
    
    result = {
        "status": stat,
        "code_error": code_error,
        "details": details,
        "pass_rate": pass_rate
    }
    
    return result