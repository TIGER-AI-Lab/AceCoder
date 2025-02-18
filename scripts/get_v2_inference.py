import fire
import json
import datasets
from llm_engines import LLMEngine
from acecoder import evaluate_test_cases
from pathlib import Path
from collections import Counter

def main(
    dataset_path="CodeDPO/AceCoderV2-mini-processed",
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    n=16,
    temperature=1.0,
    top_p=1.0,
    num_gpu_per_worker=1,
    num_workers=2,
    n_eval_workers=16,
    binary=False,
    output_path=None,
    max_samples=2000,
):
    
    dataset = datasets.load_dataset(dataset_path, split='train')
    
    if max_samples:
        dataset = dataset.select(range(max_samples))
    
    if not output_path:
        output_path = Path("results/inference_results") / f"{model_name.replace('/', '-')}.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not output_path.exists():
    
        questions = dataset['question']
        
        engine = LLMEngine()
        engine.load_model(
            model_name=model_name,
            num_gpu_per_worker=num_gpu_per_worker,
            num_workers=num_workers,
            engine="sglang",
        )
        
        outputs = engine.batch_call_model(model_name, questions, n=n, temperature=temperature, top_p=top_p)
        
        dataset = dataset.add_column("outputs", [[x] if isinstance(x, str) else x for x in outputs])
            
        samples = []
        for i, item in enumerate(dataset):
            task_id = item['id']
            test_case = item['tests']
            outputs = item['outputs']
            for j, output in enumerate(outputs):
                samples.append({
                    'task_id': task_id,
                    'prompt': item['question'],
                    'output': output,
                    'tests': test_case,
                    '_identifier': f"{task_id}_{i}_{j}"
                })
        
        all_samples_results, pass_rates = evaluate_test_cases(samples, n_workers=n_eval_workers, test_details=not binary)
        all_eval_results = [x["eval_results"] for x in all_samples_results]
        all_extracted_solutions = [x["solution"] for x in all_samples_results]
        scores = pass_rates
        if binary:
            scores = [1 if x == 1 else 0 for x in scores] # if binary
        
        idx = 0
        grouped_scores = []
        grouped_eval_results = []
        grouped_extracted_solutions = []
        
        for i, item in enumerate(dataset):
            grouped_scores.append(scores[idx:idx+len(item['outputs'])])
            grouped_eval_results.append(all_eval_results[idx:idx+len(item['outputs'])])
            grouped_extracted_solutions.append(all_extracted_solutions[idx:idx+len(item['outputs'])])
            idx += len(item['outputs'])
        
        dataset = dataset.add_column("scores", grouped_scores)
        dataset = dataset.add_column("eval_results", grouped_eval_results)
        dataset = dataset.add_column("extracted_solutions", grouped_extracted_solutions)
        
        with open(output_path, "w") as f:
            json.dump([x for x in dataset], f, indent=4)
        print(f"Results saved to {output_path}")
    else:
        with open(output_path, "r") as f:
            dataset = json.load(f)  
        dataset = datasets.Dataset.from_list(dataset)
    
    print(dataset)
    # summary of the results
    i = 1
    while i <= n:
        pass_at_n = sum([
            any([x["pass_rate"] == 1.0 for x in item['eval_results'][:i]]) for item in dataset
        ]) / len(dataset)
        print(f"Pass at {i}: {pass_at_n*100:.2f}%")
        i *= 2
    pass_rates = [sum(x) / len(x) for x in dataset['scores']]
    print(f"Average pass rate: {sum(pass_rates) / len(pass_rates) * 100:.2f}%")
    
    # analysis of the error types
    status_counter = Counter()
    test_case_error_counter = Counter() 
    for item in dataset:
        for eval_result in item['eval_results']:
            status_counter[eval_result['status']] += 1
            for test_case_detail in eval_result['details']:
                test_case_error_counter[test_case_detail['reason']] += 1
    # sort the status by frequency
    status_counter = dict(sorted(status_counter.items(), key=lambda x: x[1], reverse=True))
    total_num = sum(status_counter.values())
    print("## Status:")
    for k, v in status_counter.items():
        print(f" - {k}: {v} ({v/total_num*100:.2f}%)")
            
    # sort the error types by frequency
    test_case_error_counter = dict(sorted(test_case_error_counter.items(), key=lambda x: x[1], reverse=True))
    total_num = sum(test_case_error_counter.values())
    print("## Error types:")
    for k, v in test_case_error_counter.items():
        print(f" - {k}: {v} ({v/total_num*100:.2f}%)")
            
            
    
    
if __name__ == "__main__":
    fire.Fire(main)
    