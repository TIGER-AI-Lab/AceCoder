import fire
import json
import datasets
from llm_engines import LLMEngine
from acecoder import evaluate_test_cases
from pathlib import Path

def main(
    dataset_path="CodeDPO/AceCoderV2-mini-processed",
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    n=1,
    temperature=1.0,
    top_p=1.0,
    num_gpu_per_worker=1,
    num_workers=1,
    n_eval_workers=1,
    binary=False,
    output_path=None
):
    
    dataset = datasets.load_dataset(dataset_path, split='train')
    
    dataset = dataset.select(range(100))
    
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
    
    if not output_path:
        output_path = Path("results/inference_results") / f"{model_name.replace('/', '-')}.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([x for x in dataset], f, indent=4)
    print(f"Results saved to {output_path}")
    
    # summary of the results
    i = 1
    while i <= n:
        pass_at_n = sum([
            any([x["pass_rate"] == 1.0 for x in item['eval_results'][:i]]) for item in dataset
        ]) / len(dataset)
        print(f"Pass at {i}: {pass_at_n*100:.2f}%")
        i *= 2
    print(f"Average pass rate: {sum(pass_rates) / len(pass_rates) * 100:.2f}%")
    
if __name__ == "__main__":
    fire.Fire(main)
    