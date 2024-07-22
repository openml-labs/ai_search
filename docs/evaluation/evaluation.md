# Evaluation of LLM models and techniques

## How to run
- Start the language server at the root of this repository with `./start_llm_service.sh` . This is important, do not skip it.
- Run `python run_all_training.py` to train all models (get data, create vector store for each etc)
- Run `python evaluate.py` to run all evaluations
- Results are found in in `./evaluation_results.csv` and `evaluation_results.png`

## How to add a new evaluation

- It is "pretty easy" to add a new evaluation. 
  - (Note that `training_utils.py` already overloads some classes from the original training. Which means that you can modify this to your hearts content without affecting the main code. Enjoy~)
  - Step 1: Find the method you want to override and overload the class/method in `training_utils.py`.
  - Step 2: Add some if statements in `class ExperimentRunner` to ensure you dont break everything.
  - Step 3: Follow the ExperimentRunner templates in `run_all_training.py` to add whatever you added in Step 2 as a new experiment.
    - Give it a custom name so it is easy to understand what happens
    - Do not worry, the experiments are cached and won't run again if you have run them before.
  - Step 4: If you changed something from config, make sure you reset it. Since the file runs in one go, it will affect the following experiments otherwise.

## How to add a new metric

- In `evaluation_utils.py`, go to `class EvaluationProcessor`, add a new function that calculates your metric. (You can use the templates provided)
- Update the metric in `self.metric_methods`
- While running the evaluation, add them to your metrics list :
```python
metrics = ["precision", "recall", "map"]
eval_path = Path("../data/evaluation/")
processor = EvaluationProcessor(eval_path, sort_by=None, metrics=metrics)
```