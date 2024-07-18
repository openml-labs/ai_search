# Evaluation of LLM models and techniques

## How to run
- Start the language server at the root of this repository with `./start_llm_service.sh` 
- Run `python run_all_training.py` to train all models (get data, create vector store for each etc)
- Run `python evaluate.py` to run all evaluations
- Results are found in in `./evaluation_results.csv`