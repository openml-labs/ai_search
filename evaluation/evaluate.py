import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from evaluation.evaluation_utils import EvaluationProcessor


eval_path = Path("../data/evaluation/")
processor = EvaluationProcessor(eval_path)
results_display = processor.run()
print(results_display)

# save the results to a csv file
results_display.to_csv("evaluation_results.csv")