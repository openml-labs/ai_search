import glob
from pathlib import Path

import pandas as pd
from evaluation_utils import EvaluationProcessor
from tqdm import tqdm

eval_path = Path("../data/evaluation/")
processor = EvaluationProcessor(eval_path, sort_by=None)
results_display = processor.run()
print(results_display)

# save the results to a csv file
results_display.to_csv("evaluation_results.csv")
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 10))
sns.heatmap(results_display, annot=True, cmap="coolwarm", fmt="g")
plt.tight_layout()
plt.savefig("evaluation_results.png")
