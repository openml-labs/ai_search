
from pathlib import Path
import glob
from utilities import *
import pickle
# %% [markdown]
# ## Load the results and evaluate

eval_path = Path("../../data/evaluation/")

# pickle.dump(df_queries, open(eval_path / "df_queries.pkl", "wb"))
df_queries = pickle.load(open(eval_path / "df_queries.pkl", "rb"))
# %%
# glob all csv files in the experiments directory
experiment_dir = Path(f"../../data/experiments/")
csv_files = glob.glob(str(experiment_dir / "*/results.csv"))

# %%
results_dict = create_results_dict(csv_files, df_queries)

# %%
print(pd.DataFrame.from_dict(results_dict, orient="index"))

# %%
pd.DataFrame.from_dict(results_dict, orient="index").to_csv(
    "../../data/experiments/results.csv"
)