import pandas as pd
import glob
from tqdm import tqdm
from pathlib import Path


class EvaluationProcessor:
    def __init__(self, eval_path):
        self.eval_path = eval_path
        self.load_eval_queries = self.load_queries()
        self.query_templates = self.load_query_templates()
        self.query_key_dict = self.create_query_key_dict()

    def create_results_dict(self, csv_files):
        merged_df = pd.DataFrame()

        for exp_path in tqdm(csv_files):
            exp = pd.read_csv(exp_path).rename(columns={"did": "y_pred"})
            exp = self.preprocess_results(exp)

            # custom metrics from here
            exp = self.add_col_exists_in(exp)

            merged_df = pd.concat([merged_df, exp])

        return merged_df

    def load_queries(self):
        return pd.read_csv(self.eval_path / "merged_labels.csv")[
            ["Topics", "Dataset IDs"]
        ]

    def load_query_templates(self):
        with open(self.eval_path / "query_templates.txt", "r") as f:
            query_templates = f.readlines()
        return [x.strip() for x in query_templates]

    def create_query_key_dict(self):
        query_key_dict = {}
        for template in self.query_templates:
            for row in self.load_eval_queries.itertuples():
                new_query = f"{template} {row[1]}".strip()
                if new_query not in query_key_dict:
                    query_key_dict[new_query.strip()] = row[2]
        return query_key_dict

    def add_col_exists_in(self, df):
        df["exists_in"] = [
            any(x in y for x in y_pred) for y_pred, y in zip(df["y_pred"], df["y_true"])
        ]
        return df

    def preprocess_results(self, results_df):
        results_df["llm_before_rag"] = results_df["llm_before_rag"].fillna("None")
        results_df["y_pred"] = results_df["y_pred"].astype(str)
        results_df["query"] = results_df["query"].str.strip()
        results_df["y_true"] = results_df["query"].map(self.query_key_dict)
        results_df["y_true"] = results_df["y_true"].str.split(",")
        return results_df

    def process_results(self):
        csv_files = glob.glob(str(self.eval_path / "*/*/results.csv"))
        results_df = self.create_results_dict(csv_files)
        return results_df

    def display_results(self, results_df):
        grouped_value_counts = results_df.groupby(
            ["embedding_model", "llm_model", "llm_before_rag"]
        )["exists_in"].value_counts()
        return pd.DataFrame(grouped_value_counts)


# Example usage:
eval_path = Path("../../data/evaluation/")
processor = EvaluationProcessor(eval_path)
results_df = processor.process_results()
results_display = processor.display_results(results_df)
print(results_display)
