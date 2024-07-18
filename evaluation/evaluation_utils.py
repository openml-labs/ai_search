import glob

import pandas as pd
from tqdm.auto import tqdm

class EvaluationProcessor:
    def __init__(
        self, eval_path, metrics=["precision", "recall", "map"], sort_by="precision"
    ):
        self.eval_path = eval_path
        self.load_eval_queries = self.load_queries_from_csv()
        self.query_templates = self.load_query_templates()
        self.query_key_dict = self.create_query_key_dict()
        self.metrics = metrics
        self.sort_by = sort_by

    def run(self):
        """
        Description: Run the evaluation process and display the results
        
        """
        csv_files = self.load_result_files()
        results_df = self.generate_results(csv_files)
        results_display = self.display_results(results_df)
        return results_display

    def load_result_files(self):
        """
        Description: Find all the csv files in the evaluation directory
        
        """
        return glob.glob(str(self.eval_path / "*/*/results.csv"))

    def generate_results(self, csv_files):
        """
        Description: Load the results from the csv files, group them and compute metrics for each group. Then merge the results and sort them by the metric specified.
        """
        merged_df = pd.DataFrame()

        for exp_path in tqdm(csv_files):
            exp = pd.read_csv(exp_path).rename(columns={"did": "y_pred"})
            exp = self.preprocess_results(exp)

            grouped_results_for_y_true_and_pred = exp.groupby(
                ["embedding_model", "llm_model", "query", "llm_before_rag"]
            ).agg({"y_true": ",".join, "y_pred": ",".join})

            # add metrics
            for metric in self.metrics:
                if metric == "precision":
                    grouped_results_for_y_true_and_pred = self.add_precision(
                        grouped_results_for_y_true_and_pred
                    )
                elif metric == "recall":
                    grouped_results_for_y_true_and_pred = self.add_recall(
                        grouped_results_for_y_true_and_pred
                    )
                elif metric == "map":
                    grouped_results_for_y_true_and_pred = self.add_map(
                        grouped_results_for_y_true_and_pred
                    )

            # aggregate by computing the average of the metrics for each group
            grouped_results_for_y_true_and_pred = (
                grouped_results_for_y_true_and_pred.groupby(
                    ["embedding_model", "llm_model", "llm_before_rag"]
                ).agg({metric: "mean" for metric in self.metrics})
            )

            # merge with the results
            merged_df = pd.concat([merged_df, grouped_results_for_y_true_and_pred])

            # sort by metric
            if self.sort_by in self.metrics:
                merged_df = merged_df.sort_values(by=self.sort_by, ascending=False)
        return merged_df

    def load_queries_from_csv(self):
        """
        Description: Load the queries from the csv file
        
        """
        return pd.read_csv(self.eval_path / "merged_labels.csv")[
            ["Topics", "Dataset IDs"]
        ]

    def load_query_templates(self):
        """
        Description: Load the query templates from the txt file. This is used to generate the queries for the evaluation process. eg: {query_template} {query}
        {find me a dataset about} {cancer}
        """
        with open(self.eval_path / "query_templates.txt", "r") as f:
            query_templates = f.readlines()
        return [x.strip() for x in query_templates]

    def create_query_key_dict(self):
        """
        Description: Use the manual evaluation to create a dictionary of queries and their corresponding ground truth dataset ids. eg: Math,"45617,43383,2,45748"
        """
        query_key_dict = {}
        for template in self.query_templates:
            for row in self.load_eval_queries.itertuples():
                new_query = f"{template} {row[1]}".strip()
                if new_query not in query_key_dict:
                    query_key_dict[new_query.strip()] = row[2]
        return query_key_dict

    def preprocess_results(self, results_df):
        """
        Description: Preprocess the results dataframe by filling missing values and converting the columns to the correct data types.
        """
        results_df["llm_before_rag"] = results_df["llm_before_rag"].fillna("None")
        results_df["y_pred"] = results_df["y_pred"].astype(str)
        results_df["query"] = results_df["query"].str.strip()
        results_df["y_true"] = results_df["query"].map(self.query_key_dict)
        return results_df

    def add_precision(self, grouped_df):
        """
        Description: Compute the precision metric for each group in the dataframe
        """
        grouped_df["precision"] = [
            len(set(y_true).intersection(set(y_pred))) / len(set(y_pred))
            for y_true, y_pred in zip(grouped_df["y_true"], grouped_df["y_pred"])
        ]
        return grouped_df

    def add_recall(self, grouped_df):
        """
        Description: Compute the recall metric for each group in the dataframe
        
        """
        grouped_df["recall"] = [
            len(set(y_true).intersection(set(y_pred))) / len(set(y_true))
            for y_true, y_pred in zip(grouped_df["y_true"], grouped_df["y_pred"])
        ]
        return grouped_df

    def add_map(self, grouped_df):
        """
        Description: Compute the mean average precision metric for each group in the dataframe
        """
        grouped_df["map"] = [
            sum(
                [
                    len(set(y_true).intersection(set(y_pred[:i]))) / i
                    for i in range(1, len(set(y_pred)))
                ]
            )
            / len(set(y_true))
            for y_true, y_pred in zip(grouped_df["y_true"], grouped_df["y_pred"])
        ]
        return grouped_df

    def display_results(self, results_df):
        return pd.DataFrame(results_df)

