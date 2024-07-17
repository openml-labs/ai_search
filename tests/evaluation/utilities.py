# %% [markdown]
# # Tutorial on using multiple models for evaluation
# - This tutorial is an example of how to test multiple models on the openml data to see which one performs the best.
# - The evaluation is still a bit basic, but it is a good starting point for future research.

# %%
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import chromadb
import pandas as pd
import json

# change the path to the backend directory
sys.path.append(os.path.join(os.path.dirname("."), "../../backend/"))


# %%
from modules.llm import *
from modules.results_gen import get_result_from_query

# add modules from ui_utils
sys.path.append(os.path.join(os.path.dirname("."), "../../frontend/"))
from ui_utils import *
from tqdm.auto import tqdm


def process_embedding_model_name_hf(name: str) -> str:
    """
    Description: This function processes the name of the embedding model from Hugging Face to use as experiment name.

    Input: name (str) - name of the embedding model from Hugging Face.

    Returns: name (str) - processed name of the embedding model.
    """
    return name.replace("/", "_")


def process_llm_model_name_ollama(name: str) -> str:
    """
    Description: This function processes the name of the llm model from Ollama to use as experiment name.

    Input: name (str) - name of the llm model from Ollama.

    Returns: name (str) - processed name of the llm model.
    """
    return name.replace(":", "_")


# %% [markdown]
# ## Downloading the models
# - PLEASE MAKE SURE YOU HAVE DOWNLOADED OLLAMA (```curl -fsSL https://ollama.com/install.sh | sh```)

# %%
# download the ollama llm models

# os.system("curl -fsSL https://ollama.com/install.sh | sh")


def ollama_setup(list_of_llm_models: list):
    os.system("ollama serve&")
    print("Waiting for Ollama server to be active...")
    while os.system("ollama list | grep 'NAME'") == "":
        pass

    for llm_model in list_of_llm_models:
        os.system(f"ollama pull {llm_model}")


# overloading response parser


class ResponseParser(ResponseParser):
    def load_paths(self):
        """
        Description: Load paths from paths.json
        """
        with open("../../frontend/paths.json", "r") as file:
            return json.load(file)

    def parse_and_update_response(self, metadata):
        """
        Description: Parse the response from the RAG and LLM services and update the metadata based on the response
        """
        if self.rag_response is not None and self.llm_response is not None:
            if self.apply_llm_before_rag == False:
                filtered_metadata = metadata[
                    metadata["did"].isin(self.rag_response["initial_response"])
                ]
                llm_parser = LLMResponseParser(self.llm_response)
                llm_parser.subset_cols = ["did", "name"]

                if self.query_type.lower() == "dataset":
                    llm_parser.get_attributes_from_response()
                    return llm_parser.update_subset_cols(filtered_metadata)

            elif self.apply_llm_before_rag == True:
                llm_parser = LLMResponseParser(self.llm_response)
                llm_parser.subset_cols = ["did", "name"]
                llm_parser.get_attributes_from_response()
                filtered_metadata = llm_parser.update_subset_cols(metadata)

                return filtered_metadata[
                    filtered_metadata["did"].isin(self.rag_response["initial_response"])
                ]

            elif self.apply_llm_before_rag == None:
                # if no llm response is required, return the initial response
                return metadata
        else:
            return metadata


class ExperimentRunner:
    def __init__(
        self,
        config,
        eval_path,
        queries,
        list_of_embedding_models,
        list_of_llm_models,
        subset_ids,
        use_cached_experiment=False,
        custom_name=None,
    ):
        self.config = config
        self.eval_path = eval_path
        self.queries = queries
        self.list_of_embedding_models = list_of_embedding_models
        self.list_of_llm_models = list_of_llm_models
        self.subset_ids = subset_ids
        self.use_cached_experiment = use_cached_experiment
        self.custom_name = custom_name

    def aggregate_multiple_queries(self, qa_dataset, data_metadata):
        """
        Description: Aggregate the results of multiple queries into a single dataframe and count the number of times a dataset appears in the results.
        """

        types_of_llm_apply = [True, False, None]
        combined_results = pd.DataFrame()

        # Initialize the ResponseParser once per query type
        response_parsers = {
            apply_llm: ResponseParser(
                query_type=self.config["type_of_data"], apply_llm_before_rag=apply_llm
            )
            for apply_llm in types_of_llm_apply
        }

        for query in tqdm(self.queries):
            for apply_llm_before_rag in tqdm(types_of_llm_apply):
                response_parser = response_parsers[apply_llm_before_rag]

                result_data_frame, _ = get_result_from_query(
                    query=query,
                    qa=qa_dataset,
                    type_of_query="dataset",
                    config=self.config,
                )
                response_parser.rag_response = {
                    "initial_response": list(result_data_frame["id"].values)
                }

                response_parser.fetch_llm_response(query)
                result_data_frame = response_parser.parse_and_update_response(
                    data_metadata
                ).copy()[["did", "name"]]

                result_data_frame["query"] = query
                result_data_frame["llm_model"] = self.config["llm_model"]
                result_data_frame["embedding_model"] = self.config["embedding_model"]
                result_data_frame["llm_before_rag"] = apply_llm_before_rag

                # combined_results.append(result_data_frame)
                combined_results = pd.concat(
                    [combined_results, result_data_frame], ignore_index=True
                )

        # Concatenate all collected DataFrames at once
        # combined_df = pd.concat(combined_results, ignore_index=True)

        return combined_results

    def run_experiments(self):
        # across all embedding models
        for embedding_model in tqdm(
            self.list_of_embedding_models,
            desc="Embedding Models",
        ):
            main_experiment_directory = (
                self.eval_path/self.custom_name/ f"{process_embedding_model_name_hf(embedding_model)}"
            )
            os.makedirs(main_experiment_directory, exist_ok=True)

            # update the config with the new experiment directories
            self.config["data_dir"] = str(main_experiment_directory)
            self.config["persist_dir"] = str(main_experiment_directory / "chroma_db")

            # save training details and config in a dataframe
            config_df = pd.DataFrame.from_dict(
                self.config, orient="index"
            ).reset_index()
            config_df.columns = ["Hyperparameter", "Value"]
            config_df.to_csv(main_experiment_directory / "config.csv", index=False)

            # load the persistent database using ChromaDB
            client = chromadb.PersistentClient(path=self.config["persist_dir"])

            # Run "training"
            qa_dataset, _ = setup_vector_db_and_qa(
                config=self.config,
                data_type=self.config["type_of_data"],
                client=client,
                subset_ids=self.subset_ids,
            )

            # across all llm models
            for llm_model in tqdm(self.list_of_llm_models, desc="LLM Models"):
                # update the config with the new embedding and llm models
                self.config["embedding_model"] = embedding_model
                self.config["llm_model"] = llm_model

                # create a new experiment directory using a combination of the embedding model and llm model names
                experiment_name = f"{process_embedding_model_name_hf(embedding_model)}_{process_llm_model_name_ollama(llm_model)}"
                experiment_path = main_experiment_directory / experiment_name
                os.makedirs(experiment_path, exist_ok=True)

                if self.use_cached_experiment and os.path.exists(
                    experiment_path / "results.csv"
                ):
                    print(
                        f"Experiment {experiment_name} already exists. Skipping... To disable this behavior, set use_cached_experiment = False"
                    )
                    continue
                else:
                    data_metadata_path = (
                        Path(self.config["data_dir"]) / "all_dataset_description.csv"
                    )
                    data_metadata = pd.read_csv(data_metadata_path)

                    combined_df = self.aggregate_multiple_queries(
                        qa_dataset=qa_dataset,
                        data_metadata=data_metadata,
                    )

                    combined_df.to_csv(experiment_path / "results.csv")


def get_dataset_queries(subset_ids, query_templates, merged_labels):
    # get the dataset ids we want out evaluation to be based on
    X_val = []
    y_val = []
    labels = []

    for id in subset_ids:
        for query in query_templates:
            for label in merged_labels[id]:
                x = query.strip() + " " + label
                # check if the query is not already in the list
                if x not in X_val:
                    X_val.append(x)
                    y_val.append(id)
                    labels.append(label)

    return pd.DataFrame({"query": X_val, "id": y_val, "label": labels}).sample(frac=1)


def create_results_dict(csv_files, df_queries):
    # create a dictionary to store the results
    results_dict = {}
    for exp_path in csv_files:
        folder_name = Path(exp_path).parent.name
        exp = pd.read_csv(exp_path)
        # create y_pred
        exp["y_pred"] = exp["did"].astype(str)

        # for each row, get the true label from the df_queries dataframe
        for i, row in exp.iterrows():
            res = df_queries[df_queries["query"] == row["query"]].values[0][1]
            exp.at[i, "y_true"] = res

        # get unique queries
        all_queries = exp["query"].unique()

        # calculate number of correct and wrong predictions
        correct, wrong = 0, 0
        for query in all_queries:
            ypred = exp[exp["query"] == query]["y_pred"].unique()
            ytrue = exp[exp["query"] == query]["y_true"].unique()
            if ypred in ytrue:
                correct += 1
            else:
                wrong += 1
        results_dict[folder_name] = {"correct": correct, "wrong": wrong}
    return results_dict


# %%
