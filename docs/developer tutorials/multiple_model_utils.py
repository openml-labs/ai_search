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
from tqdm import tqdm


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
        
def aggregate_multiple_queries_and_count(
    queries, qa_dataset, config, data_metadata,group_cols=["did", "name"], sort_by="query", count=True, apply_llm_before_rag = False
) -> pd.DataFrame:
    """
    Description: Aggregate the results of multiple queries into a single dataframe and count the number of times a dataset appears in the results
    """
    combined_df = pd.DataFrame()
    for query in tqdm(queries, total=len(queries)):

        response_parser = ResponseParser(query_type = config["type_of_data"], apply_llm_before_rag = apply_llm_before_rag)
        
        result_data_frame, _ = get_result_from_query(
            query=query, qa=qa_dataset, type_of_query="dataset", config=config
        )
        response_parser.rag_response = {"initial_response": list(result_data_frame["id"].values)}

        response_parser.fetch_llm_response(query)
        result_data_frame = response_parser.parse_and_update_response(data_metadata)

        result_data_frame["query"] = query
        result_data_frame["llm_model"] = config["llm_model"]
        result_data_frame["embedding_model"] = config["embedding_model"]
        combined_df = pd.concat([combined_df, result_data_frame])
    if count:
        combined_df = (
            combined_df.groupby(group_cols)
            .count()
            .reset_index()
            .sort_values(by=sort_by, ascending=False)
        )

    return combined_df

# %% [markdown]
# ## Running the steps
# - Create an experiment directory
# - Save a config file with the models and the queries in the experiment directory
# - Download openml data for each dataset and format into a string
# - Create vectorb and embed the data
# - Get the predictions for each model for a list of queries and evaluate the performance
# - (note) At the moment, this runs for a very small subset of the entire data. To disable this behavior and run on the entire data, set ```config["test_subset_2000"] = False```

def run_experiments(
    config,
    new_path,
    queries,
    list_of_embedding_models,
    list_of_llm_models,
    subset_ids,
    use_cached_experiment=False,
    enable_llm_results=False,
    apply_llm_before_rag = False
):
    for embedding_model in tqdm(
        list_of_embedding_models,
        desc="Embedding Models",
        total=len(list_of_embedding_models),
    ):
        for llm_model in tqdm(
            list_of_llm_models, desc="LLM Models", total=len(list_of_llm_models)
        ):
            # update the config with the new embedding and llm models
            config["embedding_model"] = embedding_model
            config["llm_model"] = llm_model

            # create a new experiment directory using a combination of the embedding model and llm model names
            if enable_llm_results == True:
                experiment_name = f"{process_embedding_model_name_hf(embedding_model)}_{process_llm_model_name_ollama(llm_model)}"

                if apply_llm_before_rag == True:
                    experiment_name += "_llm_before_rag"
                
                elif apply_llm_before_rag == False:
                    experiment_name += "_llm_after_rag"

                elif apply_llm_before_rag == None:
                    experiment_name += "_llm_none"
            else:
                # create a new experiment directory using the embedding model name
                experiment_name = f"{process_embedding_model_name_hf(embedding_model)}_llm_none"

            
            experiment_path = new_path / Path(f"../data/experiments/{experiment_name}")

            if use_cached_experiment and os.path.exists(experiment_path/"results.csv"):
                print(
                    f"Experiment {experiment_name} already exists. Skipping... To disable this behavior, set use_cached_experiment = False"
                )
                continue
            else:
                # create the experiment directory if it does not exist
                os.makedirs(experiment_path, exist_ok=True)

                # update the config with the new experiment directories
                config["data_dir"] = str(experiment_path)
                config["persist_dir"] = str(experiment_path / "chroma_db")

                # save training details and config in a dataframe
                config_df = pd.DataFrame.from_dict(config, orient="index").reset_index()
                config_df.columns = ["Hyperparameter", "Value"]
                config_df.to_csv(experiment_path / "config.csv", index=False)

                # load the persistent database using ChromaDB
                client = chromadb.PersistentClient(path=config["persist_dir"])

                # Run "training"
                qa_dataset, _ = setup_vector_db_and_qa(
                    config=config,
                    data_type=config["type_of_data"],
                    client=client,
                    subset_ids=subset_ids,
                )
                data_metadata_path = Path(config["data_dir"]) / "all_dataset_description.csv"
                data_metadata = pd.read_csv(data_metadata_path)
                combined_df = aggregate_multiple_queries_and_count(
                    queries,
                    qa_dataset=qa_dataset,
                    config=config,
                    data_metadata=data_metadata,
                    group_cols=["id", "name"],
                    sort_by="query",
                    count=False,
                    apply_llm_before_rag = apply_llm_before_rag
                )
                combined_df.to_csv(experiment_path / "results.csv")


def get_dataset_queries(subset_ids, query_templates, merged_labels):
    # get the dataset ids we want out evaluation to be based on
    X_val = []
    y_val = []

    for id in subset_ids:
        for query in query_templates:
            for label in merged_labels[id]:
                X_val.append(query + " " + label)
                y_val.append(id)

    return pd.DataFrame({"query": X_val, "id": y_val}).sample(frac=1)


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
