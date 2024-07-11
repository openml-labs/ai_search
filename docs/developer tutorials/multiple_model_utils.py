# %% [markdown]
# # Tutorial on using multiple models for evaluation
# - This tutorial is an example of how to test multiple models on the openml data to see which one performs the best.
# - The evaluation is still a bit basic, but it is a good starting point for future research.

# %%
from __future__ import annotations
import os
import sys
import chromadb
from pathlib import Path
from datetime import datetime
import pandas as pd
# change the path to the backend directory
sys.path.append(os.path.join(os.path.dirname("."), '../../backend/'))

# %%
from modules.llm import *
from modules.results_gen import get_result_from_query
from tqdm import tqdm

def process_embedding_model_name_hf(name : str) -> str:
    """
    Description: This function processes the name of the embedding model from Hugging Face to use as experiment name.
    
    Input: name (str) - name of the embedding model from Hugging Face.
    
    Returns: name (str) - processed name of the embedding model.
    """
    return name.replace("/", "_")

def process_llm_model_name_ollama(name : str) -> str:
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

# %% [markdown]
# ## Running the steps
# - Create an experiment directory
# - Save a config file with the models and the queries in the experiment directory
# - Download openml data for each dataset and format into a string
# - Create vectorb and embed the data
# - Get the predictions for each model for a list of queries and evaluate the performance
# - (note) At the moment, this runs for a very small subset of the entire data. To disable this behavior and run on the entire data, set ```config["test_subset_2000"] = False```

## Aggregate multiple queries, count and save the results
# - This is part of the library already but is repeated here for clarity
def aggregate_multiple_queries_and_count(
    queries, qa_dataset, config, group_cols=["id", "name"], sort_by="query", count=True
) -> pd.DataFrame:
    """
    Description: Aggregate the results of multiple queries into a single dataframe and count the number of times a dataset appears in the results

    Input:
        queries: List of queries
        group_cols: List of columns to group by

    Returns: Combined dataframe with the results of all queries
    """
    combined_df = pd.DataFrame()
    for query in queries:
        result_data_frame, _ = get_result_from_query(
            query=query, qa=qa_dataset, type_of_query="dataset", config=config
        )
        result_data_frame = result_data_frame[group_cols]
        # Concat with combined_df with a column to store the query
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

# %%
## Override the original script to take a list of IDs as input
# - The original script creates a subset of the dataset for testing, but here we want to give it a list of dataset IDs to test on. 
# - So, we also disable the test_subset behavior and use a modified `setup_vector_db_and_qa` function
def setup_vector_db_and_qa(
    config: dict, data_type: str, client: ClientAPI, subset_ids: list = None
):
    """
    Description: Create the vector database using Chroma db with each type of data in its own collection. Doing so allows us to have a single database with multiple collections, reducing the number of databases we need to manage.
    This also downloads the embedding model if it does not exist. The QA chain is then initialized with the vector store and the configuration.

    Input: config (dict), data_type (str), client (chromadb.PersistentClient)

    Returns: qa (langchain.chains.retrieval_qa.base.RetrievalQA)
    """

    config["type_of_data"] = data_type

    # Download the data if it does not exist
    openml_data_object, data_id, all_metadata, handler = get_all_metadata_from_openml(
        config=config
    )
    # Create the combined metadata dataframe
    metadata_df, all_metadata = create_metadata_dataframe(
        handler, openml_data_object, data_id, all_metadata, config=config
    )

    # subset the metadata if subset_ids is not None
    subset_ids = [int(x) for x in subset_ids]
    if subset_ids is not None:
        metadata_df = metadata_df[metadata_df["did"].isin(subset_ids)]

    # Create the vector store
    vectordb = load_document_and_create_vector_store(
        metadata_df, config=config, chroma_client=client
    )
    # Initialize the LLM chain and setup Retrieval QA
    qa = initialize_llm_chain(vectordb=vectordb, config=config)
    return qa, all_metadata

def run_experiments(config, new_path, queries,list_of_embedding_models,list_of_llm_models, subset_ids, use_cached_experiment = False, enable_llm_results = False):
    for embedding_model in tqdm(list_of_embedding_models, desc="Embedding Models", total=len(list_of_embedding_models)):
        for llm_model in tqdm(list_of_llm_models, desc="LLM Models", total=len(list_of_llm_models)):
            # update the config with the new embedding and llm models
            config["embedding_model"] = embedding_model
            config["llm_model"] = llm_model

            # create a new experiment directory using a combination of the embedding model and llm model names
            experiment_name = f"{process_embedding_model_name_hf(embedding_model)}_{process_llm_model_name_ollama(llm_model)}"
            experiment_path = new_path/Path(f"../data/experiments/{experiment_name}")
            
            if use_cached_experiment and os.path.exists(experiment_path):
                print(f"Experiment {experiment_name} already exists. Skipping... To disable this behavior, set use_cached_experiment = False")
                continue
            else:
                # create the experiment directory if it does not exist
                os.makedirs(experiment_path, exist_ok=True)
            
                # update the config with the new experiment directories
                config["data_dir"] = str(experiment_path)
                config["persist_dir"] = str(experiment_path / "chroma_db")

                # save training details and config in a dataframe
                config_df = pd.DataFrame.from_dict(config, orient='index').reset_index()
                config_df.columns = ['Hyperparameter', 'Value']
                config_df.to_csv(experiment_path / "config.csv", index=False)

                # load the persistent database using ChromaDB
                client = chromadb.PersistentClient(path=config["persist_dir"])

                # Run "training"
                qa_dataset, _ = setup_vector_db_and_qa(
                    config=config, data_type=config["type_of_data"], client=client, subset_ids= subset_ids
                )
                combined_df = aggregate_multiple_queries_and_count(queries,qa_dataset=qa_dataset, config=config, group_cols = ["id", "name"], sort_by="query", count = False)
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
        exp["y_pred"] = exp["id"].astype(str)

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