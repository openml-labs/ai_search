# %% [markdown]
# # Tutorial on using multiple models for evaluation
# - This tutorial is an example of how to test multiple models on the openml data to see which one performs the best.
# - The evaluation is still a bit basic, but it is a good starting point for future research.

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
import pandas as pd
from langchain_community.cache import SQLiteCache
from tqdm import tqdm

# change the path to the backend directory
sys.path.append(os.path.join(os.path.dirname("."), "../../backend/"))

from modules.llm import *
from modules.results_gen import aggregate_multiple_queries_and_count
# %%
from modules.utils import load_config_and_device

# %% [markdown]
# ## Setting the config

# %%
new_path = Path("../../backend/")

config = load_config_and_device(str(new_path / "config.json"), training=True)

config["type_of_data"] = "dataset"
config["training"] = True

# %% [markdown]
# ## Defining the models used
# - Embedding models are any from Huggingface hub
# - LLM models are any from Ollama library

# %%
# list_of_embedding_models = ["BAAI/bge-small-en-v1.5", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"]
list_of_embedding_models = ["BAAI/bge-large-en-v1.5"]
list_of_llm_models = ["qwen2:1.5b", "phi3", "llama3"]


# %%
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
# ## Defining the evaluation queries
# - replace this with a proper dataframe for a more comprehensive evaluation

# %%
queries = [
    "Find datasets related to COVID-19",
    "Find datasets related to COVID-19 and India",
    "COVID-19 dataset",
    "COVID-19 dataset India",
    "Mexico historical covid",
]

# %% [markdown]
# ## Downloading the models
# - PLEASE MAKE SURE YOU HAVE DOWNLOADED OLLAMA (```curl -fsSL https://ollama.com/install.sh | sh```)

# %%
# download the ollama llm models

# os.system("curl -fsSL https://ollama.com/install.sh | sh")
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

# %% [markdown]
# ### Override setup_vector_db_and_qa to use a list of IDs instead of all of them


# %%
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
    if subset_ids is not None:
        metadata_df = metadata_df[metadata_df["did"].isin(subset_ids)]

    # Create the vector store
    vectordb = load_document_and_create_vector_store(
        metadata_df, config=config, chroma_client=client
    )
    # Initialize the LLM chain and setup Retrieval QA
    qa = initialize_llm_chain(vectordb=vectordb, config=config)
    return qa, all_metadata


# %%
# use a tiny subset of the data for testing (no need for the edited subset)

# %%
use_custom_data = False


if use_custom_data:
    config["test_subset"] = False
else:
    config["test_subset"] = True

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
        experiment_name = f"{process_embedding_model_name_hf(embedding_model)}_{process_llm_model_name_ollama(llm_model)}"
        experiment_path = new_path / Path(f"../data/experiments/{experiment_name}")

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
        qa_dataset = setup_vector_db_and_qa(
            config=config, data_type=config["type_of_data"], client=client
        )

        # # Run an evaluation by aggregating multiple queries and counting the results
        # # TODO : Replace this evaluation with a more meaningful one
        combined_df = aggregate_multiple_queries_and_count(
            queries,
            qa_dataset=qa_dataset,
            config=config,
            group_cols=["id", "name"],
            sort_by="query",
            count=False,
        )

        # # TODO : ADD LLM evaluation here when the function is ready

        combined_df.to_csv(experiment_path / "results.csv")

# %%
