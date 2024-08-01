# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: openml
#     language: python
#     name: python3
# ---

# # Tutorial on changing models
# - How would you use a different embedding and llm model?

from __future__ import annotations
from langchain_community.cache import SQLiteCache
import os
import sys
import chromadb

from backend.modules.utils import load_config_and_device
from backend.modules.rag_llm import QASetup

# ## Initial config

config = load_config_and_device("../../../backend/config.json")
config["persist_dir"] = "../../data/doc_examples/chroma_db/"
config["data_dir"] = "../../data/doc_examples/"
config["type_of_data"] = "dataset"
config["training"] = True
config["test_subset"] = True  # set this to false while training, this is for demo
# load the persistent database using ChromaDB
client = chromadb.PersistentClient(path=config["persist_dir"])
print(config)

# ## Embedding model
# - Pick a model from HF

config["embedding_model"] = "BAAI/bge-large-en-v1.5"

# ## LLM model

# - Pick a model from Ollama - https://ollama.com/library?sort=popular
# - eg : mistral
#

config["llm_model"] = "mistral"

# +
qa_dataset_handler = QASetup(
    config=config,
    data_type=config["type_of_data"],
    client=client,
)

qa_dataset, _ = qa_dataset_handler.setup_vector_db_and_qa()
# -

# # IMPORTANT
# - Do NOT forget to change the model to the best model in ollama/get_ollama.sh
