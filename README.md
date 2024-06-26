# RAG Pipeline for OpenML

- This repository contains the code for the RAG pipeline for OpenML. 
- [Documentation page](https://openml-labs.github.io/ai_search/)

## Running inference
- Run the inference using `uvicorn main:app` and `streamlit run main.py` in different processes.

## Getting started
- A docker image will be provided at a later date for easier setup
- Clone the repository
- Create a virtual environment and activate it
- Install the requirements using `pip install -r requirements.txt`
- Run training.py (for the first time/to update the model). This takes care of basically everything. (Refer to the training section for more details)
- Install Ollama (https://ollama.com/) and download the models `ollama run qwen2:1.5b` and `ollama run phi3`
- Run `uvicorn backend:app` to start the FastAPI server. 
- Run `streamlit run main.py` to start the Streamlit frontend (this uses the FastAPI server so make sure it is running)
- Enjoy :)


## Features
### RAG
- Multi-threaded downloading of OpenML datasets
- Combining all information about a dataset to enable searching
- Chunking of the data to prevent OOM errors 
- Hashed entries by document content to prevent duplicate entries to the database
- RAG pipeline for OpenML datasets/flows
- Specify config["training"] = True/False to switch between training and inference mode
- Option to modify the prompt before querying the database (Example: HTML string replacements)
- Results are returned in either JSON or dataframe format
- Easily customizable pipeline
- Easily run multiple queries and aggregate the results
- LLM summarization of the results
- Streamlit frontend for easy querying with a FastAPI backend
### Enhancements
- One config file to rule them all
- GUI search interface using FastAPI (uvicorn main:app) with extra search bar for the results
- Option for Long Context Reordering of results (https://arxiv.org/abs//2307.03172)
- Option for FlashRank Reranking of results
- Caching queries in a database for faster retrieval
- Auto detect and use GPU/MPS if available for training
- Auto retry failed queries for a set number of times (2)
