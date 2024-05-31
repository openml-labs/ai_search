# RAG Pipeline for OpenML

- This repository contains the code for the RAG pipeline for OpenML. As a start, we are focusing on using a RAG pipeline to help users find a dataset on OpenML using Natural Language queries.
- This will be further extended to include flows, runs etc.

## Setup
- Clone the repository
- Create a virtual environment and activate it
<!-- - Install the requirements using `pip install -r requirements.txt` -->
- For now, install `pip install sentence_transformers langchain langchain_community langchain_core pqdm tqdm openml chromadb oslo.concurrency fastapi`
  - A proper requirements file will be provided shortly
- Install oslo.concurrency using `pip install oslo.concurrency`
- Hugging face
  - Make an account on HF if you do not have one
  - Log in
  - Navigate to this [page](https://huggingface.co/settings/tokens) and copy the access token
  - Run `export HUGGINGFACEHUB_API_TOKEN=your_token_here` to set the token (in the shell)
      - For some weird reason, this is not read if you are using WSL2 and the VSCode connection. (Using a normal jupyter notebook/terminal works fine) 
- Run training.py (for the first time/to update the model). This takes care of basically everything. (Refer to the training section for more details)
- Run `uvicorn main:app` to start the FastAPI server. 
- Enjoy :)

### Training
- Run the training.py using `python training.py`

### Inference
- Run the inference.py using `uvicorn main:app`

## Config
- The main config file is `config.json` 
- Possible options are as follows:
  - rqa_prompt_template: The template for the RAG pipeline search prompt. This is used by the model to query the database. 
  - num_return_documents: Number of documents to return for a query. Too high a number can lead to Out of Memory errors. (Defaults to 50)
  - embedding_model: The model to use for generating embeddings. This is used to generate embeddings for the documents as a means of comparison using the LLM's embeddings. (Defaults to BAAI/bge-base-en-v1.5)
  - persist_dir: The directory to store the cached data. Defaults to ./chroma_db/ and stores the embeddings for the documents with a unique hash. (Defaults to ./chroma_db/)
  - data_download_n_jobs: Number of jobs to run in parallel for downloading data. (Defaults to 20)
  - training: Whether to train the model or not. (Defaults to False) this is automatically set to True when when running the training.py script.
  - search_type : The type of vector comparison to use. (Defaults to "similarity")

## Files and Directories
- The modules for the RAG pipeline are present in modules/
  - utils.py : Contains utility functions for the pipeline and data
  - llm.py : Contains all the functions related to the LLM
  - This will be refactored soon
- The training script is present in training.py . Runnning this script will take care of everything.
- The FastAPI server is present in inference.py. Run this to start the server.
