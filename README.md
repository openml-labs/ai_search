# RAG Pipeline for OpenML

- This repository contains the code for the RAG pipeline for OpenML. As a start, we are focusing on using a RAG pipeline to help users find a dataset on OpenML using Natural Language queries.
- This will be further extended to include flows, runs etc.

## Setup
- Clone the repository
- Create a virtual environment and activate it
- Install the requirements using `pip install -r requirements.txt`
- Install oslo.concurrency using `pip install oslo.concurrency`

### For Datasets
- Run the main.py using `python main.py --query "query"`
- Check/change the config in main.py for more options
- Example query: `python main.py --query "Find a dataset about diabetes"`
- What happens automatically:
  - Uses the openml API to download metadata and qualities of all datasets and caches them (locally by default) and to a pickle file. [This takes a while!!]
  - Creates a vector store with chunked data to a persistent dataset store (./chroma_db). Entries are added with a unique hash generated based on the text content to prevent duplicates. [This takes a while!!]
  - Relevant modules are downloaded from Huggingface automatically and cached.
  - A RAG pipeline is created with the vector store and the relevant modules. This pipeline is used to generate answers for the query.

## Config
- The config is present in main.py for now. This will be moved to a separate file/format later.
- Possible options are as follows:
  - rqa_prompt_template: The template for the RAG pipeline search prompt. This is used by the model to query the database. 
  - num_return_documents: Number of documents to return for a query. Too high a number can lead to Out of Memory errors. (Defaults to 50)
  - embedding_model: The model to use for generating embeddings. This is used to generate embeddings for the documents as a means of comparison using the LLM's embeddings. (Defaults to BAAI/bge-base-en-v1.5)
  - llm_model: The main workhorse of the pipeline. (Defaults to HuggingFaceH4/zephyr-7b-beta)
  - persist_dir: The directory to store the cached data. Defaults to ./chroma_db/ and stores the embeddings for the documents with a unique hash. (Defaults to ./chroma_db/)
  - recreate_chroma: Whether to recreate the chroma_db or not. Useful for changes in data. (Duplicate entries are not added by default) Defaults to False.
  - recreate_data_cache: Whether to redownload dataset metadata and qualities or not. (Defaults to False)
  - data_download_n_jobs: Number of jobs to run in parallel for downloading data. (Defaults to 20)
  - device: Device to create the embeddings on. (Defaults to 'cuda' if available else 'cpu'). Usually using a GPU/Apple Silicon is recommended.

## Files and Directories
- The modules for the RAG pipeline are present in modules/
- The main file for the pipeline is main.py . This file is used to run the pipeline. (This will later be converted to a FastAPI app)
    - For now, the config is present in main.py
- main.ipynb is a notebook that can be used to run the pipeline interactively. This is useful for debugging and testing.
