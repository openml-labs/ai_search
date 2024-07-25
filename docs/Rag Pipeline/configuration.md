# Configuration

- The main config file is `config.json`. Since this is loaded in every training/evaluation script, you can use this to modify the behavior inline. 

## Possible options
  - **rqa_prompt_template**: The template for the RAG pipeline search prompt. This is used by the model to query the database. 
  - **llm_prompt_template**: The template for the summary generator LLM prompt.
  - **num_return_documents**: Number of documents to return for a query. Too high a number can lead to Out of Memory errors. (Defaults to 50)
  - **embedding_model**: THIS IS FROM HUGGINGFACE. The model to use for generating embeddings. This is used to generate embeddings for the documents as a means of comparison using the LLM's embeddings. (Defaults to BAAI/bge-large-en-v1.5)
    - Other possible tested models
        - BAAI/bge-base-en-v1.5
        - BAAI/bge-large-en-v1.5
  - **llm_model**: THIS IS FROM OLLAMA. The model used for generating the result summary. (Defaults to qwen2:1.5b)
  - **data_dir**: The directory to store the intermediate data like tables/databases etc. (Defaults to ./data/)
  - **persist_dir**: The directory to store the cached data. Defaults to ./data/chroma_db/ and stores the embeddings for the documents with a unique hash. (Defaults to ./data/chroma_db/)
  - **testing_flag**: Enables testing mode by using subsets of the data for quick debugging. This is used to test the pipeline and is not recommended for normal use. (Defaults to False)
  - **test_subset**: Uses a tiny subset of the data for testing.
  - **data_download_n_jobs**: Number of jobs to run in parallel for downloading data. (Defaults to 20)
  - **training**: Whether to train the model or not. (Defaults to False) this is automatically set to True when when running the training.py script. Do **NOT** set this to True manually.
  - **search_type** : The type of vector comparison to use. (Defaults to "similarity")
  - **reraanking**: Whether to rerank the results using the FlashRank algorithm. (Defaults to False)
  - **long_context_reordering**: Whether to reorder the results using the Long Context Reordering algorithm. (Defaults to False)
  - **chunk_size**: Size of the chunks for the RAG document chunking