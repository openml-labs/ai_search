
from training_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

def exp_0(process_query_elastic_search, eval_path, query_key_dict):
    """
    EXPERIMENT 0
    Get results from elastic search
    """
    # cols = ,did,name,query,llm_model,embedding_model,llm_before_rag
    # for every query, get the results from elastic search
    if not os.path.exists(eval_path / "elasticsearch" / "elasticsearch"):
        os.makedirs(eval_path / "elasticsearch" / "elasticsearch")
    output_file_path = eval_path / "elasticsearch" / "elasticsearch" / "results.csv"
    # check if the file exists and skip
    if os.path.exists(output_file_path) == False:
        with open(output_file_path, "w") as f:
            f.write("did,name,query,llm_model,embedding_model,llm_before_rag\n")

            # Use ThreadPoolExecutor to parallelize requests
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Start a future for each query
                futures = {executor.submit(process_query_elastic_search, query, dataset_id): query for query, dataset_id
                           in
                           query_key_dict.items()}

                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    # Save the results to a CSV file
                    for id, query in result:
                        f.write(f"{id},None,{query},es,es,None\n")

def exp_1(eval_path, config, list_of_embedding_models, list_of_llm_models, subset_ids, query_key_dict):
    """
    EXPERIMENT 1
    Main evaluation loop that is used to run the base experiments using different models and embeddings.
    Takes into account the following:
    original data ingestion pipeline : combine a string of all metadata fields and the dataset description and embeds them with no pre-processing
    list_of_embedding_models = [
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "Snowflake/snowflake-arctic-embed-l",
    ]
    list_of_llm_models = ["llama3", "phi3"]
    types_of_llm_apply : llm applied as filter before the RAG pipeline, llm applied as reranker after the RAG pipeline, llm not used at all
    """

    expRunner = ExperimentRunner(
        config=config,
        eval_path=eval_path,
        queries=query_key_dict.keys(),
        list_of_embedding_models=list_of_embedding_models,
        list_of_llm_models=list_of_llm_models,
        subset_ids=subset_ids,
        use_cached_experiment=True,
    )
    expRunner.run_experiments()

def exp_2(eval_path, config, subset_ids, query_key_dict):
    """
    EXPERIMENT 2
    Evaluating temperature = 1 (default was 0.95)
    Takes into account the following:
    original data ingestion pipeline : combine a string of all metadata fields and the dataset description and embeds them with no pre-processing
    list_of_embedding_models = [
        "BAAI/bge-large-en-v1.5",
    ]
    list_of_llm_models = ["llama3"]
    types_of_llm_apply : llm applied as filter before the RAG pipeline, llm applied as reranker after the RAG pipeline, llm not used at all
    """

    list_of_embedding_models = [
        "BAAI/bge-large-en-v1.5",
    ]
    list_of_llm_models = ["llama3"]
    config["temperature"] = 1

    expRunner = ExperimentRunner(
        config=config,
        eval_path=eval_path,
        queries=query_key_dict.keys(),
        list_of_embedding_models=list_of_embedding_models,
        list_of_llm_models=list_of_llm_models,
        subset_ids=subset_ids,
        use_cached_experiment=True,
        custom_name="temperature_1",
    )
    expRunner.run_experiments()

    # reset the temperature to the default value
    config["temperature"] = 0.95

def exp_3(eval_path, config, subset_ids, query_key_dict):
    """
    EXPERIMENT 3
    Evaluating search type [mmr, similarity_score_threshold] (default was similarity)
    Takes into account the following:
    original data ingestion pipeline : combine a string of all metadata fields and the dataset description and embeds them with no pre-processing
    list_of_embedding_models = [
        "BAAI/bge-large-en-v1.5",
    ]
    list_of_llm_models = ["llama3"]
    types_of_llm_apply : llm applied as reranker after the RAG pipeline
    """


    list_of_embedding_models = [
        "BAAI/bge-large-en-v1.5",
    ]
    list_of_llm_models = ["llama3"]
    types_of_llm_apply = [False]
    types_of_search = ["mmr", "similarity_score_threshold"]

    for type_of_search in types_of_search:
        config["search_type"] = type_of_search
        expRunner = ExperimentRunner(
            config=config,
            eval_path=eval_path,
            queries=query_key_dict.keys(),
            list_of_embedding_models=list_of_embedding_models,
            list_of_llm_models=list_of_llm_models,
            subset_ids=subset_ids,
            use_cached_experiment=True,
            custom_name=f"{type_of_search}_search",
            types_of_llm_apply=types_of_llm_apply,
        )
        expRunner.run_experiments()

    # reset the search type to the default value
    config["search_type"] = "similarity"

def exp_4(eval_path, config, subset_ids, query_key_dict):
    """
    EXPERIMENT 4
    Evaluating chunk size. The default is 1000, trying out 512,128
    Takes into account the following:
    original data ingestion pipeline : combine a string of all metadata fields and the dataset description and embeds them with no pre-processing
    list_of_embedding_models = [
        "BAAI/bge-large-en-v1.5",
    ]
    list_of_llm_models = ["llama3"]
    types_of_llm_apply : llm applied as reranker after the RAG pipeline
    """


    list_of_embedding_models = [
        "BAAI/bge-large-en-v1.5",
    ]
    list_of_llm_models = ["llama3"]
    types_of_llm_apply = [False]
    types_of_chunk = [512, 128]
    for type_of_chunk in types_of_chunk:
        config["chunk_size"] = type_of_chunk
        expRunner = ExperimentRunner(
            config=config,
            eval_path=eval_path,
            queries=query_key_dict.keys(),
            list_of_embedding_models=list_of_embedding_models,
            list_of_llm_models=list_of_llm_models,
            subset_ids=subset_ids,
            use_cached_experiment=True,
            custom_name=f"{type_of_chunk}_chunk",
            types_of_llm_apply=types_of_llm_apply,
        )
        expRunner.run_experiments()

    # reset the search type to the default value
    config["chunk_size"] = 1000