# This file pertains to all the utility functions required for creating and managing the vector database

from collections import OrderedDict

import pandas as pd
from flashrank import Ranker, RerankRequest
from tqdm import tqdm

from langchain_community.document_transformers import LongContextReorder
# --- PROCESSING RESULTS ---

def long_context_reorder(results):
    """
    Description: Lost in the middle reorder: the less relevant documents will be at the
    middle of the list and more relevant elements at beginning / end.
    See: https://arxiv.org/abs//2307.03172
    
    Input: results (list)
    
    Returns: reorder results (list)
    """
    print("[INFO] Reordering results...")
    reordering = LongContextReorder()
    results = reordering.transform_documents(results)
    print("[INFO] Reordering complete.")
    return results


def fetch_results(query, qa, config, type_of_query):
    """
    Description: Fetch results for the query using the QA chain.

    Input: query (str), qa (langchain.chains.retrieval_qa.base.RetrievalQA), type_of_query (str), config (dict)

    Returns: results["source_documents"] (list)
    """
    results = qa.invoke(
        input=query,
        config={"temperature": config["temperature"], "top-p": config["top_p"]},
    )
    if config["long_context_reorder"] == True:
        results = long_context_reorder(results)
    id_column = {"dataset": "did", "flow": "id", "data": "did"}
    id_column = id_column[type_of_query]

    if config["reranking"] == True:
        try:
            print("[INFO] Reranking results...")
            ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/")
            rerankrequest = RerankRequest(
                query=query,
                passages=[
                    {"id": result.metadata[id_column], "text": result.page_content}
                    for result in results
                ],
            )
            ranking = ranker.rerank(rerankrequest)
            ids = [result["id"] for result in ranking]
            ranked_results = [
                result for result in results if result.metadata[id_column] in ids
            ]
            print("[INFO] Reranking complete.")
            return ranked_results
        except Exception as e:
            print(f"[ERROR] Reranking failed: {e}")
            return results

    else:
        return results


def process_documents(source_documents, key_name):
    """
    Description: Process the source documents and create a dictionary with the key_name as the key and the name and page content as the values.

    Input: source_documents (list), key_name (str)

    Returns: dict_results (dict)
    """
    dict_results = OrderedDict()
    for result in source_documents:
        dict_results[result.metadata[key_name]] = {
            "name": result.metadata["name"],
            "page_content": result.page_content,
        }
    ids = [result.metadata[key_name] for result in source_documents]
    return dict_results, ids


def make_clickable(val):
    """
    Description: Make the URL clickable in the dataframe.
    """
    return '<a href="{}">{}</a>'.format(val, val)


def create_output_dataframe(dict_results, type_of_data, ids_order):
    """
    Description: Create an output dataframe with the results. The URLs are API calls to the OpenML API for the specific type of data.

    Input: dict_results (dict), type_of_data (str)

    Returns: A dataframe with the results and duplicate names removed.
    """
    output_df = pd.DataFrame(dict_results).T.reset_index()
    # order the rows based on the order of the ids
    output_df["index"] = output_df["index"].astype(int)
    output_df = output_df.set_index("index").loc[ids_order].reset_index()
    # output_df["urls"] = output_df["index"].apply(
    #     lambda x: f"https://www.openml.org/api/v1/json/{type_of_data}/{x}"
    # )
    # https://www.openml.org/search?type=data&sort=runs&status=any&id=31
    output_df["urls"] = output_df["index"].apply(
        lambda x: f"https://www.openml.org/search?type={type_of_data}&id={x}"
    )
    output_df["urls"] = output_df["urls"].apply(make_clickable)
    # data = openml.datasets.get_dataset(
    # get rows with unique names
    if type_of_data == "data":
        output_df["command"] = output_df["index"].apply(
            lambda x: f"dataset = openml.datasets.get_dataset({x})"
        )
    elif type_of_data == "flow":
        output_df["command"] = output_df["index"].apply(
            lambda x: f"flow = openml.flows.get_flow({x})"
        )
    output_df = output_df.drop_duplicates(subset=["name"])
    # order the columns
    output_df = output_df[["index", "name", "command", "urls", "page_content"]].rename(
        columns={"index": "id", "urls": "OpenML URL", "page_content": "Description"}
    )
    return output_df


def check_query(query):
    """
    Description: Performs checks on the query
    - Replaces %20 with space character (browsers do this automatically when spaces are in the URL)
    - Removes leading and trailing spaces
    - Limits the query to 150 characters

    Input: query (str)

    Returns: None
    """
    if query == "":
        raise ValueError("Query cannot be empty.")
    query = query.replace(
        "%20", " "
    )  # replace %20 with space character (browsers do this automatically when spaces are in the URL)
    query = query.replace(
        "dataset", ""
    )
    query = query.replace(
        "flow", ""
    )
    query = query.strip()
    query = query[:200]
    return query


def get_result_from_query(query, qa, type_of_query, config) -> pd.DataFrame:
    """
    Description: Get the result from the query using the QA chain and return the results in a dataframe that is then sent to the frontend.

    Input: query (str), qa (langchain.chains.retrieval_qa.base.RetrievalQA), type_of_query (str)

    Returns: output_df (pd.DataFrame)
    """
    if type_of_query == "dataset":
        # Fixing the key_name for dataset because of the way the OpenML API returns the data
        type_of_query = "data"
        key_name = "did"
    elif type_of_query == "flow":
        key_name = "id"
    else:
        raise ValueError(f"Unsupported type_of_data: {type_of_query}")

    # Process the query
    query = check_query(query)
    if query == "":
        return ""
    source_documents = fetch_results(
        query, qa, config=config, type_of_query=type_of_query
    )
    dict_results, ids_order = process_documents(source_documents, key_name)
    output_df = create_output_dataframe(dict_results, type_of_query, ids_order)

    return output_df


def aggregate_multiple_queries_and_count(
    queries, qa_dataset, config, group_cols=["id", "name"], sort_by="query"
):
    """
    Description: Aggregate the results of multiple queries into a single dataframe and count the number of times a dataset appears in the results

    Input:
        queries: List of queries
        group_cols: List of columns to group by

    Returns: Combined dataframe with the results of all queries
    """
    combined_df = pd.DataFrame()
    for query in tqdm(queries, total=len(queries)):
        result_data_frame = get_result_from_query(
            query=query, qa=qa_dataset, type_of_query="dataset", config=config
        )
        result_data_frame = result_data_frame[group_cols]
        # Concat with combined_df with a column to store the query
        result_data_frame["query"] = query
        combined_df = pd.concat([combined_df, result_data_frame])
    combined_df = (
        combined_df.groupby(group_cols)
        .count()
        .reset_index()
        .sort_values(by=sort_by, ascending=False)
    )
    return combined_df
