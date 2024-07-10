import json
import os

from streamlit import session_state as ss
import requests
import streamlit as st

# load paths from paths.json
with open("paths.json", "r") as file:
    paths = json.load(file)

def feedback_cb():
    """
    Description: Callback function to save feedback to a file

    Input: None

    Returns: None
    """
    file_path = "feedback.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new feedback
    data.append({"ss": ss.fb_k, "query": ss.query})

    # Write updated content back to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def parse_llm_response(response):
    """
    Description: Parse the answers from the LLM response
    
    Input: response (dict)
    
    Returns: size (str), missing (str), classification (str), sort (str)
    """
    size, missing, classification, uploader = response["answers"]
    # Split size and sort if there is a comma
    size, sort = size.split(",") if "," in size else (size, None)

    # split uploader by = to get the name
    if uploader != "none":
        uploader = uploader.split("=")[1].strip()
    return size, missing, classification, sort, uploader


def update_subset_cols(size, missing, classification, uploader):
    """
    Description: Update the subset columns based on LLM's response
    
    Input: size (str), missing (str), classification (str)
    
    Returns: cols (list)
    """
    cols = ["did", "name"]
    if size == "yes":
        cols.append("NumberOfInstances")
    if missing == "yes":
        cols.append("NumberOfMissingValues")
    if classification != "none":
        cols.append("NumberOfClasses")
    if uploader != "none":
        cols.append("uploader")
    return cols


def filter_initial_response(response, classification, uploader):
    """
    Description: Filter the initial response based on the classification
    
    Input: response (DataFrame), classification (str)
    
    Returns: response (DataFrame)
    """
    if classification != "none":
        if "multi" in classification:
            response = response[response["NumberOfClasses"] > 2]
        elif "binary" in classification:
            response = response[response["NumberOfClasses"] == 2]
    if uploader != "none":
        try:
            uploader = int(uploader)
            response = response[response["uploader"] == uploader]
        except:
            pass
    return response


def fetch_response(query_type, query):
    """
    Description: Fetch the response from the FastAPI service
    
    Input: query_type (str), query (str)
    
    Returns: response (dict)
    """
    rag_response_path = paths["rag_response"]
    try:
        response = requests.get(
            f"{rag_response_path['docker']}{query_type.lower()}/{query}",
            json={"query": query, "type": query_type.lower()},
        ).json()
    except:
        response = requests.get(
            f"{rag_response_path['local']}{query_type.lower()}/{query}",
            json={"query": query, "type": query_type.lower()},
        ).json()
    return response

def fetch_llm_response(query):
    """
    Description: Fetch the response from the LLM service
    
    Input: query (str)
    
    Returns: llm_response (dict)
    """
    llm_response_path = paths["llm_response"]
    try:
        llm_response = requests.get(f"{llm_response_path['docker']}{query}").json()
    except:
        llm_response = requests.get(f"{llm_response_path['local']}{query}").json()
    return llm_response

def parse_and_update_response(query_type, response, llm_response, data_metadata, flow_metadata):
    """
    Description: Parse and update the response based on the query type
    
    Input: query_type (str), response (dict), llm_response (dict), data_metadata (DataFrame), flow_metadata (DataFrame)
    
    Returns: initial_response (DataFrame)
    """
    if query_type == "Dataset":
        initial_response = data_metadata[data_metadata["did"].isin(response["initial_response"])]
        subset_cols = ["did", "name"]
        try:
            dataset_size, dataset_missing, dataset_classification, dataset_sort, uploader = parse_llm_response(llm_response)
            subset_cols = update_subset_cols(dataset_size, dataset_missing, dataset_classification, uploader)
            initial_response = filter_initial_response(initial_response, dataset_classification, uploader)
        except Exception as e:
            st.error(f"Error processing LLM response: {e}")
        initial_response = initial_response[subset_cols]
    else:
        initial_response = flow_metadata[flow_metadata["id"].isin(response["initial_response"])]
    return initial_response

def display_results(initial_response):
    """
    Description: Display the results in a DataFrame
    
    Input: initial_response (DataFrame)
    
    Returns: None
    """
    st.write("Results:")
    st.dataframe(initial_response)
