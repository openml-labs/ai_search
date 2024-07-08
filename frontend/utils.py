import json
import os

from streamlit import session_state as ss


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
    # Define the function to parse the llm_response
    size, missing, classification = response["answers"]
    size, sort = size.split(",") if "," in size else (size, None)
    return size, missing, classification, sort


def update_subset_cols(size, missing, classification):
    # Define the function to update the subset columns
    cols = ["did", "name"]
    if size == "yes":
        cols.append("NumberOfInstances")
    if missing == "yes":
        cols.append("NumberOfMissingValues")
    if classification != "none":
        cols.append("NumberOfClasses")
    return cols


def filter_initial_response(response, classification):
    # Define the function to filter the initial response based on classification
    if classification != "none":
        if "multi" in classification:
            response = response[response["NumberOfClasses"] > 2]
        elif "binary" in classification:
            response = response[response["NumberOfClasses"] == 2]
    return response
