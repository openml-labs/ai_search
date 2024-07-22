import json
import os

import requests
import streamlit as st
from streamlit import session_state as ss
import pandas as pd


def feedback_cb():
    """
    Description: Callback function to save feedback to a file
    """
    file_path = "../data/feedback.json"

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


def display_results(initial_response):
    """
    Description: Display the results in a DataFrame
    """
    st.write("Results:")
    st.dataframe(initial_response)


class LLMResponseParser:
    """
    Description: Parse the response from the LLM service and update the columns based on the response.
    """

    def __init__(self, llm_response):
        self.llm_response = llm_response
        self.subset_cols = ["did", "name"]
        self.size_sort = None
        self.classification_type = None
        self.uploader_name = None

    def process_size_attribute(self, attr_size: str):
        size, sort = attr_size.split(",") if "," in attr_size else (attr_size, None)
        if size == "yes":
            self.subset_cols.append("NumberOfInstances")
        if sort:
            self.size_sort = sort

    def missing_values_attribute(self, attr_missing: str):
        if attr_missing == "yes":
            self.subset_cols.append("NumberOfMissingValues")

    def classification_type_attribute(self, attr_classification: str):
        if attr_classification != "none":
            self.subset_cols.append("NumberOfClasses")
            self.classification_type = attr_classification

    def uploader_attribute(self, attr_uploader: str):
        if attr_uploader != "none":
            self.subset_cols.append("uploader")
            self.uploader_name = attr_uploader.split("=")[1].strip()

    def get_attributes_from_response(self):
        attribute_processors = {
            "size_of_dataset": self.process_size_attribute,
            "missing_values": self.missing_values_attribute,
            "classification_type": self.classification_type_attribute,
            "uploader": self.uploader_attribute,
        }

        for attribute, value in self.llm_response.items():
            if attribute in attribute_processors:
                attribute_processors[attribute](value)

    def update_subset_cols(self, metadata: pd.DataFrame):
        """
        Description: Filter the metadata based on the updated subset columns and extra conditions
        """
        if self.classification_type is not None:
            if "multi" in self.classification_type:
                metadata = metadata[metadata["NumberOfClasses"] > 2]
            elif "binary" in self.classification_type:
                metadata = metadata[metadata["NumberOfClasses"] == 2]
        if self.uploader_name is not None:
            try:
                uploader = int(self.uploader_name)
                metadata = metadata[metadata["uploader"] == uploader]
            except:
                pass

        return metadata[self.subset_cols]


class ResponseParser:
    """
    Description : This classe is used to decide the order of operations and run the response parsing.
    It loads the paths, fetches the Query parsing LLM response, the rag response, loads the metadatas and then based on the config, decides the order in which to apply each of them.
    """

    def __init__(self, query_type: str, apply_llm_before_rag: bool = False):
        self.query_type = query_type
        self.paths = self.load_paths()
        self.rag_response = None
        self.llm_response = None
        self.apply_llm_before_rag = apply_llm_before_rag

    def load_paths(self):
        """
        Description: Load paths from paths.json
        """
        with open("paths.json", "r") as file:
            return json.load(file)

    def fetch_llm_response(self, query: str):
        """
        Description: Fetch the response from the query parsing LLM service as a json
        """
        llm_response_path = self.paths["llm_response"]
        try:
            self.llm_response = requests.get(
                f"{llm_response_path['docker']}{query}"
            ).json()
        except:
            self.llm_response = requests.get(
                f"{llm_response_path['local']}{query}"
            ).json()
        return self.llm_response

    def fetch_structured_query(self, query_type: str, query: str):
        """
        Description: Fetch the response for a structured query from the LLM service as a JSON
        """
        structured_response_path = self.paths["structured_query"]
        try:
            self.structured_query_response = requests.get(
                f"{structured_response_path['docker']}{query}",
                json={"query": query},
            ).json()
        except:
            self.structured_query_response = requests.get(
                f"{structured_response_path['local']}{query}",
                json={"query": query},
            ).json()
        print(self.structured_query_response)
        return self.structured_query_response

    def fetch_rag_response(self, query_type: str, query: str):
        """
        Description: Fetch the response from RAG pipeline

        """
        rag_response_path = self.paths["rag_response"]
        try:
            self.rag_response = requests.get(
                f"{rag_response_path['docker']}{query_type.lower()}/{query}",
                json={"query": query, "type": query_type.lower()},
            ).json()
        except:
            self.rag_response = requests.get(
                f"{rag_response_path['local']}{query_type.lower()}/{query}",
                json={"query": query, "type": query_type.lower()},
            ).json()
        return self.rag_response

    def parse_and_update_response(self, metadata: pd.DataFrame):
        """
         Description: Parse the response from the RAG and LLM services and update the metadata based on the response.
         Decide which order to apply them
         -  self.apply_llm_before_rag == False
             - Metadata is filtered based on the rag response first and then by the Query parsing LLM
        -  self.apply_llm_before_rag == False
             - Metadata is filtered based by the Query parsing LLM first and the rag response second
        """
        if self.rag_response is not None and self.llm_response is not None:
            if not self.apply_llm_before_rag:
                filtered_metadata = metadata[
                    metadata["did"].isin(self.rag_response["initial_response"])
                ]
                llm_parser = LLMResponseParser(self.llm_response)

                if self.query_type.lower() == "dataset":
                    llm_parser.get_attributes_from_response()
                    return llm_parser.update_subset_cols(filtered_metadata)
            elif self.apply_llm_before_rag:
                llm_parser = LLMResponseParser(self.llm_response)
                llm_parser.get_attributes_from_response()
                filtered_metadata = llm_parser.update_subset_cols(metadata)

                return filtered_metadata[
                    filtered_metadata["did"].isin(self.rag_response["initial_response"])
                ]

            elif self.apply_llm_before_rag is None:
                # if no llm response is required, return the initial response
                return metadata
        elif (
            self.rag_response is not None and self.structured_query_response is not None
        ):
            return metadata[["did", "name"]]
        else:
            return metadata
