import json
import os

import requests
import streamlit as st
from streamlit import session_state as ss


def feedback_cb():
    """
    Description: Callback function to save feedback to a file
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


def display_results(initial_response):
    """
    Description: Display the results in a DataFrame
    """
    st.write("Results:")
    st.dataframe(initial_response)


class LLMResponseParser:
    """
    Description: Parse the response from the LLM service and update the columns based on the response
    """

    def __init__(self, llm_response):
        self.llm_response = llm_response
        self.subset_cols = ["did", "name"]
        self.size_sort = None
        self.classification_type = None
        self.uploader_name = None

    def process_size_attribute(self, attr_size):
        size, sort = attr_size.split(",") if "," in attr_size else (attr_size, None)
        if size == "yes":
            self.subset_cols.append("NumberOfInstances")
        if sort:
            self.size_sort = sort

    def missing_values_attribute(self, attr_missing):
        if attr_missing == "yes":
            self.subset_cols.append("NumberOfMissingValues")

    def classification_type_attribute(self, attr_classification):
        if attr_classification != "none":
            self.subset_cols.append("NumberOfClasses")
            self.classification_type = attr_classification

    def uploader_attribute(self, attr_uploader):
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

    def update_subset_cols(self, metadata):
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
    def __init__(self, query_type, apply_llm_before_rag=False):
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

    def fetch_llm_response(self, query):
        """
        Description: Fetch the response from the LLM service as a json
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

    def fetch_rag_response(self, query_type, query):
        """
        Description: Fetch the response from the FastAPI service


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

    def parse_and_update_response(self, metadata):
        """
        Description: Parse the response from the RAG and LLM services and update the metadata based on the response
        """
        if self.rag_response is not None and self.llm_response is not None:
            if self.apply_llm_before_rag == False:
                filtered_metadata = metadata[
                    metadata["did"].isin(self.rag_response["initial_response"])
                ]
                llm_parser = LLMResponseParser(self.llm_response)

                if self.query_type.lower() == "dataset":
                    llm_parser.get_attributes_from_response()
                    return llm_parser.update_subset_cols(filtered_metadata)
            elif self.apply_llm_before_rag == True:
                llm_parser = LLMResponseParser(self.llm_response)
                llm_parser.get_attributes_from_response()
                filtered_metadata = llm_parser.update_subset_cols(metadata)

                return filtered_metadata[
                    filtered_metadata["did"].isin(self.rag_response["initial_response"])
                ]
            
            elif self.apply_llm_before_rag == None:
                # if no llm response is required, return the initial response 
                return metadata
        else:
            return metadata
