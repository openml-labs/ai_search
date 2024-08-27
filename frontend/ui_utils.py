import json
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit import session_state as ss

sys.path.append("../")
from structured_query.chroma_store_utilis import *


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
        self.database_filtered = None
        self.structured_query_response = None

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
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            # Print the error for debugging purposes
            print(f"Error occurred: {e}")
            # Set structured_query_response to None on error
            self.structured_query_response = None
        try:
            self.structured_query_response = requests.get(
                f"{structured_response_path['local']}{query}",
                json={"query": query},
            ).json()
        except Exception as e:
            # Print the error for debugging purposes
            print(f"Error occurred while fetching from local endpoint: {e}")
            # Set structured_query_response to None if the local request also fails
            self.structured_query_response = None

        return self.structured_query_response

    def database_filter(self, filter_condition, collec):
        """
        Apply database filter on the rag_response
        """
        ids = list(map(str, self.rag_response["initial_response"]))
        self.database_filtered = collec.get(ids=ids, where=filter_condition)["ids"]
        self.database_filtered = list(map(int, self.database_filtered))
        # print(self.database_filtered)
        return self.database_filtered

    def fetch_rag_response(self, query_type, query):
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
        ordered_set = self._order_results()
        self.rag_response["initial_response"] = ordered_set

        return self.rag_response

    def _order_results(self):
        doc_set = set()
        ordered_set = []
        for docid in self.rag_response["initial_response"]:
            if docid not in doc_set:
                ordered_set.append(docid)
            doc_set.add(docid)
        return ordered_set

    def parse_and_update_response(self, metadata: pd.DataFrame):
        """
         Description: Parse the response from the RAG and LLM services and update the metadata based on the response.
         Decide which order to apply them
         -  self.apply_llm_before_rag == False
             - Metadata is filtered based on the rag response first and then by the Query parsing LLM
        -  self.apply_llm_before_rag == False
             - Metadata is filtered based by the Query parsing LLM first and the rag response second
        - in case structured_query == true, take results are applying data filters.
        """
        if self.apply_llm_before_rag is None or self.llm_response is None:
            print("No LLM filter.")
            # print(self.rag_response, flush=True)
            filtered_metadata = self._no_filter(metadata)

            # print(filtered_metadata)
            # if no llm response is required, return the initial response
            return filtered_metadata

        elif (
            self.rag_response is not None and self.llm_response is not None
        ) and not config["structured_query"]:
            if not self.apply_llm_before_rag:
                filtered_metadata, llm_parser = self._rag_before_llm(metadata)

                if self.query_type.lower() == "dataset":
                    llm_parser.get_attributes_from_response()
                    return llm_parser.update_subset_cols(filtered_metadata)

            elif self.apply_llm_before_rag:
                filtered_metadata = self._filter_before_rag(metadata)
                return filtered_metadata

        elif (
            self.rag_response is not None and self.structured_query_response is not None
        ):
            col_name = [
                "status",
                "NumberOfClasses",
                "NumberOfFeatures",
                "NumberOfInstances",
            ]
            # print(self.structured_query_response)  # Only for debugging. Comment later.
            if self.structured_query_response[0] is not None and isinstance(
                self.structured_query_response[1], dict
            ):
                # Safely attempt to access the "filter" key in the first element

                self._structured_query_on_success(metadata)

            else:
                filtered_metadata = self._structured_query_on_fail(metadata)
                # print("Showing only rag response")
            return filtered_metadata[["did", "name", *col_name]]

    def _structured_query_on_fail(self, metadata):
        filtered_metadata = metadata[
            metadata["did"].isin(self.rag_response["initial_response"])
        ]
        filtered_metadata["did"] = pd.Categorical(
            filtered_metadata["did"],
            categories=self.rag_response["initial_response"],
            ordered=True,
        )
        filtered_metadata = filtered_metadata.sort_values("did").reset_index(drop=True)

        return filtered_metadata

    def _structured_query_on_success(self, metadata):
        if (
            self.structured_query_response[0].get("filter", None)
            and self.database_filtered
        ):
            filtered_metadata = metadata[metadata["did"].isin(self.database_filtered)]
            # print("Showing database filtered data")
        else:
            filtered_metadata = metadata[
                metadata["did"].isin(self.rag_response["initial_response"])
            ]
            # print(
            #     "Showing only rag response as filter is empty or none of the rag data satisfies filter conditions."
            # )
        filtered_metadata["did"] = pd.Categorical(
            filtered_metadata["did"],
            categories=self.rag_response["initial_response"],
            ordered=True,
        )
        filtered_metadata = filtered_metadata.sort_values("did").reset_index(drop=True)

    def _filter_before_rag(self, metadata):
        print("LLM filter before RAG")
        llm_parser = LLMResponseParser(self.llm_response)
        llm_parser.get_attributes_from_response()
        filtered_metadata = llm_parser.update_subset_cols(metadata)
        filtered_metadata = filtered_metadata[
            metadata["did"].isin(self.rag_response["initial_response"])
        ]
        filtered_metadata["did"] = pd.Categorical(
            filtered_metadata["did"],
            categories=self.rag_response["initial_response"],
            ordered=True,
        )
        filtered_metadata = filtered_metadata.sort_values("did").reset_index(drop=True)

        return filtered_metadata

    def _rag_before_llm(self, metadata):
        print("RAG before LLM filter.")
        filtered_metadata = metadata[
            metadata["did"].isin(self.rag_response["initial_response"])
        ]
        filtered_metadata["did"] = pd.Categorical(
            filtered_metadata["did"],
            categories=self.rag_response["initial_response"],
            ordered=True,
        )
        filtered_metadata = filtered_metadata.sort_values("did").reset_index(drop=True)
        llm_parser = LLMResponseParser(self.llm_response)
        return filtered_metadata, llm_parser

    def _no_filter(self, metadata):
        filtered_metadata = metadata[
            metadata["did"].isin(self.rag_response["initial_response"])
        ]
        filtered_metadata["did"] = pd.Categorical(
            filtered_metadata["did"],
            categories=self.rag_response["initial_response"],
            ordered=True,
        )
        filtered_metadata = filtered_metadata.sort_values("did").reset_index(drop=True)

        return filtered_metadata


class UILoader:
    """
    Description : Create the chat interface
    """

    def __init__(self, config_path):
        with open(config_path, "r") as file:
            # Load config
            self.config = json.load(file)
        # Paths and display information

        # Load metadata chroma database for structured query
        self.collec = load_chroma_metadata()

        # Metadata paths
        self.data_metadata_path = (
            Path(config["data_dir"]) / "all_dataset_description.csv"
        )
        self.flow_metadata_path = Path(config["data_dir"]) / "all_flow_description.csv"

        # Read metadata
        self.data_metadata = pd.read_csv(self.data_metadata_path)
        self.flow_metadata = pd.read_csv(self.flow_metadata_path)

        # defaults
        self.query_type = "Dataset"
        self.llm_filter = False
        self.paths = self.load_paths()
        self.info = """
        <p style='text-align: center; color: white;'>Machine learning research should be easily accessible and reusable. <a href = "https://openml.org/">OpenML</a> is an open platform for sharing datasets, algorithms, and experiments - to learn how to learn better, together. </p>
        """
        self.logo = "images/favicon.ico"
        self.chatbot_display = "How do I do X using OpenML? / Find me a dataset about Y"

        if "messages" not in st.session_state:
            st.session_state.messages = []

    # container for company description and logo
    def _generate_logo_header(
        self,
    ):

        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(self.logo, width=100)
        with col2:
            st.markdown(
                self.info,
                unsafe_allow_html=True,
            )

    def generate_complete_ui(self):

        self._generate_logo_header()
        chat_container = st.container()
        # self.disclaimer_dialog()
        with chat_container:
            with st.form(key="chat_form"):
                user_input = st.text_input(
                    label="Query", placeholder=self.chatbot_display
                )
                query_type = st.selectbox(
                    "Select Query Type",
                    ["General Query", "Dataset", "Flow"],
                    help="Are you looking for a dataset or a flow or just have a general query?",
                )
                ai_filter = st.toggle(
                    "Use AI powered filtering",
                    value=True,
                    help="Uses an AI model to identify what columns might be useful to you.",
                )
                st.form_submit_button(label="Search")

            self.create_chat_interface(user_input=None)
            if user_input:
                self.create_chat_interface(
                    user_input, query_type=query_type, ai_filter=ai_filter
                )

    def create_chat_interface(self, user_input, query_type=None, ai_filter=False):
        """
        Description: Create the chat interface and display the chat history and results. Show the user input and the response from the OpenML Agent.

        """
        self.query_type = query_type
        self.ai_filter = ai_filter

        if user_input is None:
            with st.chat_message(name="ai"):
                st.write("OpenML Agent: ", "Hello! How can I help you today?")
                st.write(
                    ":warning: Note that results are powered by local LLM models and may not be accurate. Please refer to the official OpenML website for accurate information."
                )

        # Handle user input
        if user_input:
            self._handle_user_input(user_input, query_type)

    def _handle_user_input(self, user_input, query_type):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Waiting for results..."):
            results = self.process_query_chat(user_input)

        if not self.query_type == "General Query":
            st.session_state.messages.append(
                    {"role": "OpenML Agent", "content": results}
                )
        else:
            self._stream_results(results)

            # reverse messages to show the latest message at the top
        reversed_messages = self._reverse_session_history()

            # Display chat history
        self._display_chat_history(query_type, reversed_messages)
        self.create_download_button()

    def _display_chat_history(self, query_type, reversed_messages):
        for message in reversed_messages:
            if query_type == "General Query":
                pass
            if message["role"] == "user":
                with st.chat_message(name="user"):
                    self.display_results(message["content"], "user")
            else:
                with st.chat_message(name="ai"):
                    self.display_results(message["content"], "ai")

    def _reverse_session_history(self):
        reversed_messages = []
        for index in range(0, len(st.session_state.messages), 2):
            reversed_messages.insert(0, st.session_state.messages[index])
            reversed_messages.insert(1, st.session_state.messages[index + 1])
        return reversed_messages

    def _stream_results(self, results):
        with st.spinner("Fetching results..."):
            with requests.get(results, stream=True) as r:
                resp_contain = st.empty()
                streamed_response = ""
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        streamed_response += chunk.decode("utf-8")
                        resp_contain.markdown(streamed_response)
                resp_contain.empty()
            st.session_state.messages.append(
                {"role": "OpenML Agent", "content": streamed_response}
            )

    @st.experimental_fragment()
    def create_download_button(self):
        data = "\n".join(
            [str(message["content"]) for message in st.session_state.messages]
        )
        st.download_button(
            label="Download chat history",
            data=data,
            file_name="chat_history.txt",
        )

    def display_results(self, initial_response, role):
        """
        Description: Display the results in a DataFrame
        """
        try:
            st.dataframe(initial_response)
        except:
            st.write(initial_response)

    # Function to handle query processing
    def process_query_chat(self, query):
        """
        Description: Process the query and return the results based on the query type and the LLM filter.

        """
        apply_llm_before_rag = None if not self.llm_filter else False
        response_parser = ResponseParser(
            self.query_type, apply_llm_before_rag=apply_llm_before_rag
        )

        if self.query_type == "Dataset" or self.query_type == "Flow":
            if not self.ai_filter:
                response_parser.fetch_rag_response(self.query_type, query)
                return response_parser.parse_and_update_response(self.data_metadata)
            else:
                # get structured query
                self._display_structured_query_results(query, response_parser)

            results = response_parser.parse_and_update_response(self.data_metadata)
            return results

        elif self.query_type == "General Query":
            # Return documentation response path
            return self.paths["documentation_query"]["local"] + query

    def _display_structured_query_results(self, query, response_parser):
        response_parser.fetch_structured_query(self.query_type, query)
        try:
            # get rag response
            # using original query instead of extracted topics.
            response_parser.fetch_rag_response(
                self.query_type,
                response_parser.structured_query_response[0]["query"],
            )

            if response_parser.structured_query_response:
                st.write(
                    "Detected Filter(s): ",
                    json.dumps(
                        response_parser.structured_query_response[0].get("filter", None)
                    ),
                )
            else:
                st.write("Detected Filter(s): ", None)
            if response_parser.structured_query_response[1].get("filter"):
                with st.spinner("Applying LLM Detected Filter(s)..."):
                    response_parser.database_filter(
                        response_parser.structured_query_response[1]["filter"],
                        collec,
                    )
        except:
            # fallback to RAG response
            response_parser.fetch_rag_response(self.query_type, query)

    def load_paths(self):
        """
        Description: Load paths from paths.json
        """
        with open("paths.json", "r") as file:
            return json.load(file)
