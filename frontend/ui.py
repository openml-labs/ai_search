import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit_feedback import streamlit_feedback
from utils import (feedback_cb, filter_initial_response, parse_llm_response,
                   update_subset_cols)

with open("../backend/config.json", "r") as file:
    config = json.load(file)

# Metadata paths
data_metadata = Path(config["data_dir"]) / "all_dataset_description.csv"
flow_metadata = Path(config["data_dir"]) / "all_flow_description.csv"

# Load metadata
data_metadata = pd.read_csv(data_metadata)
flow_metadata = pd.read_csv(flow_metadata)

# Main Streamlit App
st.title("OpenML AI Search")

query_type = st.selectbox("Select Query Type", ["Dataset", "Flow"])
query = st.text_input("Enter your query")

st.session_state["query"] = query

response = {"initial_response": None}

if st.button("Submit"):
    with st.spinner("Waiting for results..."):
        try:
            response = requests.get(
                f"http://fastapi:8000/{query_type.lower()}/{query}",
                json={"query": query, "type": query_type.lower()},
            ).json()
        except:
            response = requests.get(
                f"http://0.0.0.0:8000/{query_type.lower()}/{query}",
                json={"query": query, "type": query_type.lower()},
            ).json()

    if response["initial_response"] is not None:
        st.write("Results:")

        # response is the ids, we need to get the metdata from the ids
        if query_type == "Dataset":
            initial_response = data_metadata[
                data_metadata["did"].isin(response["initial_response"])
            ]
            # subset_cols = ["did", "name","OpenML URL","Description", "command"]
        else:
            initial_response = flow_metadata[
                flow_metadata["id"].isin(response["initial_response"])
            ]

        # def process query using results from port 8001/llmquery/{query}
        with st.spinner("Using an LLM to find the most relevent information..."):
            if query_type == "Dataset":
                try:
                    llm_response = requests.get(
                        f"http://fastapi:8081/llmquery/{query}"
                    ).json()
                except:
                    llm_response = requests.get(
                        f"http://0.0.0.0:8081/llmquery/{query}"
                    ).json()

                subset_cols = ["did", "name"]
                try:
                    (
                        dataset_size,
                        dataset_missing,
                        dataset_classification,
                        dataset_sort,
                    ) = parse_llm_response(llm_response)
                    subset_cols = update_subset_cols(
                        dataset_size, dataset_missing, dataset_classification
                    )
                    initial_response = filter_initial_response(
                        initial_response, dataset_classification
                    )
                except Exception as e:
                    st.error(f"Error processing LLM response: {e}")

                initial_response = initial_response[subset_cols]
              
                st.dataframe(initial_response)

    with st.form("fb_form"):
        streamlit_feedback(
            feedback_type="thumbs",
            align="flex-start",
            key="fb_k",
            optional_text_label="[Optional] Please provide an explanation",
        )
        st.form_submit_button("Save feedback", on_click=feedback_cb)
