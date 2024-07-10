import json
from pathlib import Path

from streamlit_feedback import streamlit_feedback
import pandas as pd
import streamlit as st
from ui_utils import *


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
st.session_state["query_type"] = query_type

response = {"initial_response": None}

# Submit button logic
if st.button("Submit"):
    with st.spinner("Waiting for results..."):
        
        response = fetch_response(query_type, query)

    if response["initial_response"] is not None:
        if query_type == "Dataset":
            with st.spinner("Using an LLM to find the most relevant information..."):
                llm_response = fetch_llm_response(query)
                initial_response = parse_and_update_response(query_type, response, llm_response, data_metadata, flow_metadata)
        else:
            initial_response = parse_and_update_response(query_type, response, None, data_metadata, flow_metadata)

        display_results(initial_response)

    with st.form("fb_form"):
        streamlit_feedback(
            feedback_type="thumbs",
            align="flex-start",
            key="fb_k",
            optional_text_label="[Optional] Please provide an explanation",
        )
        st.form_submit_button("Save feedback", on_click=feedback_cb)
