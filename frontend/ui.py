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


# Submit button logic
if st.button("Submit"):
    response_parser = ResponseParser(query_type)
    if query_type == "Dataset":
        with st.spinner("Waiting for results..."):
            # get rag response
            response_parser.fetch_rag_response(query_type, query)
            # get llm response
            response_parser.fetch_llm_response(query)
            # get updated columns based on llm response
            results = response_parser.parse_and_update_response(data_metadata)
            # display results in a table
            display_results(results)

    with st.form("fb_form"):
        streamlit_feedback(
            feedback_type="thumbs",
            align="flex-start",
            key="fb_k",
            optional_text_label="[Optional] Please provide an explanation",
        )
        st.form_submit_button("Save feedback", on_click=feedback_cb)
