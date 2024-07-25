import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_feedback import streamlit_feedback
from ui_utils import *

with open("../backend/config.json", "r") as file:
    config = json.load(file)

# Load metadata chroma database
if config['structured_query']:
    import sys
    sys.path.append('../')
    from structured_query.chroma_store_utilis import *
    collec = load_chroma_metadata() 
       
# Metadata paths
data_metadata_path = Path(config["data_dir"]) / "all_dataset_description.csv"
flow_metadata_path = Path(config["data_dir"]) / "all_flow_description.csv"

# Load metadata
data_metadata = pd.read_csv(data_metadata_path)
flow_metadata = pd.read_csv(flow_metadata_path)

# Main Streamlit App
st.title("OpenML AI Search")

query_type = st.selectbox("Select Query Type", ["Dataset", "Flow"])
query = st.text_input("Enter your query")

st.session_state["query"] = query
st.session_state["query_type"] = query_type

# Submit button logic
if st.button("Submit"):
    response_parser = ResponseParser(query_type, apply_llm_before_rag=False)
    if query_type == "Dataset":
        with st.spinner("Waiting for results..."):
            if config["structured_query"] == True:
                # get structured query
                response_parser.fetch_structured_query(
                    query_type, query
                )
                if response_parser.structured_query_response:
                    st.write(response_parser.structured_query_response[0])
                    # get rag response
                    response_parser.fetch_rag_response(
                        query_type, response_parser.structured_query_response[0]["query"]
                    )
                    if response_parser.structured_query_response[1].get("filter"):
                        response_parser.database_filter(response_parser.structured_query_response[1]["filter"], collec)
                else:
                    # get rag response
                    response_parser.fetch_rag_response(
                        query_type, query
                    )
            else:
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
