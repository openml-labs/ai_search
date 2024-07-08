import json
from pathlib import Path

import pandas as pd
import streamlit as st
from ui_utils import run_streamlit


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
run_streamlit()