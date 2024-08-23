import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_feedback import streamlit_feedback
from ui_utils import *

# Streamlit Chat Interface
logo = "images/favicon.ico"
page_title = "OpenML : A worldwide machine learning lab"
info = """
    <p style='text-align: center; color: white;'>Machine learning research should be easily accessible and reusable. OpenML is an open platform for sharing datasets, algorithms, and experiments - to learn how to learn better, together. <br>Ask me anything about OpenML or search for a dataset ... </p>
    """
st.set_page_config(page_title=page_title, page_icon=logo)
st.title("OpenML AI Search")

# container for company description and logo
col1, col2 = st.columns([1, 4])
with col1:
    st.image(logo, width=100)
with col2:
    st.markdown(
        info,
        unsafe_allow_html=True,
    )

with st.spinner("Loading Required Data"):
    config_path = Path("../backend/config.json")
    ui_loader = UILoader(config_path)

# Chat input box
user_input = ui_loader.chat_entry()

ui_loader.create_chat_interface(None)
query_type = st.selectbox("Select Query Type", ["General Query","Dataset", "Flow"], key="query_type_2")
llm_filter = st.toggle("LLM Filter")
# Chat interface
if user_input:
    ui_loader.create_chat_interface(
        user_input, query_type=query_type, llm_filter=llm_filter
    )
    ui_loader.query_type = st.selectbox("Select Query Type", ["General Query","Dataset", "Flow"], key="query_type_3")
    ui_loader.llm_filter = st.toggle("LLM Filter", key="llm_filter_2")

