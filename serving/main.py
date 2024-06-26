import streamlit as st
import aiofiles
import chromadb
from httpx import ConnectTimeout
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from tenacity import retry, retry_if_exception_type, stop_after_attempt
import asyncio
from modules.llm import *
from modules.utils import *

import requests


# Main Streamlit App
st.title("OpenML AI Search")

query_type = st.selectbox("Select Query Type", ["Dataset", "Flow"])
query = st.text_input("Enter your query")

if st.button("Submit"):
    if query_type == "Dataset":
        with st.spinner("waiting for results..."):
            response = requests.get(f"http://localhost:8000/dataset/{query}", json={"query": query, "type": "dataset"}).json()
    else:
        with st.spinner("waiting for results..."):
            response = requests.get(f"http://localhost:8000/flow/{query}", json={"query": query, "type": "flow"}).json()
    # print(response)
    
    if response["initial_response"] is not None:
        st.write("Results:")
        # st.write(response["initial_response"])
        # show dataframe
        st.dataframe(response["initial_response"])
        
        if response["llm_summary"] is not None:
            st.write("Summary:")
            st.write(response["llm_summary"])
