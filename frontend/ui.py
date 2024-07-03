import streamlit as st
from streamlit_feedback import streamlit_feedback
from streamlit import session_state as ss
import requests
import json
import os

# Main Streamlit App
st.title("OpenML AI Search")

query_type = st.selectbox("Select Query Type", ["Dataset", "Flow"])
# query = st ("Enter your query")
query = st.text_input("Enter your query")

st.session_state["query"] = query

def feedback_cb():
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
    data.append({"ss": ss.fb_k, "llm_summary": ss.llm_summary, "query": ss.query})

    # Write updated content back to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

if st.button("Submit"):
    if query_type == "Dataset":
        with st.spinner("waiting for results..."):
            try:
                response = requests.get(f"http://fastapi:8000/dataset/{query}", json={"query": query, "type": "dataset"}).json()
            except:
                response = requests.get(f"http://0.0.0.0:8000/dataset/{query}", json={"query": query, "type": "dataset"}).json()
    else:
        with st.spinner("waiting for results..."):
            try:
                response = requests.get(f"http://fastapi:8000/flow/{query}", json={"query": query, "type": "flow"}).json()
            except:
                response = requests.get(f"http://0.0.0.0:8000/flow/{query}", json={"query": query, "type": "flow"}).json()
    # print(response)

    # response = {"initial_response": "dummy", "llm_summary": "dummy"}
    
    if response["initial_response"] is not None:
        st.write("Results:")
        # st.write(response["initial_response"])
        # show dataframe
        st.dataframe(response["initial_response"])
        
        if response["llm_summary"] is not None:
            st.write("Summary:")
            st.write(response["llm_summary"])

    with st.form('fb_form'):
        st.session_state["llm_summary"] = response["llm_summary"]
        streamlit_feedback(feedback_type="thumbs", align="flex-start", key='fb_k',optional_text_label="[Optional] Please provide an explanation", on_submit=feedback_cb)
        # st.form_submit_button('Save feedback', on_click=feedback_cb)
