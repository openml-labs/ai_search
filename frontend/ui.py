import streamlit as st
import requests
# Main Streamlit App
st.title("OpenML AI Search")

query_type = st.selectbox("Select Query Type", ["Dataset", "Flow"])
query = st.text_input("Enter your query")

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
    
    if response["initial_response"] is not None:
        st.write("Results:")
        # st.write(response["initial_response"])
        # show dataframe
        st.dataframe(response["initial_response"])
        
        if response["llm_summary"] is not None:
            st.write("Summary:")
            st.write(response["llm_summary"])
