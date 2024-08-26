from pathlib import Path

import streamlit as st
from ui_utils import *

# Streamlit Chat Interface
logo = "images/favicon.ico"
page_title = "OpenML : A worldwide machine learning lab"
info = """
    <p style='text-align: center; color: white;'>Machine learning research should be easily accessible and reusable. OpenML is an open platform for sharing datasets, algorithms, and experiments - to learn how to learn better, together. <br>Ask me anything about OpenML or search for a dataset ... </p>
    """
chatbot_display = "How do I do X using OpenML? / Find me a dataset about Y"
chatbot_max_chars = 500

st.set_page_config(page_title=page_title, page_icon=logo)
st.title("OpenML AI Search")
# message_box = st.container()

with st.spinner("Loading Required Data"):
    config_path = Path("../backend/config.json")
    ui_loader = UILoader(config_path)

# container for company description and logo
# with st.sidebar:
#     query_type = st.radio(
#         "Select Query Type", ["General Query", "Dataset", "Flow"], key="query_type_2"
#     )
col1, col2 = st.columns([1, 4])
with col1:
    st.image(logo, width=100)
with col2:
    st.markdown(
        info,
        unsafe_allow_html=True,
    )

chat_container = st.container()
with chat_container:

    with st.form(key="chat_form"):
        user_input = st.text_input(label=chatbot_display)
        query_type = st.selectbox("Select Query Type", ["General Query", "Dataset", "Flow"])
        submit_button = st.form_submit_button(label="Submit")

    ui_loader.create_chat_interface(user_input=None)
    if user_input:
        ui_loader.create_chat_interface(user_input, query_type=query_type)
