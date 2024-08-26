from pathlib import Path

import streamlit as st
from ui_utils import *

# Streamlit Chat Interface
page_title = "OpenML : A worldwide machine learning lab"

chatbot_max_chars = 500

st.set_page_config(page_title=page_title)
st.title("OpenML AI Search")
# message_box = st.container()

with st.spinner("Loading Required Data"):
    config_path = Path("../backend/config.json")
    ui_loader = UILoader(config_path)
    ui_loader.generate_complete_ui()
