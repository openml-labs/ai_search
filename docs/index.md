# RAG pipeline for OpenML
- This repository contains the code for the RAG pipeline for OpenML. 

## Getting started
- A docker image will be provided at a later date for easier setup
- Clone the repository
- Create a virtual environment and activate it
- Install the requirements using `pip install -r requirements.txt`
- Run training.py (for the first time/to update the model). This takes care of basically everything. (Refer to the training section for more details)
- Install Ollama (https://ollama.com/) and download the models `ollama run qwen2:1.5b` and `ollama run phi3`
- Run `uvicorn backend:app` to start the FastAPI server. 
- Run `streamlit run main.py` to start the Streamlit frontend (this uses the FastAPI server so make sure it is running)
- Enjoy :)
