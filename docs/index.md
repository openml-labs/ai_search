# RAG pipeline for OpenML
- This repository contains the code for the RAG pipeline for OpenML. 
- [Project roadmap](https://github.com/orgs/openml-labs/projects/3)

## Getting started
- Clone the repository
- Create a virtual environment and activate it
- Install the requirements using `pip install -r requirements.txt`
- Run training.py (for the first time/to update the model). This takes care of basically everything. (Refer to the training section for more details)
- Install Ollama (https://ollama.com/) for your machine
<!-- - Run `uvicorn backend:app` to start the FastAPI server.  -->
<!-- - Run `streamlit run main.py` to start the Streamlit frontend (this uses the FastAPI server so make sure it is running) -->
- For a local setup, you can run ./start_local.sh to start Olama, FastAPI and Streamlit servers. The Streamlit server will be available at http://localhost:8501
- For docker, refer to [Docker](docker.md)
- For a complete usage example refer to [pipeline usage](./developer%20tutorials/train%20and%20evaluate%20models.ipynb)
- Enjoy :)

## Example usage
- Note that in this picture, I am using a very very tiny model for demonstration purposes. The actual results would be a lot better :)
- ![Example usage](./images/search_ui.png)

## Where do I go from here?
### I am a developer and I want to contribute to the project
- Hello! We are glad you are here. To get started, refer to the tutorials in the [developer tutorial](./developer%20tutorials/index.md) section.
- If you have any questions, feel free to ask or post an issue.


### I just want to use the pipeline
- You can use the pipeline by running the Streamlit frontend. Refer to the getting started section above for more details.

### I am on the wrong page
![](./images/work.jpg)