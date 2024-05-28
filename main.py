import chromadb
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from modules.llm import *
from modules.utils import *
# Config and DB

# load the configuration and device
config = load_config_and_device("config.json")
# load the persistent database using ChromaDB
client = chromadb.PersistentClient(path=config["persist_dir"])
print(config)
# Loading the metadata for all types

# Setup llm chain, initialize the retriever and llm, and setup Retrieval QA
qa_flow = setup_vector_db_and_qa(config=config, data_type="flow", client=client)
qa_dataset = setup_vector_db_and_qa(config=config, data_type="dataset", client=client)

# Start the FastAPI app

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_root():
    return {"Hello": "World"}

@app.get("/dataset/{query}", response_class=HTMLResponse)
def read_item(query: str):
    query = query.replace(
        "%20", " "
    )  # replace %20 with space character (browsers do this automatically when spaces are in the URL)
    # get results with unique names
    # TODO CHECK BEHAVIOR
    result_data_frame = get_result_from_query(
        query=query, qa=qa_dataset, type_of_query="dataset"
    )
    return result_data_frame.to_html()


@app.get("/flow/{query}", response_class=HTMLResponse)
def read_item(query: str):
    query = query.replace(
        "%20", " "
    )  # replace %20 with space character (browsers do this automatically when spaces are in the URL)
    # get results with unique names
    # TODO CHECK BEHAVIOR
    result_data_frame = get_result_from_query(
        query=query, qa=qa_flow, type_of_query="flow"
    )
    return result_data_frame.to_html()
