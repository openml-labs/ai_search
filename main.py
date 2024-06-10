import chromadb
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

from modules.llm import *
from modules.utils import *
import aiofiles
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache


app = FastAPI()
# Config and DB

# load the configuration and device
config = load_config_and_device("config.json")
# load the persistent database using ChromaDB
client = chromadb.PersistentClient(path=config["persist_dir"])
print(config)
# Loading the metadata for all types

# Setup llm chain, initialize the retriever and llm, and setup Retrieval QA
qa_dataset = setup_vector_db_and_qa(config=config, data_type="dataset", client=client)
qa_flow = setup_vector_db_and_qa(config=config, data_type="flow", client=client)
set_llm_cache(SQLiteCache(database_path="./data/.langchain.db"))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    async with aiofiles.open("index.html", mode="r") as f:
        html_content = await f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/dataset/{query}", response_class=JSONResponse)
async def read_dataset(query: str):
    try:
        result_data_frame = get_result_from_query(query=query, qa=qa_dataset, type_of_query="dataset", config=config)
        print(result_data_frame.head())
        return JSONResponse(content=result_data_frame.to_dict(orient="records"), status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/flow/{query}", response_class=JSONResponse)
async def read_flow(query: str):
    try:
        result_data_frame = get_result_from_query(query=query, qa=qa_flow, type_of_query="flow", config=config)
        print(result_data_frame.head())
        return JSONResponse(content=result_data_frame.to_dict(orient="records"), status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# qa_dataset = setup_vector_db_and_qa(config=config, data_type="dataset", client=client)
# result_data_frame = get_result_from_query(
#     query="Find me a dataset about flowers that has a high number of instances.",
#     qa=qa_dataset,
#     type_of_query="dataset",
#     config=config,
# )
# print(result_data_frame)

# qa_flow = setup_vector_db_and_qa(config=config, data_type="flow", client=client)
# result_data_frame = get_result_from_query(
#     query="Find me a flow for image classification.",
#     qa=qa_flow,
#     type_of_query="flow",
#     config=config,
# )
# print(result_data_frame)