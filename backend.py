import aiofiles
import chromadb
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from httpx import ConnectTimeout
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from modules.llm import *
from modules.utils import *

app = FastAPI()
# Config and DB

# load the configuration and device
config = load_config_and_device("config.json")
if config["testing_flag"] == True:
    config["persist_dir"] = "./data/chroma_db_testing/"
    config["test_subset_2000"] = True
    config["data_dir"] = "./data/testing_data/"
# load the persistent database using ChromaDB
client = chromadb.PersistentClient(path=config["persist_dir"])
print(config)
# Loading the metadata for all types

# Setup llm chain, initialize the retriever and llm, and setup Retrieval QA
qa_dataset = setup_vector_db_and_qa(config=config, data_type="dataset", client=client)
qa_flow = setup_vector_db_and_qa(config=config, data_type="flow", client=client)
# use os path to ensure compatibility with all operating systems
set_llm_cache(SQLiteCache(database_path=os.path.join(config["data_dir"], ".langchain.db")))

# Send test query as first query to avoid cold start
try:
    print("[INFO] Sending first query to avoid cold start.")
    result_data_frame = get_result_from_query(
        query="mushroom", qa=qa_dataset, type_of_query="dataset", config=config
    )
    result_data_frame = get_result_from_query(
        query="physics flow", qa=qa_flow, type_of_query="flow", config=config
    )
except Exception as e:
    print("Error in first query: ", e)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    async with aiofiles.open("index.html", mode="r") as f:
        html_content = await f.read()
    return HTMLResponse(content=html_content, status_code=200)



@app.get("/dataset/{query}", response_class=JSONResponse)
@retry(retry=retry_if_exception_type(ConnectTimeout), stop=stop_after_attempt(2))
async def read_dataset(query: str):
    try:
        # Fetch the result data frame based on the query
        result_data_frame, result_documents = get_result_from_query(
            query=query, qa=qa_dataset, type_of_query="dataset", config=config
        )

        # Respond with the result data frame
        initial_response = result_data_frame.to_dict(orient="records")
        
        llm_summary = await get_llm_result(result_documents[:config["num_documents_for_llm"]], config=config)
        response = JSONResponse(content={"initial_response": initial_response, "llm_summary": llm_summary}, status_code=200)
        
        return response

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/flow/{query}", response_class=JSONResponse)
@retry(retry=retry_if_exception_type(ConnectTimeout), stop=stop_after_attempt(2))
async def read_flow(query: str):
    try:
        result_data_frame = get_result_from_query(
            query=query, qa=qa_flow, type_of_query="flow", config=config
        )
        return JSONResponse(
            content=result_data_frame.to_dict(orient="records"), status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
