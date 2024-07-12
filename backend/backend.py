import chromadb
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ConnectTimeout
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from modules.llm import *
from modules.utils import *
from tenacity import retry, retry_if_exception_type, stop_after_attempt

app = FastAPI()
# Config and DB

# load the configuration and device
config = load_config_and_device("config.json")
if config["testing_flag"] == True:
    config["persist_dir"] = "./data/chroma_db_testing/"
    config["test_subset"] = True
    config["data_dir"] = "./data/testing_data/"
# load the persistent database using ChromaDB
client = chromadb.PersistentClient(path=config["persist_dir"])
print(config)
# Loading the metadata for all types

# Setup llm chain, initialize the retriever and llm, and setup Retrieval QA
qa_dataset, _ = setup_vector_db_and_qa(
    config=config, data_type="dataset", client=client
)
qa_flow, _ = setup_vector_db_and_qa(config=config, data_type="flow", client=client)

# get the llm chain and set the cache
llm_chain = get_llm_chain(config=config, local=True)
# use os path to ensure compatibility with all operating systems
set_llm_cache(
    SQLiteCache(database_path=os.path.join(config["data_dir"], ".langchain.db"))
)

# Send test query as first query to avoid cold start
try:
    print("[INFO] Sending first query to avoid cold start.")
    get_result_from_query(
        query="mushroom", qa=qa_dataset, type_of_query="dataset", config=config
    )
    get_result_from_query(
        query="physics flow", qa=qa_flow, type_of_query="flow", config=config
    )

except Exception as e:
    print("Error in first query: ", e)


@app.get("/dataset/{query}", response_class=JSONResponse)
@retry(retry=retry_if_exception_type(ConnectTimeout), stop=stop_after_attempt(2))
async def read_dataset(query: str):
    try:
        # Fetch the result data frame based on the query
        _, ids_order = get_result_from_query(
            query=query, qa=qa_dataset, type_of_query="dataset", config=config
        )

        response = JSONResponse(
            content={"initial_response": ids_order}, status_code=200
        )

        return response

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/flow/{query}", response_class=JSONResponse)
@retry(retry=retry_if_exception_type(ConnectTimeout), stop=stop_after_attempt(2))
async def read_flow(query: str):
    try:
        _, ids_order = get_result_from_query(
            query=query, qa=qa_flow, type_of_query="flow", config=config
        )

        response = JSONResponse(
            content={"initial_response": ids_order}, status_code=200
        )

        return response
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
