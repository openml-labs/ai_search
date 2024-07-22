import chromadb
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ConnectTimeout
from modules.rag_llm import *
from modules.utils import *
from tenacity import retry, retry_if_exception_type, stop_after_attempt

app = FastAPI()
# Config and DB

# load the configuration and device
config = load_config_and_device("config.json")
if config["testing_flag"]:
    config["persist_dir"] = "./data/chroma_db_testing/"
    config["test_subset"] = True
    config["data_dir"] = "./data/testing_data/"
# load the persistent database using ChromaDB
client = chromadb.PersistentClient(path=config["persist_dir"])
# Loading the metadata for all types

# Setup llm chain, initialize the retriever and llm, and setup Retrieval QA

qa_dataset_handler = QASetup(
    config=config,
    data_type="dataset",
    client=client,
)

qa_dataset, _ = qa_dataset_handler.setup_vector_db_and_qa()

qa_flow_handler = QASetup(
    config=config,
    data_type="flow",
    client=client,
)

qa_flow, _ = qa_flow_handler.setup_vector_db_and_qa()

# get the llm chain and set the cache
llm_chain_handler = LLMChainCreator(config=config, local=True)
llm_chain_handler.enable_cache()
llm_chain = llm_chain_handler.get_llm_chain()


# Send test query as first query to avoid cold start
try:
    print("[INFO] Sending first query to avoid cold start.")
    for type_of_query in ["dataset", "flow"]:
        QueryProcessor(
            query="mushroom",
            qa=qa_dataset if type_of_query == "dataset" else qa_flow,
            type_of_query=type_of_query,
            config=config,
        ).get_result_from_query()

except Exception as e:
    print("Error in first query: ", e)


@app.get("/dataset/{query}", response_class=JSONResponse)
@retry(retry=retry_if_exception_type(ConnectTimeout), stop=stop_after_attempt(2))
async def read_dataset(query: str):
    try:
        # Fetch the result data frame based on the query
        _, ids_order = QueryProcessor(
            query=query,
            qa=qa_dataset if type_of_query == "dataset" else qa_flow,
            type_of_query=type_of_query,
            config=config,
        ).get_result_from_query()

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
        _, ids_order = QueryProcessor(
            query=query,
            qa=qa_dataset if type_of_query == "flow" else qa_flow,
            type_of_query=type_of_query,
            config=config,
        ).get_result_from_query()

        response = JSONResponse(
            content={"initial_response": ids_order}, status_code=200
        )

        return response
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
