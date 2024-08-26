import os
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ConnectTimeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from utils import ChromaStore, Crawler

# TODO : make this into a separate thing using config
recrawl_websites = False

crawled_files_data_path = "../data/crawler/crawled_data.csv"
chroma_path = "../data/crawler/"
model_name = "BAAI/bge-small-en"
generation_model_name = "llama3"  # ollama

# Crawl the websites and save the data
num_of_websites_to_crawl = None  # none for all

if not os.path.exists(chroma_path):
    os.makedirs(chroma_path, exist_ok=True)

# Crawl the websites and save the data
crawler = Crawler(
    crawled_files_data_path=crawled_files_data_path,
    recrawl_websites=recrawl_websites,
    num_of_websites_to_crawl=num_of_websites_to_crawl,
)
crawler.do_crawl()

# Initialize the ChromaStore and embed the data
chroma_store = ChromaStore(
    model_name=model_name,
    crawled_files_data_path=crawled_files_data_path,
    chroma_file_path=chroma_path,
    generation_model_name=generation_model_name,
)
if recrawl_websites == True:
    chroma_store.read_data_and_embed()

app = FastAPI()
session_id = str(uuid.uuid4())


def stream_response(response):
    for line in response:
        try:
            yield str(line["answer"])
        except GeneratorExit:
            break
        except:
            yield ""


@app.get("/documentationquery/{query}", response_class=JSONResponse)
@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ConnectTimeout))
async def get_documentation_query(query: str):
    query = query.replace("%20", " ")
    print(f"Query: {query}")

    chroma_store.setup_inference(session_id)
    response = chroma_store.openml_page_search(input=query)
    # return JSONResponse(content=response)
    return StreamingResponse(stream_response(response), media_type="text/event-stream")
