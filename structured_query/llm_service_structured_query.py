import json
import sys

from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt, load_query_constructor_runnable)
from structured_query_examples import examples

sys.path.append("../")
sys.path.append("../data")

with open("attribute_info.json", "r") as f:
    attribute_info = json.loads(f.read())

attribute_info = attribute_info[1:]

examples = examples
document_content_description = "Metadata of datasets for various machine learning applications fetched from OpenML platform"
prompt = get_query_constructor_prompt(
    document_contents=document_content_description,
    attribute_info=attribute_info,
    examples=examples,
)

from langchain_community.chat_models import ChatOllama

content_attr = [
    "status",
    "NumberOfClasses",
    "NumberOfFeatures",
    "NumberOfInstances",
    "Combined_information",
]
# document_content_description = "Metadata of machine learning datasets including status (if dataset is active or not), number of classes in the dataset, number of instances (examples) in the dataset, number of features in the dataset, and Combined_information containing the combined metadata information about the dataset."
filter_attribute_info = tuple(ai for ai in attribute_info if ai["name"] in content_attr)

chain = load_query_constructor_runnable(
    ChatOllama(model="llama3"),
    document_content_description,
    # attribute_info,
    filter_attribute_info,
    examples=examples,
    fix_invalid=True,
)

# def structuring_query(query:str):
#     structured_query = chain.invoke(query)

#     return structured_query.query, structured_query.filter

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ConnectTimeout
# from llm_service_utils import create_chain, parse_answers_initial
from tenacity import retry, retry_if_exception_type, stop_after_attempt

app = FastAPI()


# Create port
@app.get("/structuredquery/{query}", response_class=JSONResponse)
@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ConnectTimeout))
async def get_structured_query(query: str):
    """
    Description: Get the query, replace %20 with space and invoke the chain to get the answers based on the prompt


    """
    query = query.replace("%20", " ")
    response = chain.invoke({"query": query})
    return response
