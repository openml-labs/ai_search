import json
from fastapi import FastAPI, HTTPException
from llm_service_structured_query_utils import create_query_structuring_chain
from fastapi.responses import JSONResponse
from httpx import ConnectTimeout

document_content_description = "Metadata of datasets for various machine learning applications fetched from OpenML platform."

content_attr = [
    "status",
    "NumberOfClasses",
    "NumberOfFeatures",
    "NumberOfInstances"
]

chain = create_query_structuring_chain(document_content_description, content_attr, model = "llama3")

from tenacity import retry, retry_if_exception_type, stop_after_attempt
from langchain_community.query_constructors.chroma import ChromaTranslator

app = FastAPI()

print("[INFO] Starting structured query service.")
# Create port
@app.get("/structuredquery/{query}", response_class=JSONResponse)
@retry(stop=stop_after_attempt(1), retry=retry_if_exception_type(ConnectTimeout))
async def get_structured_query(query: str):
    """
    Description: Get the query, replace %20 with space and invoke the chain to get the answers based on the prompt.

    """
    try:
        query = query.replace("%20", " ")
        response = chain.invoke({"query": query})
        obj = ChromaTranslator()
        filter_condition = obj.visit_structured_query(structured_query=response)[1]
        return response, filter_condition
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON decode error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
        
        
   
    
