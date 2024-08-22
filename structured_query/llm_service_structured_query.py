import json
from fastapi import FastAPI, HTTPException
from llm_service_structured_query_utils import create_query_structuring_chain
from fastapi.responses import JSONResponse
from httpx import ConnectTimeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from langchain_community.query_constructors.chroma import ChromaTranslator

document_content_description = "Metadata of datasets for various machine learning applications fetched from OpenML platform."

content_attr = ["status", "NumberOfClasses", "NumberOfFeatures", "NumberOfInstances"]

chain = create_query_structuring_chain(
    document_content_description, content_attr, model="llama3"
)
print("[INFO] Chain created.")

app = FastAPI(root_path="/struct")

try:
    print("[INFO] Sending first query to structured query llm to avoid cold start.")
    
    query = "mushroom data with 2 classess"
    response = chain.invoke({"query": query})    
    obj = ChromaTranslator()
    filter_condition = obj.visit_structured_query(structured_query=response)[1]
    print(response, filter_condition)

except Exception as e:
    print("Error in first query: ", e)


# Create port
@app.get("/structuredquery/{query}", response_class=JSONResponse)
@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ConnectTimeout))
async def get_structured_query(query: str):
    """
    Description: Get the query, replace %20 with space and invoke the chain to get the answers based on the prompt.

    """
    try:
        query = query.replace("%20", " ")
        response = chain.invoke({"query": query})
        print(response)
        obj = ChromaTranslator()
        filter_condition = obj.visit_structured_query(structured_query=response)[1]
        
    except Exception as e:
        print(f"An error occurred: ", HTTPException(status_code=500, detail=f"An error occurred: {e}"))
        response, filter_condition = None, None
        
    return response, filter_condition
        
        
   
    
