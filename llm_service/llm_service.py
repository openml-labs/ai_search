from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ConnectTimeout
from llm_service_utils import create_chain, parse_answers_initial
from tenacity import retry, retry_if_exception_type, stop_after_attempt

prompt_template = """User Query : {query}
Based on the query, answer the following questions one by one in one or two words only and a maximum of two with commas only if asked for. Use only the information given and do not make up answers. Do not explain your reasoning and do not add unncessesary new lines - 
"""

prompt_dict = {
    "size_of_dataset": "Does the user care about the size of the dataset? Yes/No and if yes, ascending/descending.",
    "missing_values": "Does the user care about missing values? Yes/No.",
    "classification_type": "If it seems like the user wants a classification dataset, is it binary/multi-class/multi-label. If not, say none.",
    "uploader": "If the user mentions an uploader id, say id = uploader , otherwise say none.",
}

# patterns to match the answers to
patterns = [
    r"^(yes|no|none)",
    r"^(ascending|descending)",
    r"(multi-class|binary|multi-label|name)",
]

# join the prompt dictionary to the prompt template to create the final prompt
prompt = prompt_template + "\n".join([prompt_dict[key] for key in prompt_dict.keys()])


chain = create_chain(prompt)
chain_docker = create_chain(prompt, base_url="http://ollama:11434")
app = FastAPI()


# Create port
@app.get("/llmquery/{query}", response_class=JSONResponse)
@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ConnectTimeout))
async def get_llm_query(query: str):
    """
    Description: Get the query, replace %20 (url spacing) with space and invoke the chain to get the answers based on the prompt
    """
    query = query.replace("%20", " ")
    print(f"Query: {query}")
    try:
        response = chain_docker.invoke({"query": query})
    except:
        response = chain.invoke({"query": query})
    answers = parse_answers_initial(response, patterns, prompt_dict)
    return JSONResponse(content=answers)
