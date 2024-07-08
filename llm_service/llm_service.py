import regex as re
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ConnectTimeout
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, retry_if_exception_type, stop_after_attempt

prompt = """User Query : {query}
Based on the query, answer the following questions one by one in one or two words only and a maximum of two with commas only if asked for. Use only the information given and do not make up answers. Do not explain your reasoning and do not add unncessesary new lines - 
Does the user care about the size of the dataset? Yes/No and if yes, ascending/descending.
Does the user care about missing values? Yes/No.
If it seems like the user wants a classification dataset, is it binary/multi-class/multi-label. If not, say none.
"""


def create_chain(prompt, model="llama3", temperature=0):
    """
    Description: Create a chain with the given prompt and model
    
    Input: prompt (str), model (str), temperature (float)
    
    Returns: chain (Chain)
    """
    llm = ChatOllama(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_template(prompt)

    return prompt | llm | StrOutputParser()


def parse_answers_initial(response):
    """
    Description: Parse the answers from the initial response
    
    Input: response (str)
    
    Returns: answers (list)
    """
    patterns = [
        r"^(yes|no|none)",
        r"^(ascending|descending)",
        r"(multi-class|binary|multi-label)"
    ]
    
    answers = []
    lines = response.lower().split("\n")
    
    for line in lines:
        if "?" in line:
            # Extract the part of the line after the question mark
            potential_answer = line.split("?")[1].strip()
        else:
            potential_answer = line.strip()
        
        # Check if the potential answer matches any of the patterns
        for pattern in patterns:
            if re.match(pattern, potential_answer):
                answers.append(potential_answer)
                break  # Stop checking other patterns if a match is found
    
    return answers

chain = create_chain(prompt)

app = FastAPI()


# Create port
@app.get("/llmquery/{query}", response_class=JSONResponse)
@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ConnectTimeout))
async def get_llm_query(query: str):
    query = query.replace("%20", " ")
    response = chain.invoke({"query": query})
    answers = parse_answers_initial(response)
    return JSONResponse(content={"answers": answers})
