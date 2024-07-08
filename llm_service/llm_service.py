import regex as re
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ConnectTimeout
from langchain_community.chat_models import ChatOllama
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
    llm = ChatOllama(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_template(prompt)

    # using LangChain Expressive Language chain syntax
    # learn more about the LCEL on
    # /docs/concepts/#langchain-expression-language-lcel
    return prompt | llm | StrOutputParser()


def parse_answers_initial(response):
    # for each line in the response, split by ? and check if the response is Yes/No or a comma separated string of Yes/No or ascending/descending using regex
    answers = []
    for line in response.lower().split("\n"):
        if "?" in line:
            response = line.split("?")[1].strip()
            if response in ["yes", "no", "none"]:
                answers.append(response)
            # elif re.match(r"^(Yes|No),\s?(Yes|No)$", response):
            # match for Yes/No or ascending/descending and full stop
            elif re.match(r"^(yes|no)", response):
                answers.append(response)
            elif re.match(r"^(ascending|descending)", response):
                answers.append(response)
            elif re.match(r"(multi-class|binary|multi-label)", response):
                answers.append(response)
        # if any of the words are in the line, append the line to the answers
        elif any(word in line for word in ["yes", "no", "none", "ascending", "descending", "multi-class", "binary", "multi-label"]):
            answers.append(line.strip())
    return answers


chain = create_chain(prompt)

app = FastAPI()


# Create port
@app.get("/llmquery/{query}", response_class=JSONResponse)
@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ConnectTimeout))
async def get_llm_query(query: str):
    query = query.replace("%20", " ")
    response = chain.invoke({"query": query})
    print(response)
    answers = parse_answers_initial(response)
    return JSONResponse(content={"answers": answers})
