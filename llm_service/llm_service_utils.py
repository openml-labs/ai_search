
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import regex as re

def create_chain(prompt, model="llama3", temperature=0):
    """
    Description: Create a chain with the given prompt and model
    
    Input: prompt (str), model (str), temperature (float)
    
    Returns: chain (Chain)
    """
    llm = ChatOllama(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_template(prompt)

    return prompt | llm | StrOutputParser()


def parse_answers_initial(response, patterns):
    """
    Description: Parse the answers from the initial response
    
    Input: response (str), patterns (list)
    
    Returns: answers (list)
    """
    
    answers = []
    # if the response contains a ? and a new line then join the next line with it (sometimes the LLM adds a new line after the ? instead of just printing it on the same line)
    response = response.replace("?\n", "?")

    # convert the response to lowercase and split it into lines
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