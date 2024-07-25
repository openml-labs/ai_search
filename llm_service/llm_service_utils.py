import regex as re
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def create_chain(prompt, model: str = "llama3", temperature: int = 0):
    """
    Description: Create a langchain chain with the given prompt and model and the temperature.
    The lower the temperature, the less "creative" the model will be.
    """
    llm = ChatOllama(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_template(prompt)

    return prompt | llm | StrOutputParser()


def parse_answers_initial(response: str, patterns: list, prompt_dict: dict) -> dict:
    """
    Description: Parse the answers from the initial response
    - if the response contains a ? and a new line then join the next line with it (sometimes the LLM adds a new line after the ? instead of just printing it on the same line)
    """

    answers = []
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

    # return answers as a dict using the prompt_dict keys
    answers_dict = {}
    for i, key in enumerate(prompt_dict.keys()):
        answers_dict[key] = answers[i]

    return answers_dict
