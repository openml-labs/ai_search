# LLM Query parsing

- This page is only an overview. Please refer to the api reference for more detailed information.
- The query parsing LLM reads the query and parses it into a list of filters based on a prompt. The expected result is a JSON with a list of filters to be applied to the metadata and the query.
- This is done by providing a prompt to the RAG and telling it to extract the filters/etc and either structure it or not.
- This implementation is served as a FastAPI service that can be queried quite easily.

## Structured Implementation

- TBD

## Unstructured Implementation
- This implementation is independent of `langchain`, and takes a more manual approach to parsing the filters. At the moment, this does not separate the query from the filters either. (The structured query implementation attempts to do that.)
- The response of the the LLM parser does not take into account how to apply the filters, it just provides a list of the ones that the LLM considered relevant to the UI.
- This component is the one that runs the query processing using LLMs module. It uses the Ollama server, runs queries and processes them. 
- You can start it by running `cd llm_service && uvicorn llm_service:app --host 0.0.0.0 --port 8081 &`
- Curl Example : `curl http://0.0.0.0:8081/llmquery/find%20me%20a%20mushroom%20dataset%20with%20less%20than%203000%20classes`

### llm_service.py
- A prompt template is used to tell the RAG what to do. 
- The prompt_dict defines a list of filters and their respective prompts for the LLM. This is concatenated with the prompt template.
- The response is parsed quite simply. Since the LLM is asked to provide it's answers line by line, each line is parsed for the required information according to a list of patterns provided. 
- Thus, if you want to add a new type of answer, add it to the patterns list and it should be taken care of.

### llm_service_utils.py
- The main logic of the above is defined here.

## Additional information
- In the process of testing this implementation, a blog was written about how the temperature parameter affects the results of the model. This can be [found here](https://openml-labs.github.io/blog/posts/Experiments-with-temperature/experiments_with_temp.html).
