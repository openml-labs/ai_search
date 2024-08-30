# Documentation Bot

- This bot reads the documentation of OpenML and trains an LLM model to answer questions about the project.
- It consists of two main parts
  - A crawler that reads the documentation and stores the links and text information.
  - The LLM utils -> Vectore store, conversational langchain, history.

## How to run

- First run the crawler to get the documentation from OpenML. This will create a `data` folder with the documentation in it. ```python run_crawler.py```
- For inference, run ```uvicorn documentation_query:app --host 0.0.0.0 --port 8083 &```