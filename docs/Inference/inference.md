# Inference
- Just run ./start_local.sh and it will take care of everything.
- The UI should either pop up or you can navigate to http://localhost:8501/ in your browser.
- Note that it takes a decent bit of time to load everything. 

## Stopping
- Run ./stop_local.sh
- ./start_local.sh stores the PIDs of all the processes it starts in files in all the directories it starts them in. stop_local.sh reads these files and kills the processes.

## CLI access to the API
- We all are lazy sometimes and don't want to use the interface sometimes. Or just want to test out different parts of the API without any hassle. To that end, you can either test out the individual components like so: 
- Note that the `%20` are spaces in the URL. 

### Ollama
- This is the server that runs an Ollama server (This is basically an optimized version of a local LLM. It does not do anything of itself but runs as a background service so you can use the LLM). 
- You can start it by running `cd ollama && ./get_ollama.sh &`

### LLM Service
- This component is the one that runs the query processing using LLMs module. It uses the Ollama server, runs queries and processes them. 
- You can start it by running `cd llm_service && uvicorn llm_service:app --host 0.0.0.0 --port 8081 &`
- Curl Example : `curl http://0.0.0.0:8081/llmquery/find%20me%20a%20mushroom%20dataset%20with%20less%20than%203000%20classes`

### Backend
- This component runs the RAG pipeline. It returns a JSON with dataset ids of the OpenML datasets that match the query.
- You can start it by running `cd backend && uvicorn backend:app --host 0.0.0.0 --port 8000 &`
- Curl Example : `curl http://0.0.0.0:8000/dataset/find%20me%20a%20mushroom%20dataset`

### Frontend
- This component runs the Streamlit frontend. It is the UI that you see when you navigate to `http://localhost:8501`.
- You can start it by running `cd frontend && streamlit run ui.py &`

## Errors
- If you get an error about file permissions, run `chmod +x start_local.sh` and `chmod +x stop_local.sh` to make them executable.

:::ui_utils