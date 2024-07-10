# Inference
- Just run ./start_local.sh and it will take care of everything.
- The UI should either pop up or you can navigate to http://localhost:8501/ in your browser.
- Note that it takes a decent bit of time to load everything. 

## Stopping
- Run ./stop_local.sh
- ./start_local.sh stores the PIDs of all the processes it starts in files in all the directories it starts them in. stop_local.sh reads these files and kills the processes.

## Starting individual services

# Start processes 
## Ollama
cd ollama
./get_ollama.sh &

## LLM Service
cd ../llm_service
uvicorn llm_service:app --host 0.0.0.0 --port 8081 &

## Backend
cd ../backend
uvicorn backend:app --host 0.0.0.0 --port 8000 &

## Frontend
cd ../frontend
streamlit run ui.py &

## Errors
- If you get an error about file permissions, run `chmod +x start_local.sh` and `chmod +x stop_local.sh` to make them executable.

:::ui_utils