#!/bin/bash
poetry install
killall ollama
killall streamlit
# Define a file to store the PIDs
PID_FILE="processes.pid"

# Start processes and save their PIDs
cd ollama
./get_ollama.sh &
echo $! > $PID_FILE

cd ../structured_query
uvicorn llm_service_structured_query:app --host 0.0.0.0 --port 8081 &
echo $! > $PID_FILE

cd ..
# Keep the script running to maintain the background processes
wait