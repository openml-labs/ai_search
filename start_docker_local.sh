#!/bin/bash
# poetry install
# killall ollama
# killall streamlit
# Define a file to store the PIDs
PID_FILE="processes.pid"

# Start processes and save their PIDs
cd ollama || exit
./get_ollama.sh &
echo $! > $PID_FILE

# Fetch configuration from ../backend/config.json
config_file="backend/config.json"
# structured_query=$(jq -r '.structured_query' $config_file)

cd ../structured_query || exit
uvicorn llm_service_structured_query:app --host 0.0.0.0 --port 8081 &
echo $! > $PID_FILE

cd ../documentation_bot || exit
uvicorn documentation_query:app --host 0.0.0.0 --port 8083 &
echo $! >> $PID_FILE

cd ../backend || exit
uvicorn backend:app --host 0.0.0.0 --port 8000 &
echo $! >> $PID_FILE

cd ../frontend || exit
streamlit run ui.py &
echo $! >> $PID_FILE

cd ..
# Keep the script running to maintain the background processes
wait
