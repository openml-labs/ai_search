#!/bin/bash
killall ollama
killall streamlit
# Define a file to store the PIDs
PID_FILE="processes.pid"

# Start processes and save their PIDs
cd ollama
./get_ollama.sh &
echo $! > $PID_FILE

structured_query = false
if [ "$structured_query" == true ]; then
    cd ../structured_query
    uvicorn llm_service_structured_query:app --host 0.0.0.0 --port 8082 &
    echo $! > $PID_FILE
else
    cd ../llm_service
    uvicorn llm_service:app --host 0.0.0.0 --port 8081 &
    echo $! > $PID_FILE
fi

cd ../backend
uvicorn backend:app --host 0.0.0.0 --port 8000 &
echo $! >> $PID_FILE

cd ../frontend
streamlit run ui.py &
echo $! >> $PID_FILE

cd ..
# Keep the script running to maintain the background processes
wait
