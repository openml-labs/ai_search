PID_FILE="processes.pid"
# Start processes and save their PIDs
cd ollama
# ./get_ollama.sh &
# echo $! > $PID_FILE
# read pid from PID_FILE and kill the process
kill -9 $(cat $PID_FILE)

cd ../backend
# uvicorn backend:app --host 0.0.0.0 --port 8000 &
kill -9 $(cat $PID_FILE)

cd ../llm_service
kill -9 $(cat $PID_FILE)

cd ../structured_query
kill -9 $(cat $PID_FILE)

cd ../frontend
# streamlit run ui.py &
kill -9 $(cat $PID_FILE)

cd ..

killall ollama
killall streamlit