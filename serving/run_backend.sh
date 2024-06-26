#!/usr/bin/env bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload --log-level debug&
# echo "waiting for uvicorn to be active..."
# while [ "$(curl -s localhost:8000)" == "" ]; do
#   sleep 1
# done
# echo "uvicorn is active"
# Check if uvicorn is active from running process
while [ "$(ps aux | grep uvicorn | grep -v grep)" == "" ]; do
  sleep 1
done