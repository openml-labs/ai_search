#!/usr/bin/env bash
# curl -fsSL https://ollama.com/install.sh | sh 
ollama serve&
echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done

ollama run llama3
tail -f /dev/null
