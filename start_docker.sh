docker compose up -d
docker compose exec ollama ollama pull qwen2:1.5b
# watch ndocker-compose logs
watch -n 10 docker compose logs -f --tail 10