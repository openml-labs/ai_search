docker compose build
docker compose up -d
docker compose exec ollama ollama run llama3
# watch ndocker-compose logs
# watch -n 10 docker compose logs -f --tail 10