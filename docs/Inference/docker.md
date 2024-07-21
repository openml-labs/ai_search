# Docker container
## Building
- Run `docker compose build --progress=plain`

## Running
- Run `./start_docker.sh`
- This uses the docker compose file to run the docker process in the background.
- The required LLM model is also pulled from the docker hub and the container is started.

## Stopping
- Run `./stop_docker.sh`

## Potential Errors
- Permission errors : Run `chmod +x *.sh`
- If you get a memory error you can run `docker system prune`. Please be careful with this command as it will remove all stopped containers, all dangling images, and all unused networks. So ensure you have no important data in any of the containers before running this command.
- On docker desktop for Mac, increase memory limits to as much as your system can handle.