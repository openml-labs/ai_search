# Docker container
- Still a WIP
- Run `docker compose build --progress=plain`

## Potential Errors
- If you get a memory error you can run `docker system prune`. Please be careful with this command as it will remove all stopped containers, all dangling images, and all unused networks. So ensure you have no important data in any of the containers before running this command.
- On docker desktop for Mac, increase memory limits to as much as your system can handle.