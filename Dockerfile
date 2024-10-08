# Dockerfile

# Use a base image with Python
FROM python:3.10.14

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y jq && \
    apt-get clean

# Copy the Poetry lock files and install Poetry
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev
# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN ollama serve&
# RUN while [ "$(ollama list | grep 'NAME')" == "" ]; do sleep 1 done
# RUN until ollama list | grep -q 'NAME'; do sleep 1; done
# RUN timeout 120 bash -c 'until ollama list | grep -q "NAME"; do sleep 1; done'
RUN ollama serve & sleep 5 && ollama run llama3



# RUN ollama pull llama3

# Copy the application code
COPY . .

# Expose the necessary ports
EXPOSE 8000 8081 8083 8050 11434 8501

# Start the application
CMD ["bash", "start_docker_local.sh"]
