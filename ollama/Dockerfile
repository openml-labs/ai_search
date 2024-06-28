FROM python:3.10.14

RUN apt update 
RUN apt install -y lshw

RUN mkdir /ollama

# COPY requirements.txt /ollama
COPY get_ollama.sh /ollama

WORKDIR /ollama

RUN curl -fsSL https://ollama.com/install.sh | sh 
RUN chmod +x get_ollama.sh 
EXPOSE 11434

# CMD ["./get_ollama.sh"]

ENTRYPOINT ["ollama", "serve"]
