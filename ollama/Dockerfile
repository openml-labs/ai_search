FROM python:3.10.14

RUN apt update 
RUN apt install -y lshw

RUN mkdir /ollama

# COPY requirements.txt /ollama
COPY get_ollama.sh /ollama
#COPY base_logger.py /ollama
WORKDIR /ollama

RUN curl -fsSL https://ollama.com/install.sh | sh 
RUN chmod +x get_ollama.sh 
#RUN python3 base_logger.py
EXPOSE 11434

ENTRYPOINT ["ollama", "serve"]