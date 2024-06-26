FROM python:3.10.14
# FROM ollama/ollama
COPY requirements.txt app/requirements.txt
COPY get_ollama.sh app/get_ollama.sh
# ADD serving app
WORKDIR /app
# RUN chmod +x get_ollama.sh
RUN apt update 
RUN apt install -y lshw
# RUN apt install -y python3.10 python3-pip
# RUN bash get_ollama.sh
RUN curl -fsSL https://ollama.com/install.sh | sh 
RUN chmod +x get_ollama.sh \
    && ./get_ollama.sh

# RUN ollama pull qwen2:1.5b
# EXPOSE 11434
RUN pip install -r requirements.txt

ADD data /app/data
ADD modules /app/modules
ADD tests /app/tests
COPY *.py /app/
COPY *.json /app/
EXPOSE 8000
EXPOSE 8501

# COPY run_backend.sh run_backend.sh
# RUN bash -c "streamlit run main.py&"
# RUN bash -c "uvicorn backend:app --host 0.0.0.0 --port 8000 --reload&"
# RUN chmod +x run_backend.sh \
    # && ./run_backend.sh 
# RUN streamlit run main.py&
# ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD ["python", "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000" ,"--reload" ]
# RUN uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
# RUN streamlit run main.py
# ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD ["uvicorn", "backend:app", "--port", "8000", "--reload"]
# CMD ["fastapi", "run", "backend.py", "--port", "8000"]

# RUN apt-get install -y apache2
# RUN apt-get install -y mongod #pretend this exists
RUN apt install -y supervisor # Installing supervisord

ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf 

EXPOSE 80

RUN python3 -u backend.py

ENTRYPOINT ["/usr/bin/supervisord"]
