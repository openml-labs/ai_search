FROM python:3.10.14
RUN mkdir /fastapi
COPY ./backend /fastapi
WORKDIR /fastapi

RUN pip install -r requirements.txt

#COPY . /fastapi
# COPY modules /fastapi
# COPY *.py /fastapi
# COPY *.json /fastapi
# COPY *.html /fastapi

# RUN ollama pull qwen2:1.5b

EXPOSE 8000

ENTRYPOINT ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]