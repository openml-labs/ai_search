FROM python:3.10.14
RUN mkdir /llmservice
COPY requirements.txt fastapi/requirements.txt
WORKDIR /fastapi

RUN pip install -r requirements.txt

COPY . /llmservice

EXPOSE 8001

ENTRYPOINT ["uvicorn", "llmservice:app", "--host", "0.0.0.0", "--port", "8001"]