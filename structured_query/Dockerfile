FROM python:3.10.14
RUN mkdir /structured_query
COPY . /structured_query
WORKDIR /structured_query

RUN pip install -r requirements.txt

EXPOSE 8081

ENTRYPOINT ["uvicorn", "llm_service_structured_query:app", "--host", "0.0.0.0", "--port", "8081"]