FROM python:3.10.14
RUN mkdir /documentation_query
COPY . /documentation_query
WORKDIR /documentation_query

RUN pip install -r requirements.txt


EXPOSE 8083

ENTRYPOINT ["uvicorn", "documentation_query:app", "--host", "0.0.0.0", "--port", "8083"]