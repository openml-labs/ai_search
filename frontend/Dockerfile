FROM python:3.10.14

RUN mkdir /streamlit
RUN mkdir /backend
RUN mkdir /structured_query
RUN mkdir /documentation_query
# RUN mkdir /data

# COPY ./data/all_dataset_description.csv /data
# COPY ./data/all_flow_description.csv /data
COPY ./frontend /streamlit
COPY ./backend /backend
COPY ./structured_query /structured_query
COPY ./documentation_bot /documentation_query
WORKDIR /streamlit

RUN pip install -r requirements.txt
#COPY . /streamlit

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "ui.py"]

