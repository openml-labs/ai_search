FROM python:3.10.14

RUN mkdir /streamlit
RUN mkdir /backend
RUN mkdir /data

COPY ./data/all_dataset_description.csv /data
COPY ./data/all_flow_description.csv /data
COPY ./frontend /streamlit
COPY ./backend/config.json /backend
WORKDIR /streamlit

RUN pip install -r requirements.txt
#COPY . /streamlit

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "ui.py"]

