# Data labelling tool

- This tool was used to create a manually labelled dataset for evaluating the performance of the RAG pipeline.
- The tool is a Streamlit app that allows users to search for datasets on OpenML and label them as relevant or irrelevant.


## How to use
- Install `streamlit` using `pip install streamlit`
- Copy the `all_dataset_description.csv` and `LLM Evaluation - Topic Queries - {X}.csv` files to the `data` directory.
- Run the Streamlit app using `streamlit run app.py`
- Results are saved to `data/labels.json`