# Labeling Tool

Simple streamlit app to help label (query, dataset) pairs.

## Installation

The app only requires streamlit:

```
python -m venv venv
source venv/bin/activate
python -m pip install streamlit
```

## Data
The app requires some data to operate, specifically it expects:

 - `data/all_dataset_description.csv`: any file with openml metadata with the following columns (e.g., the one Subha shared):
    - did
    - description
    - NumberOfFeatures
    - NumberOfNumericFeatures
    - NumberOfSymbolicFeatures
    - NumberOfClasses
    - NumberOfMissingValues

 - `data/LLM Evaluation - Topic Queries.csv`: export from our shared google sheet. i.e., a csv with column "topic".

## Using the App

With everything in place, just run `streamlit run app.py`.
If you already have a file with stored label data, it will load it, otherwise you start making a new one.
You can now browse through datasets, and for each dataset you can select which of the queries are relevant.
Changes are not automatically persisted. If the 'save me' button is red, there are local unsaved changes. Click it to persist the changes.

We should be able to merge the different label files later without problem.
