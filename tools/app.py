"""
Small tool for labeling data.

pip install streamlit
streamlit run app.py

Expects the metadata csv and the topic csv in the `data` directory.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

LOADED_DATA = "label_data"
UNSAVED_DATA = "unsaved_data"

SAVE_FILE = Path("data/labels.json")


@st.cache_data
def load_csv(file: Path, *args, **kwargs) -> pd.DataFrame:
    return pd.read_csv(file, *args, **kwargs)


def save_label_data() -> None:
    with open(SAVE_FILE, "w") as fh:
        json.dump(st.session_state[UNSAVED_DATA], fh)
    st.session_state[LOADED_DATA] = load_label_data()


def load_label_data() -> dict[int, list[str]]:
    if not SAVE_FILE.exists():
        return defaultdict(list)
    with open(SAVE_FILE, "r") as fh:
        loaded_data = json.load(fh)
    # JSON keys need to be strings, so we need to deserialize
    loaded_data = {int(key): data for key, data in loaded_data.items()}
    return defaultdict(list, loaded_data)


if LOADED_DATA not in st.session_state:
    st.session_state[LOADED_DATA] = load_label_data()
    st.session_state[UNSAVED_DATA] = load_label_data()


metadata = load_csv("data/all_dataset_description.csv", index_col=0)
metadata.set_index(keys="did", inplace=True)

did_left, save_right = st.columns(spec=[0.7, 0.3])
with did_left:
    did = st.number_input(
        label="dataset_id",
        min_value=1,
        max_value=60_000,
        value=2,
    )

if did not in metadata.index:
    # TODO: We should just check beforehand all our matches have metadata.
    st.write(f"No metadata for dataset with id {did}.")
    st.stop()

dataset = metadata.loc[did]

st.write(f"# {dataset['name']}")
with st.expander(label="description", expanded=True):
    st.write(dataset["description"])


FEATURE_PATTERN = re.compile(r"\d+ : \[\d+ - ([^ ]+) \((\w+)\)]")
with st.expander(label="features", expanded=True):
    features = [
        match.groups() for match in FEATURE_PATTERN.finditer(dataset["features"])
    ]
    st.write(pd.DataFrame(features, columns=["name", "type"]))


with st.expander(label="meta-features", expanded=True):
    meta_left, meta_right = st.columns(spec=2)
    feature_columns = [
        "NumberOfFeatures",
        "NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures",
    ]
    other_columns = ["NumberOfClasses", "NumberOfInstances", "NumberOfMissingValues"]

    with meta_left:
        st.write(dataset[other_columns])

    with meta_right:
        st.write(dataset[feature_columns])


topic_queries = load_csv("data/LLM Evaluation - Topic Queries.csv")
topics = topic_queries["Topic"].tolist()
st.write("## For each query, is this dataset relevant?")


def update_relevancy(var, topic):
    if st.session_state[var] and topic not in st.session_state[UNSAVED_DATA][did]:
        st.session_state[UNSAVED_DATA][did].append(topic)
    if not st.session_state[var] and topic in st.session_state[UNSAVED_DATA][did]:
        st.session_state[UNSAVED_DATA][did].remove(topic)


with st.container(height=400):
    for i, topic in enumerate(topics):

        def update_this_relevancy(var_, topic_):
            """Helper function to bind the variables to scope."""
            return lambda: update_relevancy(f"q{var_}", topic_)

        # We need to use the on_change callbacks instead of the regular
        # return values, because state needs to be up-to-date *before*
        # the next render pass.
        st.checkbox(
            label=topic,
            value=(topic in st.session_state[UNSAVED_DATA][did]),
            key=f"q{i}",
            on_change=update_this_relevancy(i, topic),
        )


unsaved_changes = st.session_state[UNSAVED_DATA] != st.session_state[LOADED_DATA]
button_type = "primary" if unsaved_changes else "secondary"
with save_right:
    st.button(
        "Save me!", use_container_width=True, type=button_type, on_click=save_label_data
    )
