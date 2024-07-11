import json
import os

import pandas as pd


def read_query_csv_and_convert_to_json(file_path: str):
    df = pd.read_csv(file_path)

    dict_ids = {}
    for i, row in df.iterrows():
        row = row.values
        if not "Found" in row[1]:
            ids_in_row = [int(x.strip()) for x in row[1].split(",")]
            for id in ids_in_row:
                if id in dict_ids:
                    dict_ids[id].append(row[0])
                else:
                    dict_ids[id] = [row[0]]
    return dict_ids


def merge_dict_and_old_json_and_save(
    dict1: dict,
    file_path_2: str = "data/labels.json",
    file_path_save: str = "data/merged_labels.json",
):

    # json_dict = json.dumps(dict1)
    if os.path.exists(file_path_2):
        with open(file_path_2, "r") as f:
            labels = json.load(f)

        for key in labels:
            if key in dict1:
                dict1[key] = dict1[key].extend(labels[key])

    # make evaluation dir
    os.makedirs(os.path.dirname(file_path_save), exist_ok=True)

    with open(file_path_save, "w") as f:
        json.dump(dict1, f)


file_paths = ["data/LLM Evaluation - Topic Queries.csv"]
for file in file_paths:
    dict_ids = read_query_csv_and_convert_to_json(file)
    merge_dict_and_old_json_and_save(
        dict_ids, file_path_save="../data/evaluation/merged_labels.json"
    )
