import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# # Get all category names
# topic_path = "data/LLM Evaluation - Topic Queries.csv"
# df = pd.read_csv(topic_path)
# CLS = df["Topic"].unique().tolist()
# CLS = [c.strip().lower() for c in CLS]
CLS = set()

labels_1 = {}
labels_2 = {}

label_files = [
    "data/all_labels/wenhan_labels.json",
    "data/all_labels/pieter_labels.json",
    "data/all_labels/jiaxu_labels.json",
    "data/all_labels/taniya_labels.json",
    "data/all_labels/subhaditya_labels.json",
]
# label_files = ["data/merged_labels.json", "data/llm_labels.json"]
for i in range(len(label_files)):
    for j in range(i + 1, len(label_files)):
        # print(f"{label_files[i]},{label_files[j]}")
        with open(label_files[i]) as f1, open(label_files[j]) as f2:
            json1 = json.load(f1)
            json2 = json.load(f2)
        intersec = set(json1.keys()).intersection(set(json2.keys()))
        for key in intersec:
            if key in labels_1 and key in labels_2:
                continue
            assert (
                key not in labels_1 and key not in labels_2
            ), f"{key}: {label_files[i]},{label_files[j]}\n{json1[key]}\n{json2[key]}\n{labels_1[key]}\n{labels_2[key]}"
            labels_1[key] = json1[key]
            labels_2[key] = json2[key]
            CLS.update(json1[key])
            CLS.update(json2[key])
print(labels_1)
print(labels_2)
CLS = [c.strip().lower() for c in CLS]


# Create label matrix
def create_label_matrix(labels, CLS):
    ids = sorted(labels.keys())
    label_matrix = np.zeros((len(ids), len(CLS)))
    for i, id in enumerate(ids):
        for label in labels[id]:
            if label.strip().lower() in CLS:
                label_matrix[i, CLS.index(label.strip().lower())] = 1
    return ids, label_matrix


# Generate label matrices for both files
ids_1, label_matrix_1 = create_label_matrix(labels_1, list(CLS))
ids_2, label_matrix_2 = create_label_matrix(labels_2, list(CLS))

# Ensure IDs in both matrices match
assert ids_1 == ids_2, "IDs in both files do not match"

# Calculate Cohen's Kappa for each category
kappas = []
for i in range(label_matrix_1.shape[1]):
    kappa = cohen_kappa_score(label_matrix_1[:, i], label_matrix_2[:, i], labels=[0, 1])
    kappas.append(kappa)

# Calculate average Cohen's Kappa
average_kappa = np.mean(kappas)
print(f"Cohen's Kappa: {average_kappa}")
