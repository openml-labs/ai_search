import json
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd

# Get all category names
topic_path = "data/LLM Evaluation - Topic Queries.csv"
df = pd.read_csv(topic_path)
CLS = df['Topic'].unique().tolist()
CLS = [c.strip().lower() for c in CLS]

# Read the JSON files
with open('data/merged_labels_1.json', 'r') as f:
    labels_1 = json.load(f)
with open('data/merged_labels_2.json', 'r') as f:
    labels_2 = json.load(f)


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
ids_1, label_matrix_1 = create_label_matrix(labels_1, CLS)
ids_2, label_matrix_2 = create_label_matrix(labels_2, CLS)

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
