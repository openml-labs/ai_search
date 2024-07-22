# %%
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

# %%

# Define paths
labels_path = Path("./data/all_labels/")
eval_path = Path("../data/evaluation/")

# Get all files in the labels directory
all_files = list(labels_path.glob("*"))


def merge_labels() -> dict[Any, list]:
    """
    Description : Merge labels from multiple JSON label files into a single dictionary.
    """
    # Read all files and merge them into a single dictionary
    merged_labels = defaultdict(set)
    for file in all_files:
        with open(file, "r") as f:
            try:
                data = json.load(f)
                for key, values in data.items():
                    merged_labels[key].update(values)
            except json.JSONDecodeError:
                print(f"Error reading {file}")
    # Remove empty lists
    merged_labels = {k: list(v) for k, v in merged_labels.items() if v}

    # Reverse the dictionary so we have topic -> [dataset_ids]
    reversed_labels = defaultdict(set)
    for key, values in merged_labels.items():
        for value in values:
            reversed_labels[value].add(key)

    # Convert sets to lists for each value
    return {k: list(v) for k, v in reversed_labels.items()}


reversed_labels = merge_labels()

# Write to CSV
with open(eval_path / "merged_labels.csv", "w") as f:
    f.write("Topics,Dataset IDs\n")
    for key, values in reversed_labels.items():
        # f.write(f'{key.strip()},"{",'.join(values)}'\n')
        f.write(f'{key.strip()},"{",".join(values)}"\n')

# %%
