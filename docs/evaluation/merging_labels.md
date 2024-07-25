# Merging labels
- Takes multiple JSON files as input and merges them into a single csv file with columns `Topics,Dataset IDs`

## How to use
- Place all the label.json files in the folder `/tools/data/all_labels`
- Run `python merge_labels.py` from the `tools` directory.
- The results would be present in `/data/evaluation/merged_labels.csv`