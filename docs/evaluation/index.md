# Evaluating the AI search
- The challenge with evaluation in this case was the lack of labels. To solve that, we created a simple streamlit app that let us label datasets according to a few tags. 
- The evaluation pipeline runs the entire RAG + Query LLM pipeline on the subset of labelled data. The RAG does not have access to the entire OpenML database but just the subset that was labelled.

## Manual labelling
### Streamlit labelling app
- Refer to [labelling app](./labelling_tool.md) for more information.

### Merging labels
- Since there were multiple people who labelled the datasets, it was useful to have a script that would merge them to create a single dataframe. 
- The labels were generated per person using the labelling app and then merged into a single consistent dataframe using this script.
- Refer to [merging labels](./merging_labels.md) for more information.

### Consistency evaluation
- Since multiple people labelled the same dataset differently, Kohn's Kappa score was used to evaluate the consistency of the labelling. A value of ~4.5 was obtained, which shows moderate consistency. 

## Running the evaluation
- Refer to [run training](./evaluation)  for more information