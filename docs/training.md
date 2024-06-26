# Training
- While we are not creating a new model, we are using the existing model to create embeddings. The name might be misleading but this was chosen as an attempt to keep the naming consistent with other codebases.
- (Perhaps we might fine tune the model in the future)
- The training script is present in `training.py`. Running this script will take care of everything.

## What does the training script do?
- Load the config file and set the necessary variables
- If `testing_flag` is set to True, the script will use a subset of the data for quick debugging
  - testing_flag is set to True
  - persist_dir is set to ./data/chroma_db_testing
  - test_subset_2000 is set to True
  - data_dir is set to ./data/testing_data/
- If `testing_flag` is set to False, the script will use the entire dataset
- For all datasets in the OpenML dataset list:
  - Download the dataset
  - Create the vector dataset with computed embeddings
- Create a vectordb retriever 
- Run some test queries