import os
import pickle
from typing import List, Union

import openml
import pandas as pd
from pqdm.threads import pqdm


def get_dataset_description(dataset_name) -> openml.datasets.dataset.OpenMLDataset:
    #TODO : Check for objects that do not have qualities being not downloaded properly
    data = openml.datasets.get_dataset(
        dataset_name,
        download_data=False,
        download_qualities=True,
        download_features_meta_data=True,
    )
       
    return data

# install the package oslo.concurrency to ensure thread safety
def get_all_dataset_metadata_from_openml(
    save_filename="all_dataset_metadata.pkl", recreate_cache=False, n_jobs=10
) -> Union[List, List]:
    
    # if the file already exists, load it else get the metadata from openml
    if os.path.exists(save_filename) and recreate_cache == False:
        with open(save_filename, "rb") as f:
            all_data_descriptions, data_id, all_datasets = pickle.load(f)
        return all_data_descriptions, data_id, all_datasets
    elif recreate_cache == True:
        # Gather all OpenML datasets
        all_datasets = openml.datasets.list_datasets(output_format="dataframe")

        # List dataset 'did' to be used as an identifier
        data_id = [all_datasets.iloc[i]["did"] for i in range(len(all_datasets))]

        dataset_names = all_datasets["name"].tolist()  # get a list of all dataset names

        print("Recreating the cache")
        # Get all dataset metadata using n_jobs parallel threads from openml
        openml_data_object = pqdm(dataset_names, get_dataset_description, n_jobs=n_jobs)

        # Save the metadata to a file
        with open(save_filename, "wb") as f:
            pickle.dump((openml_data_object, data_id, all_datasets), f)

        return openml_data_object, data_id, all_datasets

def create_metadata_dataframe(openml_data_object, data_id, all_dataset_metadata):
    # Get the descriptions of the dataset
   
    descriptions = [
        desc.description if hasattr(desc, 'description') else "" 
    for desc in openml_data_object
    ]

    joined_qualities = [
        " ".join([f"{k} : {v}," for k, v in desc.qualities.items()]) if hasattr(desc, 'qualities') else "" 
        for desc in openml_data_object
    ]

    joined_features = [
        " ".join([f"{k} : {v}," for k, v in desc.features.items()]) if hasattr(desc, 'features') else "" 
        for desc in openml_data_object
    ]
  
    # ignore different lengths of the lists
    all_data_description_df = pd.DataFrame({'did': data_id, 'description': descriptions, 'qualities': joined_qualities, 'features': joined_features})

    # Combine the descriptions with the metadata table
    all_dataset_metadata = pd.merge(all_dataset_metadata,all_data_description_df, on='did', how='inner')

    # Create a single column that has a combined string of all the metadata and the description in the form of "column - value, column - value, ... description"
    all_dataset_metadata['Combined_information'] = all_dataset_metadata.apply(lambda row: ' '.join([f"{col} - {val}," for col, val in zip(all_dataset_metadata.columns, row.values)]), axis=1)

    # We do not need all_description_metadata anymore, but keeping it here as it might be useful for composing functions later on in the pipeline
    return all_dataset_metadata[['did', 'name','Combined_information']], all_dataset_metadata

def clean_metadata_dataframe(metadata_df) -> pd.DataFrame:
    """
    Cleans the metadata dataframe by removing rows with empty descriptions. (Other cleaning steps can be added here as well)

    Input : the metadata dataframe generated from the create_metadata_dataframe function
    Output : the cleaned metadata dataframe
    """
    # remove rows with empty descriptions
    metadata_df = metadata_df[metadata_df["did"].notna()]
    # remove rows with empty 
    return metadata_df
