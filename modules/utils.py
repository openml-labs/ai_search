# Description: This file contains the utility functions for the pipeline. Basically, everything except LLM code.

import json
import os
import pickle

# from pqdm.processes import pqdm
from pathlib import Path
from typing import List, Union

import openml
import pandas as pd
import torch
from pqdm.threads import pqdm

# --- GENERAL UTILS ---

def find_device(training: bool = False):
    """
    Description: Find the device to use for the pipeline. If cuda is available, use it. If not, check if MPS is available and use it. If not, use CPU.

    Input: training (bool) : Whether the pipeline is being used for training or not.

    Returns: device (str) : The device to use for the pipeline.
    """
    print("[INFO] Finding device.")
    if training == False:
        return "cpu"
    else:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


def load_config_and_device(config_file: str, training:bool = False):
    """
    Description: Load the config file and find the device to use for the pipeline.

    Input: config_file (str) : The path to the config file.
    training (bool) : Whether the pipeline is being used for training or not.

    Returns: config (dict) : The config dictionary + device (str) : The device to use for the pipeline.
    """
    # Check if the config file exists and load it
    if not os.path.exists(config_file):
        raise Exception("Config file does not exist.")
    with open(config_file, "r") as f:
        config = json.load(f)

    # Find device and set it in the config between cpu and cuda and mps if available
    config["device"] = find_device(training)
    print(f"[INFO] Device found: {config['device']}")
    return config

# -- DOWNLOAD METADATA --

def get_dataset_description(dataset_id) -> openml.datasets.dataset.OpenMLDataset:

    """
    Get the dataset description from OpenML using the dataset id

    Input: dataset_id (int) : The dataset id

    Returns: data (openml.datasets.dataset.OpenMLDataset) : The dataset object from OpenML
    """
    # TODO : Check for objects that do not have qualities being not downloaded properly
    # try:
    data = openml.datasets.get_dataset(
        dataset_id=dataset_id,
        download_data=False,
        download_qualities=True,
        download_features_meta_data=True,
    )

    return data


def get_flow_description(flow_id):
    """
    Get the flow description from OpenML using the flow id

    Input: flow_id (int) : The flow id

    Returns: data (openml.flows.flow.OpenMLFlow) : The flow object from OpenML
    """
    return openml.flows.get_flow(flow_id=flow_id)


# install the package oslo.concurrency to ensure thread safety
def get_all_metadata_from_openml(config) -> Union[List, List]:
    """
    Description: Gets all the metadata from OpenML for the type of data specified in the config.
    If training is set to False, it loads the metadata from the files. If training is set to True, it gets the metadata from OpenML.

    This uses parallel threads (pqdm) and so to ensure thread safety, install the package oslo.concurrency.


    Input: config (dict) : The config dictionary

    Returns: all the data descriptions combined with data ids, data ids, and the raw openml objects in a dataframe.
    """

    save_filename = f"./data/all_{config['type_of_data']}_metadata.pkl"
    # If we are not training, we do not need to recreate the cache and can load the metadata from the files. If the files do not exist, raise an exception.
    # TODO : Check if this behavior is correct, or if data does not exist, send to training pipeline?
    if config["training"] == False:
        # print("[INFO] Training is set to False.")
        # Check if the metadata files exist for all types of data
        if not os.path.exists(save_filename):
            raise Exception(
                "Metadata files do not exist. Please run the training pipeline first."
            )
        print("[INFO] Loading metadata from file.")
        # Load the metadata files for all types of data
        with open(save_filename, "rb") as f:
            all_data_descriptions, data_id, all_datasets = pickle.load(f)
        print("[INFO] Metadata loaded.")
        return all_data_descriptions, data_id, all_datasets

    # If we are training, we need to recreate the cache and get the metadata from OpenML
    if config["training"] == True:
        print("[INFO] Training is set to True.")
        # the id column name is different for dataset and flow, so we need to handle that
        dict_id_column_name = {"dataset": "did", "flow": "id"}
        id_column_name = dict_id_column_name[config["type_of_data"]]

        # Gather all OpenML objects of the type of data
        if config["type_of_data"] == "dataset":
            print("[INFO] Getting dataset metadata.")
            all_objects = openml.datasets.list_datasets(output_format="dataframe")
        elif config["type_of_data"] == "flow":
            print("[INFO] Getting flow metadata.")
            all_objects = openml.flows.list_flows(output_format="dataframe")

        # get cache directory
        # cache_directory = Path(openml.config.get_cache_directory())

        if config["type_of_data"] == "dataset":
            print("[INFO] Checking downloaded files and skipping them.")

            data_id = [
                int(all_objects.iloc[i][id_column_name])
                for i in range(len(all_objects))
            ]

            # Initialize cache before using parallel (following OpenML python API documentation)
            print("[INFO] Initializing cache.")
            get_dataset_description(data_id[0])

            # Get all object metadata using n_jobs parallel threads from openml
            print("[INFO] Getting dataset metadata from OpenML.")
            openml_data_object = pqdm(
                data_id, get_dataset_description, n_jobs=config["data_download_n_jobs"]
            )
        elif config["type_of_data"] == "flow":
            print("[INFO] Checking downloaded files and skipping them.")

            data_id = [
                int(all_objects.iloc[i][id_column_name])
                for i in range(len(all_objects))
            ]
            # Initialize cache before using parallel (following OpenML python API documentation)
            print("[INFO] Initializing cache.")
            get_flow_description(data_id[0])

            # Get all object metadata using n_jobs parallel threads from openml
            print("[INFO] Getting flow metadata from OpenML.")
            openml_data_object = pqdm(
                data_id, get_flow_description, n_jobs=config["data_download_n_jobs"]
            )

        # Save the metadata to a file
        print("[INFO] Saving metadata to file.")
        with open(save_filename, "wb") as f:
            pickle.dump((openml_data_object, data_id, all_objects), f)

        return openml_data_object, data_id, all_objects

# -- COMBINE METADATA INTO A SINGLE DATAFRAME --

def extract_attribute(attribute, attr_name):
    """
    Description: Extract an attribute from the OpenML object.

    Input: attribute (object) : The OpenML object

    Returns: The attribute value if it exists, else an empty string.
    """
    return getattr(attribute, attr_name, "")


def join_attributes(attribute, attr_name):
    """
    Description: Join the attributes of the OpenML object.

    Input: attribute (object) : The OpenML object

    Returns: The joined attributes if they exist, else an empty string.
    example: "column - value, column - value, ..."
    """

    return (
        " ".join([f"{k} : {v}," for k, v in getattr(attribute, attr_name, {}).items()])
        if hasattr(attribute, attr_name)
        else ""
    )


def create_combined_information_df(
    data_id, descriptions, joined_qualities, joined_features
):
    """
    Description: Create a dataframe with the combined information of the OpenML object.

    Input: data_id (int) : The data id, descriptions (list) : The descriptions of the OpenML object, joined_qualities (list) : The joined qualities of the OpenML object, joined_features (list) : The joined features of the OpenML object

    Returns: The dataframe with the combined information of the OpenML object.
    """
    return pd.DataFrame(
        {
            "did": data_id,
            "description": descriptions,
            "qualities": joined_qualities,
            "features": joined_features,
        }
    )


def merge_all_columns_to_string(row):
    """
    Description: Create a single column that has a combined string of all the metadata and the description in the form of "column - value, column - value, ... description"

    Input: row (pd.Series) : The row of the dataframe
    
    Returns: The combined string of all the metadata and the description in the form of "column - value, column - value, ... description"
    """

    return " ".join([f"{col} - {val}," for col, val in zip(row.index, row.values)])


def combine_metadata(all_dataset_metadata, all_data_description_df):
    """
    Description: Combine the descriptions with the metadata table.

    Input: all_dataset_metadata (pd.DataFrame) : The metadata table, 
    all_data_description_df (pd.DataFrame) : The descriptions

    Returns: The combined metadata table.
    """
    # Combine the descriptions with the metadata table
    all_dataset_metadata = pd.merge(
        all_dataset_metadata, all_data_description_df, on="did", how="inner"
    )

    # Create a single column that has a combined string of all the metadata and the description in the form of "column - value, column - value, ... description"

    all_dataset_metadata["Combined_information"] = all_dataset_metadata.apply(
        merge_all_columns_to_string, axis=1
    )
    return all_dataset_metadata


def create_metadata_dataframe(
    openml_data_object, data_id, all_dataset_metadata, config
):
    """
    Description: Creates a dataframe with all the metadata, joined columns with all information for the type of data specified in the config. If training is set to False, the dataframes are loaded from the files. If training is set to True, the dataframes are created and then saved to the files.
    
    Input: openml_data_object (list) : The list of OpenML objects, data_id (list) : The list of data ids, all_dataset_metadata (pd.DataFrame) : The metadata table, config (dict) : The config dictionary

    Returns: The combined metadata dataframe and the metadata table.
    """
    if config["training"] == False:
        # If we are not training, we do not need to recreate the cache and can load the metadata from the files. If the files do not exist, raise an exception.
        try:
            with open(f"./data/all_{config['type_of_data']}_description.csv", "r") as f:
                all_data_description_df = pd.read_csv(f)
            return all_data_description_df, all_dataset_metadata
        except:
            raise Exception(
                "Metadata files do not exist. Please run the training pipeline first."
            )
    if config["training"] == True:
        if config["type_of_data"] == "dataset":
            descriptions = [
                extract_attribute(attr, "description") for attr in openml_data_object
            ]
            joined_qualities = [
                join_attributes(attr, "qualities") for attr in openml_data_object
            ]
            joined_features = [
                join_attributes(attr, "features") for attr in openml_data_object
            ]

            all_data_description_df = create_combined_information_df(
                data_id, descriptions, joined_qualities, joined_features
            )
            all_dataset_metadata = combine_metadata(
                all_dataset_metadata, all_data_description_df
            )

            all_data_description_df.to_csv(
                f"./data/all_{config['type_of_data']}_description.csv", index=False
            )

            return (
                all_dataset_metadata[["did", "name", "Combined_information"]],
                all_dataset_metadata,
            )

        elif config["type_of_data"] == "flow":
            descriptions = [
                extract_attribute(attr, "description") for attr in openml_data_object
            ]
            names = [extract_attribute(attr, "name") for attr in openml_data_object]
            tags = [extract_attribute(attr, "tags") for attr in openml_data_object]

            all_data_description_df = pd.DataFrame(
                {
                    "id": data_id,
                    "description": descriptions,
                    "name": names,
                    "tags": tags,
                }
            )
            # Create a single column that has a combined string of all the metadata and the description in the form of "column - value, column - value, ... description"

            all_data_description_df["Combined_information"] = (
                all_data_description_df.apply(merge_all_columns_to_string, axis=1)
            )

            all_data_description_df.to_csv(
                f"./data/all_{config['type_of_data']}_description.csv", index=False
            )

            return (
                all_data_description_df[["id", "name", "Combined_information"]],
                all_data_description_df,
            )
