from __future__ import annotations
import os
import pickle
# from pqdm.processes import pqdm
from typing import Sequence, Tuple, Union

import openml
import pandas as pd
from pqdm.threads import pqdm

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


def get_flow_description(flow_id: int) -> openml.flows.flow.OpenMLFlow:
    """
    Get the flow description from OpenML using the flow id

    Input: flow_id (int) : The flow id

    Returns: data (openml.flows.flow.OpenMLFlow) : The flow object from OpenML
    """
    return openml.flows.get_flow(flow_id=flow_id)


def load_metadata_from_file(save_filename: str) -> Tuple[pd.DataFrame, Sequence[int], pd.DataFrame]:
    """
    Load metadata from a file.
    """
    with open(save_filename, "rb") as f:
        return pickle.load(f)


def save_metadata_to_file(data, save_filename: str):
    """
    Save metadata to a file.
    """
    with open(save_filename, "wb") as f:
        pickle.dump(data, f)


def initialize_cache(type_of_data: str, data_id: Sequence[int]) -> None:
    """
    Initialize cache for the OpenML objects.
    """
    if type_of_data == "dataset":
        get_dataset_description(data_id[0])
    elif type_of_data == "flow":
        get_flow_description(data_id[0])


def get_metadata_from_openml(config, data_id: Sequence[int]):
    """
    Get metadata from OpenML using parallel processing.
    """
    if config["type_of_data"] == "dataset":
        return pqdm(
            data_id, get_dataset_description, n_jobs=config["data_download_n_jobs"]
        )
    elif config["type_of_data"] == "flow":
        return pqdm(
            data_id, get_flow_description, n_jobs=config["data_download_n_jobs"]
        )


def get_openml_objects(type_of_data: str):
    """
    Get OpenML objects based on the type of data.
    """
    if type_of_data == "dataset":
        return openml.datasets.list_datasets(output_format="dataframe")
    elif type_of_data == "flow":
        all_objects = openml.flows.list_flows(output_format="dataframe")
        return all_objects.rename(columns={"id": "did"})
    else:
        raise ValueError("Invalid type_of_data specified")


# install the package oslo.concurrency to ensure thread safety
# def get_all_metadata_from_openml(config) -> Tuple[pd.DataFrame, Sequence[int], pd.DataFrame]:
def get_all_metadata_from_openml(config: dict) -> Tuple[pd.DataFrame, Sequence[int], pd.DataFrame] | None:
    """
    Description: Gets all the metadata from OpenML for the type of data specified in the config.
    If training is set to False, it loads the metadata from the files. If training is set to True, it gets the metadata from OpenML.

    This uses parallel threads (pqdm) and so to ensure thread safety, install the package oslo.concurrency.


    Input: config (dict) : The config dictionary

    Returns: all the data descriptions combined with data ids, data ids, and the raw openml objects in a dataframe.
    """

    # save_filename = f"./data/all_{config['type_of_data']}_metadata.pkl"
    # use os.path.join to ensure compatibility with different operating systems
    save_filename = os.path.join(
        config["data_dir"], f"all_{config['type_of_data']}_metadata.pkl"
    )
    # If we are not training, we do not need to recreate the cache and can load the metadata from the files. If the files do not exist, raise an exception.
    # TODO : Check if this behavior is correct, or if data does not exist, send to training pipeline?
    if config["training"] == False or config["ignore_downloading_data"] == True:
        # print("[INFO] Training is set to False.")
        # Check if the metadata files exist for all types of data
        if not os.path.exists(save_filename):
            raise Exception(
                "Metadata files do not exist. Please run the training pipeline first."
            )
        print("[INFO] Loading metadata from file.")
        # Load the metadata files for all types of data
        return load_metadata_from_file(save_filename)

    # If we are training, we need to recreate the cache and get the metadata from OpenML
    if config["training"] == True:
        print("[INFO] Training is set to True.")
        # Gather all OpenML objects of the type of data
        all_objects = get_openml_objects(config["type_of_data"])

        # subset the data for testing
        if config["test_subset_2000"] == True:
            print("[INFO] Subsetting the data to 2000 rows.")
            all_objects = all_objects[:2000]

        data_id = [int(all_objects.iloc[i]["did"]) for i in range(len(all_objects))]

        print("[INFO] Initializing cache.")
        initialize_cache(config["type_of_data"], data_id)

        print(f"[INFO] Getting {config['type_of_data']} metadata from OpenML.")
        openml_data_object = get_metadata_from_openml(config, data_id)

        print("[INFO] Saving metadata to file.")
        save_metadata_to_file((openml_data_object, data_id, all_objects), save_filename)

        return openml_data_object, data_id, all_objects


# -- COMBINE METADATA INTO A SINGLE DATAFRAME --


def extract_attribute(attribute: object, attr_name: str) -> str:
    """
    Description: Extract an attribute from the OpenML object.

    Input: attribute (object) : The OpenML object

    Returns: The attribute value if it exists, else an empty string.
    """
    return getattr(attribute, attr_name, "")


def join_attributes(attribute: object, attr_name: str) -> str:
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
    # data_id, descriptions, joined_qualities, joined_features
    data_id: int| Sequence[int], descriptions: Sequence[str], joined_qualities: Sequence[str], joined_features: Sequence[str]
) -> pd.DataFrame:
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


def merge_all_columns_to_string(row: pd.Series) -> str:
    """
    Description: Create a single column that has a combined string of all the metadata and the description in the form of "column - value, column - value, ... description"

    Input: row (pd.Series) : The row of the dataframe

    Returns: The combined string of all the metadata and the description in the form of "column - value, column - value, ... description"
    """

    return " ".join([f"{col} - {val}," for col, val in zip(row.index, row.values)])


# def combine_metadata(all_dataset_metadata, all_data_description_df):
def combine_metadata(all_dataset_metadata: pd.DataFrame, all_data_description_df: pd.DataFrame) -> pd.DataFrame:
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


def load_metadata(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise Exception(
            "Metadata files do not exist. Please run the training pipeline first."
        )


def process_dataset_metadata(
    openml_data_object: Sequence[openml.datasets.dataset.OpenMLDataset], data_id: Sequence[int], all_dataset_metadata: pd.DataFrame, file_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Description: Process the dataset metadata.
    
    Input: openml_data_object (list) : The list of OpenML objects, data_id (list) : The list of data ids, all_dataset_metadata (pd.DataFrame) : The metadata table, file_path (str) : The file path
    
    Returns: The combined metadata dataframe and the updated metadata table.
    """
    descriptions = [
        extract_attribute(attr, "description") for attr in openml_data_object
    ]
    joined_qualities = [
        join_attributes(attr, "qualities") for attr in openml_data_object
    ]
    joined_features = [join_attributes(attr, "features") for attr in openml_data_object]

    all_data_description_df = create_combined_information_df(
        data_id, descriptions, joined_qualities, joined_features
    )
    all_dataset_metadata = combine_metadata(
        all_dataset_metadata, all_data_description_df
    )

    all_dataset_metadata.to_csv(file_path)

    return (
        all_dataset_metadata[["did", "name", "Combined_information"]],
        all_dataset_metadata,
    )


# def process_flow_metadata(openml_data_object, data_id, file_path):
def process_flow_metadata(openml_data_object: Sequence[openml.flows.flow.OpenMLFlow], data_id: Sequence[int], file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Description: Process the flow metadata.
    
    Input: openml_data_object (list) : The list of OpenML objects, data_id (list) : The list of data ids, file_path (str) : The file path
    
    Returns: The combined metadata dataframe and the updated metadata table.
    """
    descriptions = [
        extract_attribute(attr, "description") for attr in openml_data_object
    ]
    names = [extract_attribute(attr, "name") for attr in openml_data_object]
    tags = [extract_attribute(attr, "tags") for attr in openml_data_object]

    all_data_description_df = pd.DataFrame(
        {
            "did": data_id,
            "description": descriptions,
            "name": names,
            "tags": tags,
        }
    )

    all_data_description_df["Combined_information"] = all_data_description_df.apply(
        merge_all_columns_to_string, axis=1
    )
    all_data_description_df.to_csv(file_path)

    return (
        all_data_description_df[["did", "name", "Combined_information"]],
        all_data_description_df,
    )


def create_metadata_dataframe(
    # openml_data_object, data_id, all_dataset_metadata, config
    openml_data_object: Sequence[Union[openml.datasets.dataset.OpenMLDataset, openml.flows.flow.OpenMLFlow]], data_id: Sequence[int], all_dataset_metadata: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a dataframe with all the metadata, joined columns with all information
    for the type of data specified in the config. If training is set to False,
    the dataframes are loaded from the files. If training is set to True, the
    dataframes are created and then saved to the files.

    Args:
        openml_data_object (list): The list of OpenML objects.
        data_id (list): The list of data ids.
        all_dataset_metadata (pd.DataFrame): The metadata table.
        config (dict): The config dictionary.

    Returns:
        pd.DataFrame: The combined metadata dataframe.
        pd.DataFrame: The updated metadata table.
    """
    # use os.path.join to ensure compatibility with different operating systems
    file_path = os.path.join(
        config["data_dir"], f"all_{config['type_of_data']}_description.csv"
    )

    if not config["training"]:
        return load_metadata(file_path), all_dataset_metadata

    if config["type_of_data"] == "dataset":
        return process_dataset_metadata(
            openml_data_object, data_id, all_dataset_metadata, file_path
        )

    if config["type_of_data"] == "flow":
        return process_flow_metadata(openml_data_object, data_id, file_path)

    raise ValueError(f"Unsupported type_of_data: {config['type_of_data']}")
