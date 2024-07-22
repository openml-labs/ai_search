from __future__ import annotations

import os
import pickle

# from pqdm.processes import pqdm
from typing import Sequence, Tuple, Union

import chromadb
import openml
import pandas as pd
from pqdm.threads import pqdm

from .vector_store_utils import *

# -- DOWNLOAD METADATA --


class OpenMLObjectHandler:
    """
    Description: The base class for handling OpenML objects.
    """

    def __init__(self, config):
        self.config = config

    def get_description(self, data_id: int):
        """
        Description: Get the description of the OpenML object.


        """
        raise NotImplementedError

    def get_openml_objects(self):
        """
        Description: Get the OpenML objects.


        """
        raise NotImplementedError

    def initialize_cache(self, data_id: Sequence[int]) -> None:
        """
        Description: Initialize the cache for the OpenML objects.


        """
        self.get_description(data_id[0])

    def get_metadata(self, data_id: Sequence[int]):
        """
        Description: Get metadata from OpenML using parallel processing.


        """
        return pqdm(
            data_id, self.get_description, n_jobs=self.config["data_download_n_jobs"]
        )

    def process_metadata(
        self,
        openml_data_object,
        data_id: Sequence[int],
        all_dataset_metadata: pd.DataFrame,
        file_path: str,
        subset_ids=None,
    ):
        """
        Description: Process the metadata.


        """
        raise NotImplementedError

    def load_metadata(self, file_path: str):
        """
        Description: Load metadata from a file.


        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise Exception(
                "Metadata files do not exist. Please run the training pipeline first."
            )

    def extract_attribute(self, attribute: object, attr_name: str) -> str:
        """
        Description: Extract an attribute from the OpenML object.
        """
        return getattr(attribute, attr_name, "")

    def join_attributes(self, attribute: object, attr_name: str) -> str:
        """
        Description: Join the attributes of the OpenML object.
        """
        return (
            " ".join(
                [f"{k} : {v}," for k, v in getattr(attribute, attr_name, {}).items()]
            )
            if hasattr(attribute, attr_name)
            else ""
        )

    def create_combined_information_df_for_datasets(
        self,
        data_id: int | Sequence[int],
        descriptions: Sequence[str],
        joined_qualities: Sequence[str],
        joined_features: Sequence[str],
    ) -> pd.DataFrame:
        """
        Description: Create a dataframe with the combined information of the OpenML object.
        """
        return pd.DataFrame(
            {
                "did": data_id,
                "description": descriptions,
                "qualities": joined_qualities,
                "features": joined_features,
            }
        )

    def merge_all_columns_to_string(self, row: pd.Series) -> str:
        """
        Description: Create a single column that has a combined string of all the metadata and the description in the form of "column - value, column - value, ... description"
        """
        return " ".join([f"{col} - {val}," for col, val in zip(row.index, row.values)])

    def combine_metadata(
        self, all_dataset_metadata: pd.DataFrame, all_data_description_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Description: Combine the descriptions with the metadata table.
        """
        all_dataset_metadata = pd.merge(
            all_dataset_metadata, all_data_description_df, on="did", how="inner"
        )
        all_dataset_metadata["Combined_information"] = all_dataset_metadata.apply(
            self.merge_all_columns_to_string, axis=1
        )
        return all_dataset_metadata

    def subset_metadata(
        self, subset_ids: Sequence[int] | None, all_dataset_metadata: pd.DataFrame
    ):
        if subset_ids is not None:
            subset_ids = [int(x) for x in subset_ids]
            all_dataset_metadata = all_dataset_metadata[
                all_dataset_metadata["did"].isin(subset_ids)
            ]
        return all_dataset_metadata


class OpenMLDatasetHandler(OpenMLObjectHandler):
    """
    Description: The class for handling OpenML dataset objects.
    """

    def get_description(self, data_id: int):
        return openml.datasets.get_dataset(
            dataset_id=data_id,
            download_data=False,
            download_qualities=True,
            download_features_meta_data=True,
        )

    def get_openml_objects(self):
        return openml.datasets.list_datasets(output_format="dataframe")

    def process_metadata(
        self,
        openml_data_object: Sequence[openml.datasets.dataset.OpenMLDataset],
        data_id: Sequence[int],
        all_dataset_metadata: pd.DataFrame,
        file_path: str,
        subset_ids=None,
    ):
        descriptions = [
            self.extract_attribute(attr, "description") for attr in openml_data_object
        ]
        joined_qualities = [
            self.join_attributes(attr, "qualities") for attr in openml_data_object
        ]
        joined_features = [
            self.join_attributes(attr, "features") for attr in openml_data_object
        ]

        all_data_description_df = self.create_combined_information_df_for_datasets(
            data_id, descriptions, joined_qualities, joined_features
        )
        all_dataset_metadata = self.combine_metadata(
            all_dataset_metadata, all_data_description_df
        )

        # subset the metadata if subset_ids is not None
        all_dataset_metadata = self.subset_metadata(subset_ids, all_dataset_metadata)

        all_dataset_metadata.to_csv(file_path)

        if self.config.get("use_chroma_for_saving_metadata") == True:
            client = chromadb.PersistentClient(
                path=self.config["persist_dir"] + "metadata_db"
            )
            vecmanager = VectorStoreManager(client, self.config)
            vecmanager.add_df_chunks_to_db(all_dataset_metadata)

        return (
            all_dataset_metadata[["did", "name", "Combined_information"]],
            all_dataset_metadata,
        )


class OpenMLFlowHandler(OpenMLObjectHandler):
    """
    Description: The class for handling OpenML flow objects.
    """

    def get_description(self, data_id: int):
        return openml.flows.get_flow(flow_id=data_id)

    def get_openml_objects(self):
        all_objects = openml.flows.list_flows(output_format="dataframe")
        return all_objects.rename(columns={"id": "did"})

    def process_metadata(
        self,
        openml_data_object: Sequence[openml.flows.flow.OpenMLFlow],
        data_id: Sequence[int],
        all_dataset_metadata: pd.DataFrame,
        file_path: str,
        subset_ids=None,
    ):
        descriptions = [
            self.extract_attribute(attr, "description") for attr in openml_data_object
        ]
        names = [self.extract_attribute(attr, "name") for attr in openml_data_object]
        tags = [self.extract_attribute(attr, "tags") for attr in openml_data_object]

        all_data_description_df = pd.DataFrame(
            {
                "did": data_id,
                "description": descriptions,
                "name": names,
                "tags": tags,
            }
        )

        all_data_description_df["Combined_information"] = all_data_description_df.apply(
            self.merge_all_columns_to_string, axis=1
        )
        # subset the metadata if subset_ids is not None

        all_dataset_metadata = self.subset_metadata(subset_ids, all_dataset_metadata)

        all_data_description_df.to_csv(file_path)

        return (
            all_data_description_df[["did", "name", "Combined_information"]],
            all_data_description_df,
        )


class OpenMLMetadataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.save_filename = os.path.join(
            config["data_dir"], f"all_{config['type_of_data']}_metadata.pkl"
        )
        self.description_filename = os.path.join(
            config["data_dir"], f"all_{config['type_of_data']}_description.csv"
        )

    def get_all_metadata_from_openml(self):
        """
        Description: Gets all the metadata from OpenML for the type of data specified in the config.
        If training is set to False, it loads the metadata from the files. If training is set to True, it gets the metadata from OpenML.

        This uses parallel threads (pqdm) and so to ensure thread safety, install the package oslo.concurrency.
        """
        if not self.config.get("training", False) or self.config.get(
            "ignore_downloading_data", False
        ):
            if not os.path.exists(self.save_filename):
                raise Exception(
                    "Metadata files do not exist. Please run the training pipeline first."
                )
            print("[INFO] Loading metadata from file.")
            return self.load_metadata_from_file(self.save_filename)

        print("[INFO] Training is set to True.")
        handler = (
            OpenMLDatasetHandler(self.config)
            if self.config["type_of_data"] == "dataset"
            else OpenMLFlowHandler(self.config)
        )

        all_objects = handler.get_openml_objects()

        if self.config.get("test_subset", False):
            print("[INFO] Subsetting the data.")
            all_objects = all_objects[:500]

        data_id = [int(all_objects.iloc[i]["did"]) for i in range(len(all_objects))]

        print("[INFO] Initializing cache.")
        handler.initialize_cache(data_id)

        print(f"[INFO] Getting {self.config['type_of_data']} metadata from OpenML.")
        openml_data_object = handler.get_metadata(data_id)

        print("[INFO] Saving metadata to file.")
        self.save_metadata_to_file(
            (openml_data_object, data_id, all_objects, handler), self.save_filename
        )

        return openml_data_object, data_id, all_objects, handler

    def load_metadata_from_file(self, filename: str):
        # Implement the function to load metadata from a file
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save_metadata_to_file(self, data: Tuple, save_filename: str):
        # Implement the function to save metadata to a file
        with open(save_filename, "wb") as f:
            pickle.dump(data, f)

    def create_metadata_dataframe(
        self,
        handler: Union["OpenMLDatasetHandler", "OpenMLFlowHandler"],
        openml_data_object: Sequence[
            Union[openml.datasets.dataset.OpenMLDataset, openml.flows.flow.OpenMLFlow]
        ],
        data_id: Sequence[int],
        all_dataset_metadata: pd.DataFrame,
        subset_ids=None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Description: Creates a dataframe with all the metadata, joined columns with all information
        for the type of data specified in the config. If training is set to False,
        the dataframes are loaded from the files. If training is set to True, the
        dataframes are created and then saved to the files.
        """
        if not self.config.get("training", False):
            return (
                handler.load_metadata(self.description_filename),
                all_dataset_metadata,
            )

        return handler.process_metadata(
            openml_data_object,
            data_id,
            all_dataset_metadata,
            self.description_filename,
            subset_ids,
        )
