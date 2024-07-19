from __future__ import annotations

import os
import pickle
import uuid
# from pqdm.processes import pqdm
from typing import Sequence, Tuple, Union

import langchain
import openml
import pandas as pd
from chromadb.api import ClientAPI
from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from pqdm.threads import pqdm
from tqdm import tqdm

# -- DOWNLOAD METADATA --


# def load_metadata_from_file(
#     save_filename: str,
# ) -> Tuple[pd.DataFrame, Sequence[int], pd.DataFrame]:
#     """
#     Load metadata from a file.
#     """
#     with open(save_filename, "rb") as f:
#         return pickle.load(f)


# def save_metadata_to_file(data, save_filename: str):
#     """
#     Save metadata to a file.
#     """
#     with open(save_filename, "wb") as f:
#         pickle.dump(data, f)


class OpenMLObjectHandler:
    """
    Description: The base class for handling OpenML objects.
    """

    def __init__(self, config):
        self.config = config
        self.collection_name = ""

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

        # subset the metadata if subset_ids is not None
        if subset_ids is not None:
            subset_ids = [int(x) for x in subset_ids]
            all_dataset_metadata = all_dataset_metadata[
                all_dataset_metadata["did"].isin(subset_ids)
            ]

        all_dataset_metadata.to_csv(file_path)

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
        # subset the metadata if subset_ids is not None
        if subset_ids is not None:
            subset_ids = [int(x) for x in subset_ids]
            all_dataset_metadata = all_dataset_metadata[
                all_dataset_metadata["did"].isin(subset_ids)
            ]
        all_data_description_df.to_csv(file_path)

        return (
            all_data_description_df[["did", "name", "Combined_information"]],
            all_data_description_df,
        )


# install the package oslo.concurrency to ensure thread safety
# def get_all_metadata_from_openml(config) -> Tuple[pd.DataFrame, Sequence[int], pd.DataFrame]:
# def get_all_metadata_from_openml(
#     config: dict,
# ) -> Tuple[pd.DataFrame, Sequence[int], pd.DataFrame] | None:
#     """
#     Description: Gets all the metadata from OpenML for the type of data specified in the config.
#     If training is set to False, it loads the metadata from the files. If training is set to True, it gets the metadata from OpenML.

#     This uses parallel threads (pqdm) and so to ensure thread safety, install the package oslo.concurrency.
#     """

#     # save_filename = f"./data/all_{config['type_of_data']}_metadata.pkl"
#     # use os.path.join to ensure compatibility with different operating systems
#     save_filename = os.path.join(
#         config["data_dir"], f"all_{config['type_of_data']}_metadata.pkl"
#     )
#     # If we are not training, we do not need to recreate the cache and can load the metadata from the files. If the files do not exist, raise an exception.
#     # TODO : Check if this behavior is correct, or if data does not exist, send to training pipeline?
#     if config["training"] == False or config["ignore_downloading_data"] == True:
#         # print("[INFO] Training is set to False.")
#         # Check if the metadata files exist for all types of data
#         if not os.path.exists(save_filename):
#             raise Exception(
#                 "Metadata files do not exist. Please run the training pipeline first."
#             )
#         print("[INFO] Loading metadata from file.")
#         # Load the metadata files for all types of data
#         return load_metadata_from_file(save_filename)

#     # If we are training, we need to recreate the cache and get the metadata from OpenML
#     if config["training"] == True:
#         print("[INFO] Training is set to True.")
#         # Gather all OpenML objects of the type of data
#         handler = (
#             OpenMLDatasetHandler(config)
#             if config["type_of_data"] == "dataset"
#             else OpenMLFlowHandler(config)
#         )

#         all_objects = handler.get_openml_objects()

#         # subset the data for testing
#         if config["test_subset"] == True:
#             print("[INFO] Subsetting the data.")
#             all_objects = all_objects[:500]

#         data_id = [int(all_objects.iloc[i]["did"]) for i in range(len(all_objects))]

#         print("[INFO] Initializing cache.")
#         handler.initialize_cache(data_id)

#         print(f"[INFO] Getting {config['type_of_data']} metadata from OpenML.")
#         openml_data_object = handler.get_metadata(data_id)

#         print("[INFO] Saving metadata to file.")
#         save_metadata_to_file(
#             (openml_data_object, data_id, all_objects, handler), save_filename
#         )

#         return openml_data_object, data_id, all_objects, handler


# -- COMBINE METADATA INTO A SINGLE DATAFRAME --


def extract_attribute(attribute: object, attr_name: str) -> str:
    """
    Description: Extract an attribute from the OpenML object.


    """
    return getattr(attribute, attr_name, "")


def join_attributes(attribute: object, attr_name: str) -> str:
    """
    Description: Join the attributes of the OpenML object.


    """

    return (
        " ".join([f"{k} : {v}," for k, v in getattr(attribute, attr_name, {}).items()])
        if hasattr(attribute, attr_name)
        else ""
    )


def create_combined_information_df(
    # data_id, descriptions, joined_qualities, joined_features
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


def merge_all_columns_to_string(row: pd.Series) -> str:
    """
    Description: Create a single column that has a combined string of all the metadata and the description in the form of "column - value, column - value, ... description"


    """

    return " ".join([f"{col} - {val}," for col, val in zip(row.index, row.values)])


# def combine_metadata(all_dataset_metadata, all_data_description_df):
def combine_metadata(
    all_dataset_metadata: pd.DataFrame, all_data_description_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Description: Combine the descriptions with the metadata table.


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


# def create_metadata_dataframe(
#     # openml_data_object, data_id, all_dataset_metadata, config
#     handler: OpenMLObjectHandler,
#     openml_data_object: Sequence[
#         Union[openml.datasets.dataset.OpenMLDataset, openml.flows.flow.OpenMLFlow]
#     ],
#     data_id: Sequence[int],
#     all_dataset_metadata: pd.DataFrame,
#     config: dict,
#     subset_ids=None,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Description: Creates a dataframe with all the metadata, joined columns with all information
#     for the type of data specified in the config. If training is set to False,
#     the dataframes are loaded from the files. If training is set to True, the
#     dataframes are created and then saved to the files.

#     """
#     # use os.path.join to ensure compatibility with different operating systems
#     file_path = os.path.join(
#         config["data_dir"], f"all_{config['type_of_data']}_description.csv"
#     )

#     if not config["training"]:
#         return handler.load_metadata(file_path), all_dataset_metadata

#     return handler.process_metadata(
#         openml_data_object, data_id, all_dataset_metadata, file_path, subset_ids
#     )


class OpenMLMetadataGetter:
    def __init__(self, config, chroma_client):
        self.config = config
        self.handler = (
            OpenMLDatasetHandler(self.config)
            if self.config["type_of_data"] == "dataset"
            else OpenMLFlowHandler(self.config)
        )
        self.all_objects = self.handler.get_openml_objects()
        self.chroma_client = chroma_client
        self.db = self.call_vector_store()
        self.all_metadata = None

    def load_metadata_from_file(self, save_filename: str):
        """
        Load metadata from a file.
        """
        with open(save_filename, "rb") as f:
            return pickle.load(f)

    def save_metadata_to_file(self, data, save_filename: str):
        """
        Save metadata to a file.
        """
        with open(save_filename, "wb") as f:
            pickle.dump(data, f)

    def get_all_metadata_from_openml(self):
        """
        Description: Gets all the metadata from OpenML for the type of data specified in the config.
        If training is set to False, it loads the metadata from the files. If training is set to True, it gets the metadata from OpenML.

        This uses parallel threads (pqdm) and so to ensure thread safety, install the package oslo.concurrency.
        """

        save_filename = os.path.join(
            self.config["data_dir"], f"all_{self.config['type_of_data']}_metadata.pkl"
        )
        # If we are not training, we do not need to recreate the cache and can load the metadata from the files. If the files do not exist, raise an exception.
        # TODO : Check if this behavior is correct, or if data does not exist, send to training pipeline?
        if (
            self.config["training"] == False
            or self.config["ignore_downloading_data"] == True
        ):
            # print("[INFO] Training is set to False.")
            # Check if the metadata files exist for all types of data
            if not os.path.exists(save_filename):
                raise Exception(
                    "Metadata files do not exist. Please run the training pipeline first."
                )
            print("[INFO] Loading metadata from file.")
            # Load the metadata files for all types of data
            return self.load_metadata_from_file(save_filename)

        # If we are training, we need to recreate the cache and get the metadata from OpenML

        if self.config["training"] == True:
            print("[INFO] Training is set to True.")
            # Gather all OpenML objects of the type of data
            # subset the data for testing
            if self.config["test_subset"] == True:
                print("[INFO] Subsetting the data.")
                self.all_objects = self.all_objects[:500]

            self.data_id = [
                int(self.all_objects.iloc[i]["did"])
                for i in range(len(self.all_objects))
            ]

            print("[INFO] Initializing cache.")
            self.handler.initialize_cache(self.data_id)

            print(f"[INFO] Getting {self.config['type_of_data']} metadata from OpenML.")
            self.openml_data_object = self.handler.get_metadata(self.data_id)

            print("[INFO] Saving metadata to file.")

            self.save_metadata_to_file(
                (self.openml_data_object, self.data_id, self.all_objects, self.handler),
                save_filename,
            )

    def create_metadata_dataframe_and_return_qa(self, subset_ids):
        """
        Description: Creates a dataframe with all the metadata, joined columns with all information
        for the type of data specified in the config. If training is set to False,
        the dataframes are loaded from the files. If training is set to True, the
        dataframes are created and then saved to the files.

        """
        file_path = os.path.join(
            self.config["data_dir"],
            f"all_{self.config['type_of_data']}_description.csv",
        )

        if not self.config["training"]:
            return self.handler.load_metadata(file_path), self.all_objects

        self.metadata_df, self.all_metadata = self.handler.process_metadata(
            self.openml_data_object,
            self.data_id,
            self.all_objects,
            file_path,
            subset_ids,
        )

        self.db = self.call_vector_store()

        # Create the vector store
        self.db.load_document_and_create_vector_store(self.metadata_df)
        create_llm_chain = CreateLLMChain(self.db, self.config)
        create_llm_chain.initialize_llm_chain()
        return create_llm_chain.get_llm_chain()


    def call_vector_store(self) -> VectorStoreClass:
        return VectorStoreClass(self.chroma_client, self.config)
    
    # def return_qa_chain(self):
        
        

class VectorStoreClass:
    def __init__(self, chroma_client, config):
        self.chroma_client = chroma_client
        self.config = config

    def load_model(self) -> HuggingFaceEmbeddings | None:
        """
        Description: Load the model using HuggingFaceEmbeddings.
        """
        print("[INFO] Loading model...")
        model_kwargs = {"device": self.config["device"], "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"],
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=False,
            # trust_remote_code=True
        )
        print("[INFO] Model loaded.")
        return embeddings

    def get_collection_name(self) -> str:
        """
        Description: Get the collection name based on the type of data provided in the config.

        """
        return {"dataset": "datasets", "flow": "flows"}.get(
            self.config["type_of_data"], "default"
        )

    def load_vector_store(
        self,
        embeddings: HuggingFaceEmbeddings | None,
    ) -> Chroma:
        """
        Description: Load the vector store from the persist directory.
        """
        if not os.path.exists(self.config["persist_dir"]):
            raise Exception(
                "Persist directory does not exist. Please run the training pipeline first."
            )

        return Chroma(
            client=self.chroma_client,
            persist_directory=self.config["persist_dir"],
            embedding_function=embeddings,
            collection_name=self.get_collection_name(),
        )

    def add_documents_to_db(self, db, unique_docs, unique_ids):
        """
        Description: Add documents to the vector store in batches of 200.

        """
        bs = 512
        if len(unique_docs) < bs:
            db.add_documents(unique_docs, ids=unique_ids)
        else:
            # for i in tqdm(range(0, len(unique_docs), bs)):
            for i in range(0, len(unique_docs), bs):
                db.add_documents(unique_docs[i : i + bs], ids=unique_ids[i : i + bs])

    def generate_unique_documents(self, documents: list, db: Chroma) -> tuple:
        """
        Description: Generate unique documents by removing duplicates. This is done by generating unique IDs for the documents and keeping only one of the duplicate IDs.
            Source: https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist


        """

        # Remove duplicates based on ID (from database)
        new_document_ids = set([str(x.metadata["did"]) for x in documents])
        print(f"[INFO] Generating unique documents. Total documents: {len(documents)}")
        try:
            old_dids = set([str(x["did"]) for x in db.get()["metadatas"]])
        except KeyError:
            old_dids = set([str(x["id"]) for x in db.get()["metadatas"]])

        new_dids = new_document_ids - old_dids
        documents = [x for x in documents if str(x.metadata["did"]) in new_dids]
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents
        ]

        # Remove duplicates based on document content (from new documents)
        unique_ids = list(set(ids))
        seen_ids = set()
        unique_docs = [
            doc
            for doc, id in zip(documents, ids)
            if id not in seen_ids and (seen_ids.add(id) or True)
        ]

        return unique_docs, unique_ids

    def load_and_process_data(
        self, metadata_df: pd.DataFrame, page_content_column: str
    ) -> list:
        """
        Description: Load and process the data for the vector store. Split the documents into chunks of 1000 characters.


        """
        # Load data
        loader = DataFrameLoader(metadata_df, page_content_column=page_content_column)
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        documents = text_splitter.split_documents(documents)

        return documents

    def create_vector_store(self, embeddings, metadata_df) -> Chroma:
        """
        Description: Create the vector store using Chroma db. The documents are loaded and processed, unique documents are generated, and the documents are added to the vector store.

        """

        self.db = Chroma(
            client=self.chroma_client,
            embedding_function=embeddings,
            persist_directory=self.config["persist_dir"],
            collection_name=self.get_collection_name(),
        )

        documents = self.load_and_process_data(
            metadata_df, page_content_column="Combined_information"
        )
        if self.config["testing_flag"]:
            # subset the data for testing
            if self.config["test_subset"] == True:
                print("[INFO] Subsetting the data.")
                documents = documents[:500]
        unique_docs, unique_ids = self.generate_unique_documents(documents, self.db)

        print(
            f"Number of unique documents: {len(unique_docs)} vs Total documents: {len(documents)}"
        )
        if len(unique_docs) == 0:
            print("No new documents to add.")
            return self.db
        else:
            # db.add_documents(unique_docs, ids=unique_ids)
            self.add_documents_to_db(self.db, unique_docs, unique_ids)

        return self.db

    def load_document_and_create_vector_store(self, metadata_df):
        embeddings = self.load_model()
        if not self.config["training"]:
            return self.load_vector_store(embeddings)

        return self.create_vector_store(embeddings, metadata_df)
    

class CreateLLMChain:
    def __init__(self, vectordb, config) -> None:
        self.vectordb = vectordb
        self.config = config

    def initialize_llm_chain(
        self
    ) -> langchain.chains.retrieval_qa.base.RetrievalQA:
        """
        Description: Initialize the LLM chain and setup Retrieval QA with the specified configuration.

        """
        search_type=self.config["search_type"]
        if search_type == "mmr" or search_type == "similarity_score_threshold":
            search_kwargs={"k": self.config["num_return_documents"], "score_threshold": 0.5}
        else:
            search_kwargs={"k": self.config["num_return_documents"]}

        self.ret = self.vectordb.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
    

def get_llm_chain(config,local: bool = False) -> LLMChain | bool:
    """
    Description: Get the LLM chain with the specified model and prompt template.

    """
    base_url = "http://127.0.0.1:11434" if local else "http://ollama:11434"
    llm = Ollama(model=config["llm_model"], base_url=base_url)
    # llm = Ollama(
    # model = config["llm_model"]
    # )
    # print(llm)
    map_template = config["llm_prompt_template"]
    map_prompt = PromptTemplate.from_template(map_template)
    # return LLMChain(llm=llm, prompt=map_prompt)
    return map_prompt | llm | StrOutputParser()
        