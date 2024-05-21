import os, tempfile
from pathlib import Path
from glob import glob

from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

# from langchain.llms import VertexAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    PyPDFDirectoryLoader,
    DirectoryLoader,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    TextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from tqdm import tqdm
import pickle
from pqdm.threads import pqdm

import pandas as pd

from typing import Union, List

import openml


def get_dataset_description(dataset_name) -> openml.datasets.dataset.OpenMLDataset:
    try:
        data = openml.datasets.get_dataset(dataset_name, download_data = False, download_qualities = False, download_features_meta_data = False)
    except Exception as e:
        print(e)
    return data

def get_all_dataset_metadata_from_openml(save_filename = "all_dataset_metadata.pkl") -> Union[List, List]:
    # Gather all OpenML datasets
    all_datasets = openml.datasets.list_datasets(output_format="dataframe")

    # List dataset 'did' to be used as an identifier 
    data_id = [all_datasets.iloc[i]['did'] for i in range(len(all_datasets))]

    dataset_names = all_datasets['name'].tolist() # get a list of all dataset names

    # if the file already exists, load it else get the metadata from openml
    if os.path.exists(save_filename):
        with open(save_filename, 'rb') as f:
            all_data_descriptions = pickle.load(f)
        return all_data_descriptions, data_id
    else:
        # Get all dataset metadata using n_jobs parallel threads from openml
        all_data_descriptions = pqdm(dataset_names, get_dataset_description, n_jobs=10)

        # Save the metadata to a file
        with open(save_filename, 'wb') as f:
            pickle.dump(all_data_descriptions, f)
        
        return all_data_descriptions, data_id


def create_metadata_dataframe(all_data_descriptions, data_id) -> pd.DataFrame:
    descriptions = [all_data_descriptions[i].description for i in range(len(all_data_descriptions))]

    all_data_description = dict(zip(data_id, descriptions))

    return pd.DataFrame(list(all_data_description.items()),columns = ['did','description'])

def clean_metadata_dataframe(metadata_df) -> pd.DataFrame:
    # remove rows with empty descriptions
    metadata_df = metadata_df[metadata_df['description'].notna()]
    return metadata_df