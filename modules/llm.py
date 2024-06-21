# This file contains all the LLM related code - models, vector stores, and the retrieval QA chain etc.
from __future__ import annotations
import os
import uuid
from chromadb.api import ClientAPI
import langchain
import pandas as pd

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores.chroma import Chroma
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain import PromptTemplate
from typing import Sequence
from langchain_core.documents import Document
from langchain.chains.llm import LLMChain

from .metadata_utils import (create_metadata_dataframe,
                             get_all_metadata_from_openml)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# --- ADDING OBJECTS TO CHROMA DB AND LOADING THE VECTOR STORE ---


# def load_and_process_data(metadata_df, page_content_column):
def load_and_process_data(metadata_df: pd.DataFrame, page_content_column: str) -> list:
    """
    Description: Load and process the data for the vector store. Split the documents into chunks of 1000 characters.

    Input: metadata_df (pd.DataFrame), page_content_column (str)

    Returns: chunked documents (list)
    """
    # Load data
    loader = DataFrameLoader(metadata_df, page_content_column=page_content_column)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(documents)

    return documents


def generate_unique_documents(documents: list, db: Chroma) -> tuple:
    """
    Description: Generate unique documents by removing duplicates. This is done by generating unique IDs for the documents and keeping only one of the duplicate IDs.
        Source: https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist

    Input: documents (list)

    Returns: unique_docs (list), unique_ids (list)
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
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS,doc.page_content)) for doc in documents]

    # Remove duplicates based on document content (from new documents)
    unique_ids = list(set(ids))
    seen_ids = set()
    unique_docs = [
            doc
            for doc, id in zip(documents, ids)
            if id not in seen_ids and (seen_ids.add(id) or True)
        ]

    return unique_docs, unique_ids


# def load_document_and_create_vector_store(metadata_df, chroma_client, config) -> Chroma:
def load_document_and_create_vector_store(metadata_df: pd.DataFrame, chroma_client:ClientAPI , config: dict) -> Chroma:
    """
    Loads the documents and creates the vector store. If the training flag is set to True,
    the documents are added to the vector store. If the training flag is set to False,
    the vector store is loaded from the persist directory.

    Args:
        metadata_df (pd.DataFrame): The metadata dataframe.
        chroma_client (chromadb.PersistentClient): The Chroma client.
        config (dict): The configuration dictionary.

    Returns:
        Chroma: The Chroma vector store.
    """
    embeddings = load_model(config)
    collection_name = get_collection_name(config)

    if not config["training"]:
        return load_vector_store(chroma_client, config, embeddings, collection_name)

    return create_vector_store(
        metadata_df, chroma_client, config, embeddings, collection_name
    )


def load_model(config: dict) -> HuggingFaceEmbeddings | None:
    """
    Description: Load the model using HuggingFaceEmbeddings.
    
    Input: config (dict)
    
    Returns: HuggingFaceEmbeddings
    """
    print("[INFO] Loading model...")
    model_kwargs = {"device": config["device"], "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embedding_model"],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress = True,
        # trust_remote_code=True
    )
    print("[INFO] Model loaded.")
    return embeddings


def get_collection_name(config: dict) -> str:
    """
    Description: Get the collection name based on the type of data provided in the config.
    
    Input: config (dict)
    
    Returns: str
    """
    return {"dataset": "datasets", "flow": "flows"}.get(
        config["type_of_data"], "default"
    )


def load_vector_store(chroma_client: ClientAPI, config: dict, embeddings: HuggingFaceEmbeddings, collection_name: str) -> Chroma:
    """
    Description: Load the vector store from the persist directory.
    
    Input: chroma_client (chromadb.PersistentClient), config (dict), embeddings (HuggingFaceEmbeddings), collection_name (str)
    
    Returns: Chroma
    """
    if not os.path.exists(config["persist_dir"]):
        raise Exception(
            "Persist directory does not exist. Please run the training pipeline first."
        )

    return Chroma(
        client=chroma_client,
        persist_directory=config["persist_dir"],
        embedding_function=embeddings,
        collection_name=collection_name,
    )

def add_documents_to_db(db, unique_docs, unique_ids):
    """
    Description: Add documents to the vector store in batches of 200.

    Input: db (Chroma), unique_docs (list), unique_ids (list)

    Returns: None
    """
    bs = 512
    if len(unique_docs) < bs:
        db.add_documents(unique_docs, ids=unique_ids)
    else:
        for i in tqdm(range(0, len(unique_docs), bs)):
            db.add_documents(unique_docs[i : i + bs], ids=unique_ids[i : i + bs])


# def create_vector_store(
#     metadata_df, chroma_client, config, embeddings, collection_name
# ):
def create_vector_store(
    metadata_df: pd.DataFrame, chroma_client:ClientAPI, config: dict, embeddings: HuggingFaceEmbeddings, collection_name: str 
) -> Chroma:
    """
    Description: Create the vector store using Chroma db. The documents are loaded and processed, unique documents are generated, and the documents are added to the vector store.
    
    Input: metadata_df (pd.DataFrame), chroma_client (chromadb.PersistentClient), config (dict), embeddings (HuggingFaceEmbeddings), collection_name (str)
    
    Returns: db (Chroma)
    """

    db = Chroma(
        client=chroma_client,
        embedding_function=embeddings,
        persist_directory=config["persist_dir"],
        collection_name=collection_name,
    )

    documents = load_and_process_data(
        metadata_df, page_content_column="Combined_information"
    )
    if config["testing_flag"]:
        # subset the data for testing
        if config["test_subset_2000"] == True:
            print("[INFO] Subsetting the data to 2000 rows.")
            documents = documents[:2000]
    unique_docs, unique_ids = generate_unique_documents(documents, db)

    print(
        f"Number of unique documents: {len(unique_docs)} vs Total documents: {len(documents)}"
    )
    if len(unique_docs) == 0:
        print("No new documents to add.")
        return db
    else:
        # db.add_documents(unique_docs, ids=unique_ids)
        add_documents_to_db(db, unique_docs, unique_ids)

    return db


# --- LLM CHAIN SETUP ---


def initialize_llm_chain(
    vectordb: Chroma,
    config : dict
) -> langchain.chains.retrieval_qa.base.RetrievalQA:
    """
    Description: Initialize the LLM chain and setup Retrieval QA with the specified configuration.

    Input: vectordb (Chroma), config (dict)

    Returns: qa (langchain.chains.retrieval_qa.base.RetrievalQA)
    """

    return vectordb.as_retriever(
        search_type=config["search_type"],
        search_kwargs={"k": config["num_return_documents"]},
    )


def setup_vector_db_and_qa(config: dict, data_type: str, client:ClientAPI) -> langchain.chains.retrieval_qa.base.RetrievalQA:
    """
    Description: Create the vector database using Chroma db with each type of data in its own collection. Doing so allows us to have a single database with multiple collections, reducing the number of databases we need to manage.
    This also downloads the embedding model if it does not exist. The QA chain is then initialized with the vector store and the configuration.

    Input: config (dict), data_type (str), client (chromadb.PersistentClient)

    Returns: qa (langchain.chains.retrieval_qa.base.RetrievalQA)
    """

    config["type_of_data"] = data_type
    # Download the data if it does not exist
    openml_data_object, data_id, all_metadata = get_all_metadata_from_openml(
        config=config
    )
    # Create the combined metadata dataframe
    metadata_df, all_metadata = create_metadata_dataframe(
        openml_data_object, data_id, all_metadata, config=config
    )
    # Create the vector store
    vectordb = load_document_and_create_vector_store(
        metadata_df, config=config, chroma_client=client
    )
    # Initialize the LLM chain and setup Retrieval QA
    qa = initialize_llm_chain(vectordb=vectordb, config=config)
    return qa


def get_llm_chain(config: dict) -> LLMChain:
    """
    Description: Get the LLM chain with the specified model and prompt template.
    
    Input: config (dict)
    
    Returns: LLMChain
    """

    llm = Ollama(
        model = config["llm_model"] 
    )  
    map_template = config["llm_prompt_template"]
    map_prompt = PromptTemplate.from_template(map_template)
    return LLMChain(llm=llm, prompt=map_prompt)

async def get_llm_result(docs: Sequence[Document], config:dict):
    llm_chain = get_llm_chain(config=config)
    return llm_chain.invoke({"docs": docs})["text"]