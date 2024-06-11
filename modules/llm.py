# This file contains all the LLM related code - models, vector stores, and the retrieval QA chain etc.

import os
import uuid

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores.chroma import Chroma
from tqdm import tqdm

from .metadata_utils import (create_metadata_dataframe,
                             get_all_metadata_from_openml)

# from langchain_community.llms import HuggingFaceHub


os.environ["TOKENIZERS_PARALLELISM"] = "true"

# --- ADDING OBJECTS TO CHROMA DB AND LOADING THE VECTOR STORE ---


def load_and_process_data(metadata_df, page_content_column):
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


def generate_unique_documents(documents):
    """
    Description: Generate unique documents by removing duplicates. This is done by generating unique IDs for the documents and keeping only one of the duplicate IDs.
        Source: https://stackoverflow.com/questions/76265631/chromadb-add-single-document-only-if-it-doesnt-exist

    Input: documents (list)

    Returns: unique_docs (list), unique_ids (list)
    """
    #
    # Generate unique IDs for the documents
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]
    unique_ids = list(set(ids))

    # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
    seen_ids = set()
    unique_docs = [
        doc
        for doc, id in zip(documents, ids)
        if id not in seen_ids and (seen_ids.add(id) or True)
    ]

    return unique_docs, unique_ids


def add_documents_to_db(db, unique_docs, unique_ids):
    """
    Description: Add documents to the vector store in batches of 200.

    Input: db (Chroma), unique_docs (list), unique_ids (list)

    Returns: None
    """
    if len(unique_docs) < 200:
        db.add_documents(unique_docs, ids=unique_ids)
    else:
        for i in tqdm(range(0, len(unique_docs), 200)):
            db.add_documents(unique_docs[i : i + 200], ids=unique_ids[i : i + 200])


def load_document_and_create_vector_store(
    metadata_df,
    chroma_client,
    config,
) -> Chroma:
    """
    Description: Load the documents and create the vector store. If the training flag is set to True, the documents are added to the vector store. If the training flag is set to False, the vector store is loaded from the persist directory.

    Input: metadata_df (pd.DataFrame), chroma_client (chromadb.PersistentClient), config (dict)

    Returns: db (Chroma)
    """
    # load model
    print("[INFO] Loading model...")
    model_kwargs = {"device": config["device"]}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embedding_model"],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        # show_progress = True
    )
    print("[INFO] Model loaded.")
    # Collection names are used to separate the different types of data in the database

    dict_collection_names = {"dataset": "datasets", "flow": "flows"}

    # If we are not training, we do not need to recreate the cache and can load the metadata from the files. If the files do not exist, raise an exception.
    if config["training"] == False:
        # Check if the directory already exists, if not raise an exception
        if not os.path.exists(config["persist_dir"]):
            raise Exception(
                "Persist directory does not exist. Please run the training pipeline first."
            )
        # load the vector store
        return Chroma(
            client=chroma_client,
            persist_directory=config["persist_dir"],
            embedding_function=embeddings,
            collection_name=dict_collection_names[config["type_of_data"]],
        )

    elif config["training"] == True:
        # Load and process data
        documents = load_and_process_data(
            metadata_df, page_content_column="Combined_information"
        )
        # Generate unique documents
        unique_docs, unique_ids = generate_unique_documents(documents)

        # Determine the collection name based on the type of data
        collection_name = dict_collection_names[config["type_of_data"]]

        # Initialize the database
        db = Chroma(
            embedding_function=embeddings,
            persist_directory=config["persist_dir"],
            collection_name=collection_name,
        )

        # Add documents to the database
        add_documents_to_db(db, unique_docs, unique_ids)

        return db


def setup_vector_db_and_qa(config, data_type, client):
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
    # Add Bm25Retriever and EnsembleRetriever 

    # Initialize the LLM chain and setup Retrieval QA
    qa = initialize_llm_chain(vectordb=vectordb, config=config)
    return qa


# --- LLM CHAIN SETUP ---


def initialize_llm_chain(
    vectordb,
    config,
):
    """
    Description: Initialize the LLM chain and setup Retrieval QA with the specified configuration.

    Input: vectordb (Chroma), config (dict)

    Returns: qa (langchain.chains.retrieval_qa.base.RetrievalQA)
    """
    # HUGGINGFACEHUB_API_KEY = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    # use export HUGGINGFACEHUB_API_TOKEN=your_token_here to set the token (in the shell)

    return vectordb.as_retriever(
        search_type=config["search_type"],
        search_kwargs={"k": config["num_return_documents"]},
    )


def setup_vector_db_and_qa(config, data_type, client):
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
