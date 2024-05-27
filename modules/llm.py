import os
from typing import Tuple
import uuid

import langchain
import langchain_community
import langchain_core
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain_community.embeddings import QuantizedBiEncoderEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores.chroma import Chroma
from tqdm import tqdm


def load_and_process_data(metadata_df, page_content_column):
    # Load data
    loader = DataFrameLoader(metadata_df, page_content_column=page_content_column)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(documents)

    return documents


def generate_unique_documents(documents):
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
    # Add documents to the vector store in batches of 200
    if len(unique_docs) < 200:
        db.add_documents(unique_docs, ids=unique_ids)
    else:
        for i in tqdm(range(0, len(unique_docs), 200)):
            db.add_documents(unique_docs[i : i + 200], ids=unique_ids[i : i + 200])


def load_document_and_create_vector_store(
    metadata_df,
    config,
) -> Chroma:

    # load model
    model_kwargs = {"device": config["device"]}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embedding_model"],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Collection names are used to separate the different types of data in the database

    dict_collection_names = {"dataset": "datasets", "flow": "flows"}

    # If we are not training, we do not need to recreate the cache and can load the metadata from the files. If the files do not exist, raise an exception.
    if config["training"] == False:
        # Check if the directory already exists, if not raise an exception
        if not os.path.exists(config["persist_directory"]):
            raise Exception(
                "Persist directory does not exist. Please run the training pipeline first."
            )
        # load the vector store
        return Chroma(
            persist_directory=config["persist_directory"],
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
            persist_directory=config["persist_directory"],
            collection_name=collection_name,
        )

        # Add documents to the database
        add_documents_to_db(db, unique_docs, unique_ids)

        return db


def create_retriever_and_llm(
    vectordb,
    config,
):
    HUGGINGFACEHUB_API_KEY = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    # use export HUGGINGFACEHUB_API_TOKEN=your_token_here to set the token (in the shell)

    retriever = vectordb.as_retriever(
        search_type=config["search_type"],
        search_kwargs={"k": config["num_return_documents"]},
    )
    llm = HuggingFaceHub(
        repo_id=config["llm_model"],
        # Temperature=0.1,
        # max_length=512,
        model_kwargs={"temperature": 0.1, "max_length": 512},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_KEY,
    )
    return retriever, llm


def initialize_llm_chain(
    vectordb,
    config,
) -> langchain.chains.retrieval_qa.base.RetrievalQA:
    HUGGINGFACEHUB_API_KEY = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    # use export HUGGINGFACEHUB_API_TOKEN=your_token_here to set the token (in the shell)

    retriever = vectordb.as_retriever(
        search_type=config["search_type"],
        search_kwargs={"k": config["num_return_documents"]},
    )
    llm = HuggingFaceHub(
        repo_id=config["llm_model"],
        # Temperature=0.1,
        # max_length=512,
        model_kwargs={"temperature": 0.1, "max_length": 512},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_KEY,
    )
    RQA_PROMPT = PromptTemplate(
        template=config["rqa_prompt_template"], input_variables=["context", "question"]
    )

    rqa_chain_type_kwargs = {"prompt": RQA_PROMPT}
    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs=rqa_chain_type_kwargs,
        return_source_documents=True,
        verbose=False,
    )

def fetch_results(query, qa):
    results = qa.invoke({"query": query})
    return results["source_documents"]

def process_documents(source_documents, key_name):
    dict_results = {}
    for result in source_documents:
        dict_results[result.metadata[key_name]] = {
            "name": result.metadata["name"],
            "page_content": result.page_content,
        }
    return dict_results

def create_output_dataframe(dict_results, type_of_data):
    output_df = pd.DataFrame(dict_results).T.reset_index()
    output_df["urls"] = output_df["index"].apply(
        lambda x: f"https://www.openml.org/api/v1/json/{type_of_data}/{x}"
    )
    return output_df

def get_result_from_query(query, qa, config) -> pd.DataFrame:
    type_of_data = config["type_of_data"]
    if type_of_data == "dataset":
        # Fixing the key_name for dataset because of the way the OpenML API returns the data
        type_of_data = "data"
        key_name = "did"
    elif type_of_data == "flow":
        key_name = "id"
    else:
        raise ValueError(f"Unsupported type_of_data: {type_of_data}")

    source_documents = fetch_results(query, qa)
    dict_results = process_documents(source_documents, key_name)
    output_df = create_output_dataframe(dict_results, type_of_data)
    
    return output_df
