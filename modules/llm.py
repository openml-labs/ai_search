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


def load_document_and_create_vector_store(
    metadata_df,
    persist_directory="./chroma_db/",
    # model_name="BAAI/bge-base-en-v1.5",
    model_name = "Intel/bge-small-en-v1.5-rag-int8-static",
    device="cpu",
    normalize_embeddings=True,
    recreate_chroma=False,
) -> Chroma:
    # load model
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": normalize_embeddings, "quantized": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs,
    )
    # embeddings = QuantizedBiEncoderEmbeddings(
    #     model_name=model_name,
    #     encode_kwargs=encode_kwargs,
    #     model_kwargs=model_kwargs,
    # )

    # if the directory already exists, load the vector store else create a new one
    if os.path.exists(persist_directory) and recreate_chroma == False:
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return db
    else:
        # load data
        loader = DataFrameLoader(metadata_df, page_content_column="Combined_information")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        documents = text_splitter.split_documents(documents)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents
        ]
        unique_ids = list(set(ids))
        # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
        seen_ids = set()
        unique_docs = [
            doc
            for doc, id in zip(documents, ids)
            if id not in seen_ids and (seen_ids.add(id) or True)
        ]

        db = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
        # add documents to the vector store in batches of 100
        # if len(unique_docs) < 100:
        #     db.add_documents(unique_docs, ids=unique_ids)
        # else:
        #     for i in tqdm(range(0, len(unique_docs), 100)):
        #         db.add_documents(unique_docs[i : i + 100], ids=unique_ids[i : i + 100])
        # return db
        for i in tqdm(range(len(unique_docs))):
            db.add_documents(unique_docs[i], ids=unique_ids[i])
        return db


def create_retriever_and_llm(
    vectordb,
    model_repo_id="HuggingFaceH4/zephyr-7b-beta",
    num_return_documents=50,
    search_type="similarity",
):
    HUGGINGFACEHUB_API_KEY = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    # use export HUGGINGFACEHUB_API_TOKEN=your_token_here to set the token (in the shell)

    retriever = vectordb.as_retriever(
        search_type=search_type, search_kwargs={"k": num_return_documents}
    )
    llm = HuggingFaceHub(
        repo_id=model_repo_id,
        # Temperature=0.1,
        # max_length=512,
        model_kwargs={"temperature": 0.1, "max_length": 512},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_KEY,

    )
    return retriever, llm


def create_llm_chain_and_query(
    vectordb,
    retriever,
    llm,
    prompt_template="Answer {question} from the following context: {context}",
)-> langchain.chains.retrieval_qa.base.RetrievalQA:
    RQA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    rqa_chain_type_kwargs = {"prompt": RQA_PROMPT}
    return RetrievalQA.from_chain_type(
        llm,
        # chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=rqa_chain_type_kwargs,
        return_source_documents=True,
        verbose=False,
    )

def create_result_dataframe(query, qa, all_dataset_metadata) -> pd.DataFrame:
    results = qa.invoke({"query": query})
    result_to_dict = {
        result.metadata["did"]: result.page_content
        for result in results["source_documents"]
    }
    df = pd.DataFrame(list(result_to_dict.items()), columns=["did", "name", "Combined_information"])
    # add short description
    # return pd.merge(all_dataset_metadata, df, on='did', how='inner')
    return df
