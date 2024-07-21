import os
import uuid

import pandas as pd
from chromadb.api import ClientAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from tqdm.auto import tqdm


class DataLoader:
    def __init__(self, metadata_df: pd.DataFrame, page_content_column: str):
        self.metadata_df = metadata_df
        self.page_content_column = page_content_column

    def load_and_process_data(self) -> list:
        loader = DataFrameLoader(
            self.metadata_df, page_content_column=self.page_content_column
        )
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        documents = text_splitter.split_documents(documents)

        return documents


class DocumentProcessor:
    @staticmethod
    def generate_unique_documents(documents: list, db: Chroma) -> tuple:
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

        unique_ids = list(set(ids))
        seen_ids = set()
        unique_docs = [
            doc
            for doc, id in zip(documents, ids)
            if id not in seen_ids and (seen_ids.add(id) or True)
        ]

        return unique_docs, unique_ids


class VectorStoreManager:
    def __init__(self, chroma_client: ClientAPI, config: dict):
        self.chroma_client = chroma_client
        self.config = config
        self.chunk_size = 100

    def chunk_dataframe(self, df, chunk_size):
        for i in range(0, df.shape[0], self.chunk_size):
            yield df.iloc[i : i + self.chunk_size]

    def add_df_chunks_to_db(self, metadata_df):
        collec = self.chroma_client.get_or_create_collection("metadata")
        for chunk in tqdm(
            self.chunk_dataframe(metadata_df, self.chunk_size),
            total=(len(metadata_df) // self.chunk_size) + 1,
        ):
            ids = chunk["did"].astype(str).tolist()
            documents = chunk["description"].astype(str).tolist()
            metadatas = chunk.to_dict(orient="records")

            # Add to ChromaDB collection
            collec.add(ids=ids, documents=documents, metadatas=metadatas)

    def load_model(self) -> HuggingFaceEmbeddings | None:
        print("[INFO] Loading model...")
        model_kwargs = {"device": self.config["device"], "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"],
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=False,
        )
        print("[INFO] Model loaded.")
        return embeddings

    def get_collection_name(self) -> str:
        return {"dataset": "datasets", "flow": "flows"}.get(
            self.config["type_of_data"], "default"
        )

    def load_vector_store(
        self, embeddings: HuggingFaceEmbeddings, collection_name: str
    ) -> Chroma:
        if not os.path.exists(self.config["persist_dir"]):
            raise Exception(
                "Persist directory does not exist. Please run the training pipeline first."
            )

        return Chroma(
            client=self.chroma_client,
            persist_directory=self.config["persist_dir"],
            embedding_function=embeddings,
            collection_name=collection_name,
        )

    def add_documents_to_db(self, db, unique_docs, unique_ids):
        bs = 512
        if len(unique_docs) < bs:
            db.add_documents(unique_docs, ids=unique_ids)
        else:
            for i in range(0, len(unique_docs), bs):
                db.add_documents(unique_docs[i : i + bs], ids=unique_ids[i : i + bs])

    def create_vector_store(self, metadata_df: pd.DataFrame) -> Chroma:
        embeddings = self.load_model()
        collection_name = self.get_collection_name()

        db = Chroma(
            client=self.chroma_client,
            embedding_function=embeddings,
            persist_directory=self.config["persist_dir"],
            collection_name=collection_name,
        )

        data_loader = DataLoader(
            metadata_df, page_content_column="Combined_information"
        )
        documents = data_loader.load_and_process_data()

        if self.config["testing_flag"]:
            if self.config["test_subset"]:
                print("[INFO] Subsetting the data.")
                documents = documents[:500]

        unique_docs, unique_ids = DocumentProcessor.generate_unique_documents(
            documents, db
        )
        print(
            f"Number of unique documents: {len(unique_docs)} vs Total documents: {len(documents)}"
        )

        if len(unique_docs) == 0:
            print("No new documents to add.")
        else:
            self.add_documents_to_db(db, unique_docs, unique_ids)

        return db
