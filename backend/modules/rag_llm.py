# This file contains all the LLM related code - models, vector stores, and the retrieval QA chain etc.
from __future__ import annotations

import os
from typing import Union

import langchain
import pandas as pd
from chromadb.api import ClientAPI
from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.llms import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

from .metadata_utils import *
from .vector_store_utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class LLMChainInitializer:
    """
    Description: Setup the vectordb (Chroma) as a retriever with parameters
    """

    @staticmethod
    def initialize_llm_chain(
        vectordb: Chroma, config: dict
    ) -> langchain.chains.retrieval_qa.base.RetrievalQA:
        if config["search_type"] == "similarity_score_threshold":
            return vectordb.as_retriever(
                search_type=config["search_type"],
                search_kwargs={
                    "k": config["num_return_documents"],
                    "score_threshold": 0.5,
                },
            )
        else:
            return vectordb.as_retriever(
                search_type=config["search_type"],
                search_kwargs={"k": config["num_return_documents"]},
            )


class QASetup:
    """
    Description: Setup the VectorDB, QA and initalize the LLM for each type of data
    """

    def __init__(
        self, config: dict, data_type: str, client: ClientAPI, subset_ids: list = None
    ):
        self.config = config
        self.data_type = data_type
        self.client = client
        self.subset_ids = subset_ids

    def setup_vector_db_and_qa(self):
        self.config["type_of_data"] = self.data_type

        metadata_processor = OpenMLMetadataProcessor(config=self.config)
        openml_data_object, data_id, all_metadata, handler = (
            metadata_processor.get_all_metadata_from_openml()
        )
        metadata_df, all_metadata = metadata_processor.create_metadata_dataframe(
            handler,
            openml_data_object,
            data_id,
            all_metadata,
            subset_ids=self.subset_ids,
        )

        vector_store_manager = VectorStoreManager(self.client, self.config)
        vectordb = vector_store_manager.create_vector_store(metadata_df, self.data_type)
        qa = LLMChainInitializer.initialize_llm_chain(vectordb, self.config)

        return qa, all_metadata


class LLMChainCreator:
    """
    Description: Gets Ollama, sends query, enables query caching
    """

    def __init__(self, config: dict, local: bool = False):
        self.config = config
        self.local = local

    def get_llm_chain(self) -> LLMChain | bool:
        """
        Description: Send a query to Ollama using the paths.
        """
        base_url = "http://127.0.0.1:11434" if self.local else "http://ollama:11434"
        llm = Ollama(model=self.config["llm_model"], base_url=base_url)
        map_template = self.config["llm_prompt_template"]
        map_prompt = PromptTemplate.from_template(map_template)
        return map_prompt | llm | StrOutputParser()

    def enable_cache(self):
        """
        Description: Enable a cache for queries to prevent running the same query again for no reason.
        """
        set_llm_cache(
            SQLiteCache(
                database_path=os.path.join(self.config["data_dir"], ".langchain.db")
            )
        )
