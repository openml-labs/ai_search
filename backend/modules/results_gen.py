# This file pertains to all the utility functions required for creating and managing the vector database

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence, Tuple

import langchain
import pandas as pd
from flashrank import Ranker, RerankRequest
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_transformers.long_context_reorder import (
    LongContextReorder,
)
from langchain_core.documents import BaseDocumentTransformer, Document
from tqdm import tqdm

# --- PROCESSING RESULTS ---


def long_context_reorder(results):
    """
    Description: Lost in the middle reorder: the less relevant documents will be at the
    middle of the list and more relevant elements at beginning / end.
    See: https://arxiv.org/abs//2307.03172


    """
    print("[INFO] Reordering results...")
    reordering = LongContextReorder()
    results = reordering.transform_documents(results)
    print("[INFO] Reordering complete.")
    return results


class QueryProcessor:
    def __init__(self, query: str, qa: RetrievalQA, type_of_query: str, config: dict):
        self.query = query
        self.qa = qa
        self.type_of_query = type_of_query
        self.config = config

    def fetch_results(self):
        """
        Fetch results for the query using the QA chain.
        """
        results = self.qa.invoke(
            input=self.query,
            config={
                "temperature": self.config["temperature"],
                "top-p": self.config["top_p"],
            },
        )
        if self.config["long_context_reorder"]:
            results = long_context_reorder(results)
        id_column = {"dataset": "did", "flow": "id", "data": "did"}[self.type_of_query]

        if self.config["reranking"]:
            try:
                print("[INFO] Reranking results...")
                ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/")
                rerankrequest = RerankRequest(
                    query=self.query,
                    passages=[
                        {"id": result.metadata[id_column], "text": result.page_content}
                        for result in results
                    ],
                )
                ranking = ranker.rerank(rerankrequest)
                ids = [result["id"] for result in ranking]
                ranked_results = [
                    result for result in results if result.metadata[id_column] in ids
                ]
                print("[INFO] Reranking complete.")
                return ranked_results
            except Exception as e:
                print(f"[ERROR] Reranking failed: {e}")
                return results
        else:
            return results

    @staticmethod
    def process_documents(
        source_documents: Sequence[Document],
    ) -> Tuple[OrderedDict, list]:
        """
        Process the source documents and create a dictionary with the key_name as the key and the name and page content as the values.
        """
        dict_results = OrderedDict()
        for result in source_documents:
            dict_results[result.metadata["did"]] = {
                "name": result.metadata["name"],
                "page_content": result.page_content,
            }
        ids = [result.metadata["did"] for result in source_documents]
        return dict_results, ids

    @staticmethod
    def make_clickable(val: str) -> str:
        """
        Make the URL clickable in the dataframe.
        """
        return '<a href="{}">{}</a>'.format(val, val)

    def create_output_dataframe(
        self, dict_results: dict, type_of_data: str, ids_order: list
    ) -> pd.DataFrame:
        """
        Create an output dataframe with the results. The URLs are API calls to the OpenML API for the specific type of data.
        """
        output_df = pd.DataFrame(dict_results).T.reset_index()
        output_df["index"] = output_df["index"].astype(int)
        output_df = output_df.set_index("index").loc[ids_order].reset_index()
        output_df["urls"] = output_df["index"].apply(
            lambda x: f"https://www.openml.org/search?type={type_of_data}&id={x}"
        )
        output_df["urls"] = output_df["urls"].apply(self.make_clickable)

        if type_of_data == "data":
            output_df["command"] = output_df["index"].apply(
                lambda x: f"dataset = openml.datasets.get_dataset({x})"
            )
        elif type_of_data == "flow":
            output_df["command"] = output_df["index"].apply(
                lambda x: f"flow = openml.flows.get_flow({x})"
            )
        output_df = output_df.drop_duplicates(subset=["name"])
        replace_dict = {
            "index": "id",
            "command": "Command",
            "urls": "OpenML URL",
            "page_content": "Description",
        }
        for col in ["index", "command", "urls", "page_content"]:
            if col in output_df.columns:
                output_df = output_df.rename(columns={col: replace_dict[col]})
        return output_df

    @staticmethod
    def check_query(query: str) -> str:
        """
        Performs checks on the query:
        - Replaces %20 with space character (browsers do this automatically when spaces are in the URL)
        - Removes leading and trailing spaces
        - Limits the query to 200 characters
        """
        if query == "":
            raise ValueError("Query cannot be empty.")
        query = query.replace("%20", " ")
        query = query.strip()
        query = query[:200]
        return query

    def get_result_from_query(self) -> Tuple[pd.DataFrame, Sequence[Document]]:
        """
        Get the result from the query using the QA chain and return the results in a dataframe that is then sent to the frontend.
        """
        if self.type_of_query == "dataset":
            type_of_query = "data"
        elif self.type_of_query == "flow":
            type_of_query = "flow"
        else:
            raise ValueError(f"Unsupported type_of_data: {self.type_of_query}")

        query = self.check_query(self.query)
        if query == "":
            return pd.DataFrame(), []

        source_documents = self.fetch_results()
        dict_results, ids_order = self.process_documents(source_documents)
        output_df = self.create_output_dataframe(dict_results, type_of_query, ids_order)

        return output_df, ids_order
