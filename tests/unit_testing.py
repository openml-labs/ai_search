import unittest
import chromadb
import sys
import os

from backend.modules.llm import *
from backend.modules.utils import *

config = load_config_and_device("./backend/config.json", training=False)

# change directory for tests
config["data_dir"] = "./backend/data/"
config["persist_dir"]= "./backend/data/chroma_db"

class TestConfig(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = chromadb.PersistentClient(path=config["persist_dir"])
        self.config_keys = ["rqa_prompt_template", "llm_prompt_template",
        "num_return_documents", "embedding_model", "llm_model", "num_documents_for_llm", "data_dir", "persist_dir", "testing_flag", "ignore_downloading_data", "test_subset_2000", "data_download_n_jobs", "training", "temperature", "top_p", "search_type", "reranking", "long_context_reorder"]
        self.query_test_dict = {
            "dataset": "Find me a dataset about flowers that has a high number of instances.",
            "flow": "Find me a flow that uses the RandomForestClassifier.",
        }
    def test_check_data_dirs(self):
        """
        Description: Check if the data directory exists.
        Returns: None
        """
        self.assertTrue(os.path.exists(config["data_dir"]))
        self.assertTrue(os.path.exists(config["persist_dir"]))

    def test_config(self):
        """
        Description: Check if the config has the required keys.
        Returns: None
        """
        for key in self.config_keys:
            self.assertIn(key, config.keys())
    
    def test_setup_vector_db_and_qa(self):
        """
        Description: Check if the setup_vector_db_and_qa function works as expected.
        Returns: None
        """
        for type_of_data in ["dataset", "flow"]:
            self.qa = setup_vector_db_and_qa(
                config=config, data_type=type_of_data, client=self.client
            )
            self.assertIsNotNone(self.qa)
            self.result_data_frame = get_result_from_query(
                query=self.query_test_dict[type_of_data],
                qa=self.qa,
                type_of_query=type_of_data,
                config=config,
            )
            self.assertIsNotNone(self.result_data_frame)
        

if __name__ == "__main__":
    unittest.main(verbosity=2)