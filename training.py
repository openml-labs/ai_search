import chromadb

from modules.llm import *
from modules.utils import *

# Load the config file and set training to true
config = load_config_and_device("config.json", training=True)
if config["testing_flag"] == True:
    config["persist_dir"] = "./data/chroma_db_testing/"
    config["test_subset_2000"] = True
    config["data_dir"] = "./data/testing_data/"
    config["training"] = True
    # config["device"] = "cpu"

client = chromadb.PersistentClient(path=config["persist_dir"])


print(config)
print("[INFO] Training is set to True.")
# Set up Test query
query_test_dict = {
    "dataset": "Find me a dataset about flowers that has a high number of instances.",
    "flow": "Find me a flow that uses the RandomForestClassifier.",
}

# Download the data, generate metadata, create the vector database, create the LLM chain, and run a test query
for type_of_data in ["dataset", "flow"]:
    print(f"[INFO] Type of data - {type_of_data}")
    config["type_of_data"] = type_of_data
    # Check if ./data/ folder exists if not create it
    if not os.path.exists(config["data_dir"]):
        os.makedirs(config["data_dir"])

    qa = setup_vector_db_and_qa(
        config=config, data_type=config["type_of_data"], client=client
    )

    # Run the test query
    result_data_frame = get_result_from_query(
        query=query_test_dict[type_of_data],
        qa=qa,
        type_of_query=type_of_data,
        config=config,
    )
    print(result_data_frame)
