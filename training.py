from modules.llm import *
from modules.utils import *

# Load the config file and set training to true
config = load_config_and_device("config.json")
config["training"] == True

# Download the data, generate metadata, create the vector database, create the LLM chain, and run a test query 
for type_of_data in ["dataset", "flow"]:
    # Check if ./data/ folder exists if not create it
    if not os.path.exists("./data/"):
        os.makedirs("./data/")
    # Download the data if it does not exist
    openml_data_object, data_id, all_dataset_metadata = get_all_metadata_from_openml(
        config=config
    )

    # Create the combined metadata dataframe
    metadata_df, all_dataset_metadata = create_metadata_dataframe(
        openml_data_object, data_id, all_dataset_metadata, config=config
    )

    # Create the vector database using Chroma db with each type of data in its own collection. Doing so allows us to have a single database with multiple collections, reducing the number of databases we need to manage.
    # This also downloads the embedding model if it does not exist
    vectordb = load_document_and_create_vector_store(metadata_df, config=config)

    # TESTING

    # Set up Test query
    query_test_dict = {
        "dataset": "Find me a dataset about flowers that has a high number of instances.",
        "flow": "Find me a flow that uses the RandomForestClassifier.",
    }

    # Setup llm chain, initialize the retriever and llm, and setup Retrieval QA
    qa = initialize_llm_chain(vectordb=vectordb, config=config)

    # Run the test query
    result_data_frame = get_result_from_query(
        query=query_test_dict[type_of_data], qa=qa, config=config
    )
