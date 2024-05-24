import argparse
from modules.llm import *
from modules.utils import *

# Argument parser for the query string
parser = argparse.ArgumentParser(description="Query the database")
parser.add_argument(
    "--query",
    type=str,
    help="The query to search the database",
)
args = parser.parse_args()
## Config
config = {
    "rqa_prompt_template" : "This database is a list of dataset metadata. Use the following pieces of context to find the relevant document. Answer only from the context given using the {question} given. If you do not know the answer, say you do not know. {context}",
    "num_return_documents" : 50,
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "llm_model": "HuggingFaceH4/zephyr-7b-beta",
    "persist_dir": "./chroma_db/",
    # "recreate_chroma": False,
    "recreate_chroma": True,
    # "recreate_data_cache" : False,
    "recreate_data_cache" : True,
    "data_download_n_jobs" : 20,

}

openml_data_object, data_id, all_dataset_metadata = get_all_dataset_metadata_from_openml(recreate_cache=config["recreate_data_cache"], n_jobs=config["data_download_n_jobs"])
descriptions = [
    desc.description if hasattr(desc, 'description') else "" 
for desc in openml_data_object
]
dataset_id = [
    desc.dataset_id if hasattr(desc, 'dataset_id') else "" 
for desc in openml_data_object
]

joined_qualities = [
    " ".join([f"{k} : {v}," for k, v in desc.qualities.items()]) if hasattr(desc, 'qualities') else "" 
    for desc in openml_data_object
]

joined_features = [
    " ".join([f"{k} : {v}," for k, v in desc.features.items()]) if hasattr(desc, 'features') else "" 
    for desc in openml_data_object
]

data_dict = {
    "did": dataset_id,
    "description": descriptions,
    "qualities": joined_qualities,
    "features": joined_features
}

metadata_df, all_dataset_metadata = create_metadata_dataframe(*get_all_dataset_metadata_from_openml(recreate_cache= config["recreate_data_cache"], n_jobs=config["data_download_n_jobs"]))
metadata_df = clean_metadata_dataframe(metadata_df)
print(metadata_df.head(), metadata_df.shape)

vectordb = load_document_and_create_vector_store(metadata_df, model_name=config['embedding_model'], recreate_chroma=config['recreate_chroma'], persist_directory=config['persist_dir'])
retriever, llm = create_retriever_and_llm(vectordb,num_return_documents=config["num_return_documents"], model_repo_id=config["llm_model"])
qa = create_llm_chain_and_query(vectordb=vectordb,retriever=retriever,llm=llm, prompt_template = config["rqa_prompt_template"])
test_data = openml.datasets.get_dataset(
        'credit-g',
        download_data=False,
        download_qualities=True,
        download_features_meta_data=True,
        # force_refresh_cache=True,
    )
test_data.id
## Getting results
# %time
query = "Which datasets would be useful for stock market support?"
# query = "Which datasets would be useful for heart disease"
# query = "Which datasets would be useful for flowers"
# query = "Which datasets would be useful for image classification"
# query = "My supervisor wants me to work on cloud cover, which datasets can I use"
# query = "Are there any datasets from the netherlands?"
# query = "Are there any datasets about farm animals?"
# query = "Find chinese authors"

if args.query is not None:
    query = args.query

results = create_result_dataframe(query, qa, all_dataset_metadata)
results
results['description'].values[:10]
results['name'].values[:10]