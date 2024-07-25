import sqlalchemy
import pandas as pd
import chromadb
from langchain_community.vectorstores.chroma import Chroma
from tqdm.auto import tqdm

import sys

sys.path.append("../")
sys.path.append("../backend/")
from backend.modules.utils import load_config_and_device

config = load_config_and_device("../backend/config.json")

# load the persistent database using ChromaDB
client = chromadb.PersistentClient(path=config["chroma_metadata_dir"])

collec = client.get_or_create_collection(name = "metadata")

metadata_df = pd.read_csv("../data/all_dataset_description.csv")
metadata_df = metadata_df.drop(columns=["Combined_information"])

# Function to chunk the DataFrame
def chunk_dataframe(df, chunk_size):
    for i in range(0, df.shape[0], chunk_size):
        yield df.iloc[i : i + chunk_size]

def load_chroma_metadata():        
    # Define the chunk size
    chunk_size = config['chunk_size']  # Adjust the chunk size as needed

    # Process each chunk
    for chunk in tqdm(
        chunk_dataframe(metadata_df, chunk_size), total=(len(metadata_df) // chunk_size) + 1
    ):
        ids = chunk["did"].astype(str).tolist()
        documents = chunk["description"].astype(str).tolist()
        metadatas = chunk.to_dict(orient="records")

        # Add to ChromaDB collection
        if collec.get(ids=ids) == []:
            collec.add(ids=ids, documents=documents, metadatas=metadatas)
        
    return collec
