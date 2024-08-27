import os

from documentation_query_utils import ChromaStore, Crawler

recrawl_websites = True

crawled_files_data_path = "../data/crawler/crawled_data.csv"
chroma_path = "../data/crawler/"
model_name = "BAAI/bge-small-en"
generation_model_name = "llama3"  # ollama

# Crawl the websites and save the data
num_of_websites_to_crawl = None  # none for all

if not os.path.exists(chroma_path):
    os.makedirs(chroma_path, exist_ok=True)

# Crawl the websites and save the data
crawler = Crawler(
    crawled_files_data_path=crawled_files_data_path,
    recrawl_websites=recrawl_websites,
    num_of_websites_to_crawl=num_of_websites_to_crawl,
)
crawler.do_crawl()

# Initialize the ChromaStore and embed the data
chroma_store = ChromaStore(
    model_name=model_name,
    crawled_files_data_path=crawled_files_data_path,
    chroma_file_path=chroma_path,
    generation_model_name=generation_model_name,
)
if recrawl_websites == True:
    chroma_store.read_data_and_embed()
