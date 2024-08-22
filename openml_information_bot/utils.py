import requests
from bs4 import BeautifulSoup
import csv
import os
import pandas as pd
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from urllib.parse import urljoin
from tqdm.auto import tqdm


class Crawler:
    """
    Description: This class is used to crawl the OpenML website and gather both code and general information for a bot.
    """

    def __init__(
        self,
        crawled_files_data_path,
        recrawl_websites=False,
        num_of_websites_to_crawl=None,
    ):
        self.base_urls = [
            "https://openml.github.io/openml-python/main/",
            "https://docs.openml.org/",
        ]
        self.crawled_files_data_path = crawled_files_data_path
        self.recrawl_websites = recrawl_websites
        self.num_of_websites_to_crawl = num_of_websites_to_crawl
        self.crawl_count = 0
        self.visited = set()
        self.data_queue = []

    def extract_text_from_tags(self, soup, tags):
        """Extract and return the concatenated text from all given tags."""
        return {
            tag: " ".join(
                element.get_text(strip=True) for element in soup.find_all(tag)
            )
            for tag in tags
        }

    def fetch_soup(self, url):
        """Fetch and return a BeautifulSoup object for the given URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return None

    def extract_data(self, url):
        """Extract and return relevant data from the given URL."""
        soup = self.fetch_soup(url)
        if not soup:
            return None

        tags_to_extract = ["h1", "h2", "h3", "h4"]
        header_data = self.extract_text_from_tags(soup, tags_to_extract)

        return {
            "url": url,
            "title": soup.title.string if soup.title else "No title",
            "body_text": (
                soup.body.get_text(separator=" ", strip=True)
                if soup.body
                else "No body text"
            ),
            "header_links_text": " ".join(
                link.get_text(strip=True) for link in soup.find_all("a", href=True)
            ),
            **header_data,
        }

    def save_data(self, writer):
        """Save all extracted data in the queue to the CSV file."""
        for data in self.data_queue:
            writer.writerow(
                [
                    data["url"],
                    data["body_text"],
                    data["header_links_text"],
                    data["h1"],
                    data["h2"],
                    data["h3"],
                    data["h4"],
                    data["title"],
                ]
            )

    def crawl(self, url, progress_bar):
        """Crawl the given URL and its linked pages."""
        try:
            if url in self.visited or (self.num_of_websites_to_crawl and self.crawl_count >= self.num_of_websites_to_crawl):
                return

            self.visited.add(url)
            self.crawl_count += 1
            progress_bar.update(1)  # Update progress bar

            data = self.extract_data(url)
            if data:
                self.data_queue.append(data)

            soup = self.fetch_soup(url)
            if soup:
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(url, link["href"])
                    if any(full_url.startswith(base_url) for base_url in self.base_urls):
                        self.crawl(full_url, progress_bar)
        except RecursionError:
            print(f"Recursion error while crawling {url}")

    def do_crawl(self):
        """Manage the entire crawling and saving process with a progress bar."""
        if not self.recrawl_websites and os.path.exists(self.crawled_files_data_path):
            print("Data already exists. Set recrawl_websites=True to recrawl.")
            return

        os.makedirs(os.path.dirname(self.crawled_files_data_path), exist_ok=True)

        print("Crawling the websites...")

        # the progress bar is not accurate because we don't know the total number of URLs to crawl. this is just to see if the script is running or not

        total_urls = self.num_of_websites_to_crawl or len(self.base_urls)  # Estimate total URLs for progress bar
        with tqdm(total=total_urls, desc="Crawling URLs") as progress_bar:
            for start_url in self.base_urls:
                self.crawl(start_url, progress_bar)

        with open(self.crawled_files_data_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["URL", "Body Text", "Header Links Text", "H1", "H2", "H3", "H4", "Title"])

            self.save_data(writer)

        print("Crawling complete.")
    
class ChromaStore:
    def __init__(self, model_name, crawled_files_data_path, chroma_file_path) -> None:
        self.model_name = model_name
        self.device = self._find_device()
        self.hf_embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.crawled_files_data_path = crawled_files_data_path
        self.chroma_file_path = chroma_file_path

    def _find_device(self) -> str:
        """
        Determines the best available device: 'cuda', 'mps', or 'cpu'.
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def read_data_and_embed(self):
        if not os.path.exists(self.crawled_files_data_path):
            print("Crawled data does not exist. Please run the crawler first.")
            return

        df = pd.read_csv(self.crawled_files_data_path)
        df["joined"] = df.apply(self._join_columns, axis=1)
        docs = DataFrameLoader(df, page_content_column="joined").load()

        # Splitting the document texts into smaller chunks
        docs_texts = self._split_documents(docs)

        # Convert metadata values to strings
        for doc in docs_texts:
            doc.metadata = {k: str(v) for k, v in doc.metadata.items()}

        print("Creating the vector store")
        Chroma.from_documents(
            documents=docs_texts,
            embedding=self.hf_embeddings,
            persist_directory=self.chroma_file_path,
        )

    def _join_columns(self, row) -> str:
        """
        Joins specified columns of a row into a single string.
        """
        columns = ['URL', 'Body Text', 'Header Links Text', 'H1', 'H2', 'H3', 'H4', 'Title']
        return ', '.join(f"{col.lower().replace(' ', '_')}: {row[col]}" for col in columns)

    def _split_documents(self, docs):
        """
        Splits documents into chunks using RecursiveCharacterTextSplitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )
        return splitter.split_documents(docs)