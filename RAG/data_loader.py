import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader

def load_from_url(url: str) -> List[Document]:
    if not url or not url.startswith(('http://', 'https://')):
        print(f"Invalid URL provided: {url}")
        return []
    print(f"Loading documents from URL: {url}")
    loader = WebBaseLoader(web_paths=(url,))
    try:
        docs = loader.load()
        print(f"Loaded {len(docs)} document(s) from URL.")
        return docs
    except Exception as e:
        print(f"Error loading URL {url}: {e}")
        return []

def load_from_directory(directory_path: str, glob_pattern: str = "**/*") -> List[Document]:
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return []
    print(f"Loading documents from directory: {directory_path} (pattern: {glob_pattern})")
 
    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        show_progress=True,
        use_multithreading=True,
    )
