import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader
)
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_file(file_path: str):
    # Get the file extension in lowercase
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    # Select appropriate loader based on file type
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.csv':
        loader = CSVLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    # Load the raw documents
    documents = loader.load()
    
    # Split the documents using recursive strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Split and return
    docs = text_splitter.split_documents(documents)
    return docs

# Use the function
if __name__ == "__main__":
    try:
        path = "your_file_path_here.pdf"  # Replace with actual path
        chunks = process_file(path)
        print(f"Processed {len(chunks)} document chunks.")
    except Exception as e:
        print(f"Error processing file: {e}")

def process_web_page(url: str):
    # Create a web loader for the given URL
    loader = WebBaseLoader(url)
    
    # Load the raw documents from the web page
    documents = loader.load()
    
    # Split the documents using recursive strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Split and return
    docs = text_splitter.split_documents(documents)
    return docs
