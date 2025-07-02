import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_file(file_path: str):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
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
    
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    docs = text_splitter.split_documents(documents)
    return docs