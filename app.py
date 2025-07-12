import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print(embeddings.embed_query("hello world"))
# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure folders exist
Path("uploads").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

st.set_page_config(page_title="RAG with Gemini AI", page_icon="📄", layout="wide")

st.title("📄 RAG Document Chatbot using Gemini AI")
st.markdown("Upload documents (PDF, CSV, TXT, XLSX) and ask questions. Gemini AI will answer using RAG.")

# Global variables
vectorstore = None
retriever = None
qa_chain = None


def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def get_retriever(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return vectorstore, retriever


def get_qa_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    template = """Use the following context to answer the question. 
If you don't know, say you don't know. Be concise.
    
Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# Upload section
with st.sidebar:
    st.header("📤 Upload Your File")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "csv", "txt", "xlsx", "xls"])
    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"{uploaded_file.name} uploaded!")

        try:
            docs = process_file(file_path)
            vectorstore, retriever = get_retriever(docs)
            qa_chain = get_qa_chain(retriever)
            st.success("File processed and embeddings created!")
        except Exception as e:
            st.error(f"Error: {e}")

# Query section
if qa_chain:
    st.subheader("💬 Ask a Question")
    query = st.text_input("Type your question:")
    if st.button("Get Answer") and query:
        try:
            result = qa_chain.invoke({"query": query})
            st.success(result)
        except Exception as e:
            st.error(f"Failed to answer: {e}")
else:
    st.info("Please upload a file to begin.")

