from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
import os

def get_retriever(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return vectorstore, retriever

def get_qa_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and to the point.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain