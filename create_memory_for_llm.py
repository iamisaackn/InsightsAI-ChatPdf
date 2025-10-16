import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Author: Isaac Kinyanjui Ngugi

# Load environment variables
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/")
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vectorstore/db_faiss")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_pdfs(path: str):
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def build_faiss_index(chunks, model_name: str, save_path: str):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_path)
    return db

if __name__ == "__main__":
    docs = load_pdfs(DATA_PATH)
    print(f"PDF pages loaded: {len(docs)}")

    chunks = split_docs(docs)
    print(f"Text chunks created: {len(chunks)}")

    db = build_faiss_index(chunks, EMBEDDING_MODEL, DB_FAISS_PATH)
    print(f"FAISS vector store saved at: {DB_FAISS_PATH}")