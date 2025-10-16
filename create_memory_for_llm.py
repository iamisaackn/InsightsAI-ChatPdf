# ---------------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------------

# Loaders for reading PDF files
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Utility for splitting long texts into smaller, overlapping chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding model from HuggingFace (turns text into vectors)
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS vector store (used to index and search embeddings efficiently)
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------
# STEP 1: Load Raw PDF(s)
# ---------------------------------------------------------
# - DirectoryLoader scans a folder for PDF files
# - Each PDF is processed page by page using PyPDFLoader
# - Returns a list of "Document" objects that contain text + metadata
# ---------------------------------------------------------

DATA_PATH = "data/"  # Folder containing PDF files

def load_pdf_files(data_path: str):
    loader = DirectoryLoader(
        data_path,          # directory containing PDFs
        glob="*.pdf",       # match only PDF files
        loader_cls=PyPDFLoader  # use PyPDFLoader to read PDFs
    )
    documents = loader.load()  # Load and return all PDF pages
    return documents

documents = load_pdf_files(DATA_PATH)
print("Number of PDF pages loaded:", len(documents))


# ---------------------------------------------------------
# STEP 2: Split Documents into Chunks
# ---------------------------------------------------------
# - Large PDF texts can exceed model limits → must be chunked
# - RecursiveCharacterTextSplitter splits text into overlapping chunks
#   * chunk_size=500 → each chunk has max 500 characters
#   * chunk_overlap=50 → chunks overlap by 50 characters (keeps context)
# - Output: list of smaller, more manageable text chunks
# ---------------------------------------------------------

def create_chunks(extracted_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # max characters per chunk
        chunk_overlap=50   # overlap between chunks
    )
    text_chunks = text_splitter.split_documents(extracted_docs)
    return text_chunks

text_chunks = create_chunks(documents)
print("Number of text chunks created:", len(text_chunks))


# ---------------------------------------------------------
# STEP 3: Create Embedding Model
# ---------------------------------------------------------
# - Embeddings = numerical representation of text
# - HuggingFaceEmbeddings uses a pretrained transformer model
# - Model: "all-MiniLM-L6-v2" → small, fast, high-quality sentence embeddings
# ---------------------------------------------------------

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

embedding_model = get_embedding_model()


# ---------------------------------------------------------
# STEP 4: Store Embeddings in FAISS Vector Database
# ---------------------------------------------------------
# - FAISS = library for efficient similarity search on embeddings
# - We create a FAISS index from the text chunks + embeddings
# - Then we save the FAISS index locally for reuse
# ---------------------------------------------------------

DB_FAISS_PATH = "vectorstore/db_faiss"

# Build FAISS index from documents
db = FAISS.from_documents(text_chunks, embedding_model)

# Save FAISS index locally so you don’t have to rebuild every time
db.save_local(DB_FAISS_PATH)

print("FAISS vector store saved at:", DB_FAISS_PATH)
