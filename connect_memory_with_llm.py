"""
LangChain + HuggingFace + FAISS Example
---------------------------------------
This script sets up a Retrieval-based Question Answering (QA) system using:
- A HuggingFace-hosted LLM (Mistral model)
- FAISS vector database for document retrieval
- Custom prompt template for better responses
"""

# Import required libraries
import os

# HuggingFace integration for LLMs and embeddings
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

# LangChain utilities
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Vector store (FAISS)
from langchain_community.vectorstores import FAISS

# Optional: load environment variables (if not using pipenv/poetry)
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())


# =========================================================
# STEP 1: Setup LLM (Large Language Model) via HuggingFace
# =========================================================

# Grab the HuggingFace API token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")

# Define which HuggingFace model to use (here: Mistral Instruct model)
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    """
    Initialize a HuggingFace LLM endpoint.
    Arguments: huggingface_repo_id (str): The HuggingFace model repository ID
    Returns: HuggingFaceEndpoint: LLM object ready for inference
    """
    llm = HuggingFaceEndpoint(
    repo_id=huggingface_repo_id,
    task="conversational",
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512   # <-- moved out of model_kwargs
    )
    return llm

# =========================================================
# STEP 2: Define Custom Prompt Template
# =========================================================

# This ensures the LLM sticks to retrieved context and avoids hallucinations
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say you don't know. 
Do not make up an answer. 
Do not provide information outside the given context.

Context: {context}
Question: {question}

Answer:
"""

CUSTOM_PROMPT_TEMPLATE = """
            You are an AI business analyst specializing in helping SMEs unlock growth through
            data-driven decisions, predictive analytics, and AI-powered solutions. 

            The SME has provided operational documents (sales logs, stock reports, and invoices). 
            Your role is to analyze this context and answer their question in a way that helps them 
            make better business decisions.

            Context: {context}
            Question: {question}

            When answering:
            - Identify trends, risks, or opportunities hidden in the data.
            - Always link insights to business impact (e.g. sales growth, stockouts avoided, 
            cost savings, improved cash flow, or customer retention).
            - Where useful, suggest predictive/AI-driven approaches 
            (e.g. demand forecasting, churn prediction, anomaly detection).
            - Give actionable, plain-language recommendations that a small business owner can apply.
            - If data is missing, be explicit: recommend what data would help (e.g. more 
            detailed customer segmentation, historical records, seasonal info).
            - Keep the answer concise, practical, and focused on SME decision-making.

            Answer:
"""

def set_custom_prompt(custom_prompt_template):
    """
    Creates a PromptTemplate object from the provided template.
    Arguments: custom_prompt_template (str): Template string with placeholders
    Returns: PromptTemplate: LangChain prompt template object
    """
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# =========================================================
# STEP 3: Load Vector Database (FAISS)
# =========================================================

# Path to stored FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"

# Embedding model used for vector search (Sentence Transformer)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index from local storage
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# =========================================================
# STEP 4: Build Retrieval-based QA Chain
# =========================================================

# Create RetrievalQA chain that connects:
#   - LLM
#   - FAISS retriever (fetches top-k relevant documents)
#   - Custom prompt template
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",  # "stuff" means all retrieved docs are passed into the prompt
    retriever=db.as_retriever(search_kwargs={'k': 3}),  # retrieve top 3 documents
    return_source_documents=True,  # include docs used in answer
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


# =========================================================
# STEP 5: Run Query
# =========================================================

# Take user input
user_query = input("Write Query Here: ")

# Invoke QA chain with the user query
response = qa_chain.invoke({'query': user_query})

# Print results
print("RESULT: ", response["result"])
print("\nSOURCE DOCUMENTS: ", response["source_documents"])