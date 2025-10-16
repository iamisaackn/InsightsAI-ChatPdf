"""
Retrieval-based Question Answering System
-----------------------------------------
Author: Isaac Kinyanjui Ngugi

This script uses:
- HuggingFace LLM (Mistral-7B-Instruct)
- FAISS vector store for document retrieval
- LangChain for question answering with a custom prompt
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# --- LLM Setup ---
def load_llm(repo_id: str):
    """Initialize HuggingFace model."""
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )

# --- Custom Prompt ---
CUSTOM_PROMPT = """
You are an AI business analyst helping SMEs make data-driven decisions.
Use the provided context to answer the question.

Context: {context}
Question: {question}

Guidelines:
- Focus on trends, risks, and opportunities.
- Link insights to business impact (e.g. sales growth, cost savings).
- Suggest predictive or AI-driven approaches when useful.
- Give actionable, practical recommendations.
- If data is missing, mention what data would help.
- Keep it concise and relevant to SME decision-making.

Answer:
"""

def build_prompt(template: str):
    """Create LangChain prompt template."""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- Load Vector Store ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --- Create Retrieval QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": build_prompt(CUSTOM_PROMPT)}
)

# --- Run Query ---
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    response = qa_chain.invoke({"query": user_query})
    print("\nAnswer:\n", response["result"])
    print("\nSource Documents:\n", response["source_documents"])