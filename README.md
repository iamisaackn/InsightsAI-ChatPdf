# PROJECT LAYOUT
## Phase 1 - Setup Memory for LLM (Vector Database)
- Load raw PDFs
- Create chunks
- Create Vector Embeddings
- Store embeddings in FAISS

## Phase 2 - Connect Memory with LLM
- Setup LLM (Mistral with HuggingFace)
- Connect LLM with FAISS
- Create chain

## Phase 3 - Setup UI for the Chatbot
- Chatbot with Streamlit
- Load Vector store (FAISS) in cache
- Rerieval Augmented Generation (RAG)

## Tools & Technologies
- Langchain (AI Framework for LLM applications)
- HuggingFace (ML/AI hub)
- Mistral (LLM Model)
- FAISS (Vector Database)
- Streamlit (For Chatbot UI)
- Python (Programming Language)
- VS Code (IDE)

## Setting Up Your Environment with Pipenv
## Prerequisite: Install Pipenv
Follow the official Pipenv installation guide to set up Pipenv on your system:  
[Install Pipenv Documentation](https://pipenv.pypa.io/en/latest/installation.html)

## Steps to Set Up the Environment

### Install Required Packages
Run the following commands in your terminal (assuming Pipenv is already installed):

```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install streamlit