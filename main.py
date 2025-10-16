# Importing required libraries
import os
import streamlit as st

# LangChain HuggingFace embedding models (for vectorization)
from langchain_huggingface import HuggingFaceEmbeddings
# RetrievalQA allows us to ask questions based on retrieved context
from langchain.chains import RetrievalQA
# FAISS is the vector database we use to store & search embeddings
from langchain_community.vectorstores import FAISS
# PromptTemplate allows us to define custom instructions for the model
from langchain_core.prompts import PromptTemplate
# For calling HuggingFace endpoints (optional, shown for completeness)
from langchain_huggingface import HuggingFaceEndpoint
# Groq LLM integration (free hosted LLaMA/Mistral models)
from langchain_groq import ChatGroq

# Optional: load environment variables from .env if not using pipenv
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

# Path to pre-built FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"


# -------------------------------------------------------------------
# Function: get_vectorstore
# Loads the FAISS vector store with pre-computed embeddings
# Cached by Streamlit to avoid reloading every time
# -------------------------------------------------------------------
@st.cache_resource
def get_vectorstore():
    # Define embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    # Load FAISS index from local folder
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True  # required for some FAISS configs
    )
    
    return db


# -------------------------------------------------------------------
# Function: set_custom_prompt
# Creates a structured prompt template for the LLM
# -------------------------------------------------------------------
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]  # must match placeholders in template
    )
    return prompt


# -------------------------------------------------------------------
# Function: load_llm
# Example function to use HuggingFace LLM endpoint (not active here)
# -------------------------------------------------------------------
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )
    return llm


# -------------------------------------------------------------------
# Main Streamlit app
# -------------------------------------------------------------------
def main():
    # -------------------- PAGE CONFIG --------------------
    st.set_page_config(
        page_title="InsightAI",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # -------------------- SIDEBAR --------------------
    with st.sidebar:
        st.header("💡 InsightAI")
        st.markdown(
            "Your AI assistant for extracting actionable business insights from your documents."
        )
        if st.button("Reset Chat"):
            st.session_state.messages = []
    
    # -------------------- HEADER --------------------
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Ask InsightAI!</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # -------------------- CHAT HISTORY --------------------
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message("user").markdown(message['content'])
        else:
            st.chat_message("assistant").markdown(message['content'])

    # -------------------- USER INPUT --------------------
    prompt = st.chat_input("Type your business question here...")

    if prompt:
        # Show user’s message in chat
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Custom system prompt to control chatbot behavior
        CUSTOM_PROMPT_TEMPLATE = """
        You are an AI business analyst. Use only the information provided in the context to answer the user's question. 
        Do not guess or fabricate data. If the answer is not in the context, say "I don't know."

        The user may also greet you or make casual conversation. Respond politely and briefly, then guide the conversation back to business analysis.

        The user has shared operational documents (e.g. sales logs, stock reports, invoices). 
        Your job is to extract qualitative and quantitative insights that help the business owner make smarter decisions.

        Context: {context}
        Question: {question}

        Answering guidelines:
        - Be brief, clear, and to the point.
        - Focus on trends, risks, or opportunities in the data.
        - If data is missing, say so clearly and recommend what would help.
        - Avoid jargon. Keep it simple and decision-focused.
        - If the user greets or chats casually, respond warmly but briefly, then return to the business context.
        - When the answer requires a chart or a user requests a chart and the data includes quantities, always generate a **vertical bar chart displayed strictly as a table** using stacked emojis.
        - Each category must be represented as a **separate column**, with emojis stacked **vertically** to indicate quantity.
        - Sort the columns based on the values, either from **highest to lowest** or **lowest to highest**, depending on the user's preference.
        - Label each column clearly with the **category name** and its **numerical value**.
        - Use a **distinct colored square emoji** (e.g. 🟩, 🟥, 🟦, 🟧, 🟨, 🟪, 🟫) for each category to visually differentiate them.
        - The chart must be aligned **vertically**, meaning:
        - Each column is stacked from top to bottom.
        - Categories appear **one below the other**, not side by side.
        - This layout ensures the chart remains easy to read and visually consistent.

        
        """

        try:
            # Load FAISS vectorstore
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            # Build RetrievalQA pipeline with Groq LLaMA model
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free Groq model
                    temperature=0.0,  # deterministic responses
                    groq_api_key=os.environ["GROQ_API_KEY"],  # must be set in environment
                ),
                chain_type="stuff",  # simplest chain type: stuff retrieved docs into prompt
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),  # retrieve top-3 docs
                return_source_documents=True,  # also return retrieved docs
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}  # use our custom prompt
            )

            # Run the chain with the user query
            response = qa_chain.invoke({'query': prompt})

            # Extract results
            result = response["result"]
            source_documents = response["source_documents"]

            # Display result + sources in chat
            # Display result only (no sources)
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
