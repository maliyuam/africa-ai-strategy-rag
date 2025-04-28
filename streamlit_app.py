import streamlit as st
import os
import pymongo
from pymongo import MongoClient
# Removed fitz, io, numpy, PIL.Image, easyocr, tempfile, text_splitter, Document imports as they are for ingestion
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
# Removed RetrievalQA import
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder # Still needed for re-ranking
from dotenv import load_dotenv
import pandas # Often useful with Streamlit
import uuid # To generate unique keys for messages

# --- Configuration & Secrets ---
load_dotenv()

st.set_page_config(page_title="African AI Strategies Chat", layout="wide")
st.title("💬 African AI & ICT Strategies Chatbot")
st.caption("Chat about AI/ICT strategies (Re-ranking Accuracy | DB updated via Colab)") # Updated caption

# Secrets Management
try:
    OPENAI_API_KEY = str(st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    MONGO_URI = str(st.secrets.get("MONGO_URI", os.environ.get("MONGO_URI", "")))
    if not OPENAI_API_KEY or not MONGO_URI:
        st.error("🚨 OpenAI API Key or MongoDB URI not found.")
        st.stop()
except Exception as e:
    st.error(f"Error accessing secrets: {e}"); st.stop()

# Configuration constants
DB_NAME = "african_ai_strategies"
COLLECTION_NAME = "strategy_documents"
VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
# Removed CHUNK constants as chunking is handled elsewhere
# Re-ranking constants
INITIAL_RETRIEVAL_K = 20
FINAL_RANKED_K = 5
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
# Removed OCR constants

# --- Caching Expensive Initializations ---

@st.cache_resource(show_spinner="Connecting to MongoDB...")
def get_mongo_client(uri):
    # (Unchanged)
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client
    except pymongo.errors.ConnectionFailure as e:
        st.error(f"MongoDB Connection Failure: {e}. Check URI/IP Access List/Network.")
        return None
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

@st.cache_resource(show_spinner="Initializing AI Components...")
def initialize_openai_langchain(api_key):
    # (Unchanged)
    try:
        if not api_key: raise ValueError("OpenAI API Key is empty.")
        llm = ChatOpenAI(openai_api_key=api_key, model_name=LLM_MODEL, temperature=0.1)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=EMBEDDING_MODEL)
        openai_client = OpenAI(api_key=api_key)
        return llm, embeddings, openai_client
    except Exception as e:
        st.error(f"Error initializing OpenAI components: {e}")
        return None, None, None

@st.cache_resource(show_spinner="Connecting to Vector Store...")
def get_vector_store(_mongo_client, _embeddings):
    # (Unchanged)
    if _mongo_client is None or _embeddings is None:
        st.error("Cannot initialize Vector Store: Dependencies missing.")
        return None
    try:
        db = _mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]
        vector_store = MongoDBAtlasVectorSearch(collection=collection, embedding=_embeddings, index_name=VECTOR_INDEX_NAME)
        try: # Health check
            vector_store.similarity_search("test query health check", k=1)
        except Exception as index_e:
            st.sidebar.warning(f"Vector Index '{VECTOR_INDEX_NAME}' check failed: {index_e}.")
        return vector_store
    except Exception as e:
        st.error(f"Error initializing MongoDBAtlasVectorSearch: {e}")
        return None

@st.cache_resource(show_spinner="Loading Re-ranking Model...")
def get_cross_encoder(model_name=CROSS_ENCODER_MODEL):
    # (Unchanged)
    try:
        model = CrossEncoder(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading CrossEncoder model '{model_name}': {e}")
        return None

# --- Removed EasyOCR Reader Initialization ---
# @st.cache_resource(...)
# def get_easyocr_reader(...): ...

# --- Initialize Clients ---
mongo_client = get_mongo_client(MONGO_URI)
llm, embeddings, openai_client = initialize_openai_langchain(OPENAI_API_KEY)
vector_store = get_vector_store(mongo_client, embeddings)
cross_encoder = get_cross_encoder()
# Removed easyocr_reader initialization

# --- Removed Helper Functions for Ingestion ---
# def extract_text_from_pdf_stream(...): ...
# def chunk_text(...): ...
# def process_and_store_pdf(...): ...

# --- Custom Prompt Template ---
# (Unchanged)
prompt_template = """
You are an AI assistant specialized in analyzing African AI and ICT strategy documents.
Use the following pieces of context retrieved from the documents to answer the question at the end.
Context:
{context}

Question: {question}

Answer the question based *ONLY* on the provided context above.
- If the context contains the answer, provide a clear and concise answer referencing the key information found.
- If the context does *not* contain information relevant to the question, explicitly state: "Based on the provided documents, I cannot answer this question."
- Do not add any information that is not present in the context. Do not make assumptions or provide external knowledge.

Answer:"""
CUSTOM_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- RAG Function (incorporates re-ranking) ---
# (Unchanged)
def get_rag_response(query):
    """Performs RAG, Re-ranking, and LLM call."""
    if not all([vector_store, llm, cross_encoder]):
        missing = [comp_name for comp, comp_name in zip([vector_store, llm, cross_encoder], ["Vector Store", "LLM", "Re-ranker"]) if comp is None]
        return f"Error: Cannot perform query. Missing components: {', '.join(missing)}.", []
    try:
        with st.spinner(f"Searching for top {INITIAL_RETRIEVAL_K} candidates..."):
            initial_docs = vector_store.similarity_search(query, k=INITIAL_RETRIEVAL_K)
        if not initial_docs:
            return "I couldn't find any potentially relevant documents based on your query.", []
        with st.spinner(f"Re-ranking {len(initial_docs)} candidates..."):
            rerank_pairs = [(query, doc.page_content) for doc in initial_docs]
            scores = cross_encoder.predict(rerank_pairs)
            docs_with_scores = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
            re_ranked_docs = [doc for doc, score in docs_with_scores[:FINAL_RANKED_K]]
        context_str = "\n\n---\n\n".join([doc.page_content for doc in re_ranked_docs])
        formatted_prompt = CUSTOM_PROMPT.format(context=context_str, question=query)
        with st.spinner("Generating answer..."):
            response = llm.invoke(formatted_prompt)
            answer = response.content
        return answer, re_ranked_docs
    except Exception as e:
        st.error(f"An error occurred during the RAG process: {e}"); st.exception(e)
        return "Sorry, an error occurred while processing your request.", []

# --- UI Components ---

# Sidebar Status and Config Display (Removed OCR status/config)
with st.sidebar:
    st.divider()
    st.subheader("System Status")
    status_ok = True
    if mongo_client: st.success("MongoDB Connected")
    else: st.error("MongoDB Disconnected"); status_ok = False
    if embeddings: st.success("Embeddings Initialized")
    else: st.error("Embeddings Failed"); status_ok = False
    if llm: st.success("LLM Initialized")
    else: st.error("LLM Failed"); status_ok = False
    if vector_store: st.success("Vector Store Ready")
    else: st.error("Vector Store Failed"); status_ok = False
    if cross_encoder: st.success("Re-ranker Loaded")
    else: st.error("Re-ranker Failed"); status_ok = False
    # Removed EasyOCR status check
    st.divider()
    st.subheader("Configuration")
    if embeddings: st.write(f"**Embedding:** `{EMBEDDING_MODEL}`")
    if llm: st.write(f"**LLM:** `{LLM_MODEL}`")
    if cross_encoder: st.write(f"**Re-ranker:** `{CROSS_ENCODER_MODEL}`")
    # Removed OCR info
    st.divider()
    st.info("Database is updated via external process (e.g., Colab notebook).") # Added info message

# --- Removed Sidebar PDF Upload Section ---
# with st.sidebar:
#    st.header("📄 Add New Document")
#    ... (all upload widgets and logic removed) ...


# --- Main Chat Interface ---
# (Unchanged from previous version)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources associated with previous assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Show sources used", key=f"expander_{message['id']}"):
                for i, doc in enumerate(message["sources"]):
                    source = doc.metadata.get('source', 'N/A')
                    country = doc.metadata.get('country', 'N/A')
                    year = doc.metadata.get('year', '')
                    chunk_idx = doc.metadata.get('chunk_index', '')
                    label = f"Source {i+1} | Src: {source} | Ctry: {country} | Yr: {year} | Idx: {chunk_idx}"
                    source_key = f"src_hist_{message['id']}_{i}"
                    st.text_area(label, doc.page_content, height=100, key=source_key)


# Get user input using chat_input
if prompt := st.chat_input("Ask a question about the documents..." if status_ok else "System not fully ready..."):
    if not status_ok:
        st.warning("Please wait for all system components to initialize (check sidebar).")
    else:
        # Add user message to history and display it
        user_msg_id = str(uuid.uuid4())
        st.session_state.messages.append({"role": "user", "content": prompt, "id": user_msg_id})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")
            answer, sources = get_rag_response(prompt) # RAG function call
            response_placeholder.markdown(answer)

            # Display sources for *this* response in an expander
            if sources:
                # Use a unique key for the expander based on the latest message count
                expander_key = f"expander_curr_{len(st.session_state.messages)}"
                with st.expander("Show sources used for this response", key=expander_key):
                    for i, doc in enumerate(sources):
                         source = doc.metadata.get('source', 'N/A')
                         country = doc.metadata.get('country', 'N/A')
                         year = doc.metadata.get('year', '')
                         chunk_idx = doc.metadata.get('chunk_index', '')
                         label = f"Source {i+1} | Src: {source} | Ctry: {country} | Yr: {year} | Idx: {chunk_idx}"
                         source_key = f"src_curr_{len(st.session_state.messages)}_{i}"
                         st.text_area(label, doc.page_content, height=100, key=source_key)

        # Add assistant response and sources to history
        asst_msg_id = str(uuid.uuid4())
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "id": asst_msg_id
            })