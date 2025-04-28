import streamlit as st
import os
import pymongo
from pymongo import MongoClient
import fitz  # PyMuPDF
import io # For handling image bytes
import numpy as np # For image array conversion
from PIL import Image # For image handling
import easyocr # Import EasyOCR
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder # Import for re-ranking
from dotenv import load_dotenv
import tempfile
import pandas
import uuid # To generate unique keys for messages

# --- Configuration & Secrets ---
load_dotenv()

st.set_page_config(page_title="African AI Strategies Chat", layout="wide")
st.title("💬 African AI & ICT Strategies Chatbot")
st.caption("Chat about AI/ICT strategies (Handles Scanned PDFs | Re-ranking Accuracy | DB updated via Colab)")

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
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# Re-ranking constants
INITIAL_RETRIEVAL_K = 20
FINAL_RANKED_K = 5
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
# OCR constants
OCR_LANGUAGES = ['en']
OCR_MIN_TEXT_LENGTH = 50

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
    """Loads the CrossEncoder model and caches it."""
    try:
        model = CrossEncoder(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading CrossEncoder model '{model_name}': {e}")
        return None

@st.cache_resource(show_spinner="Loading OCR Model...")
def get_easyocr_reader(langs=OCR_LANGUAGES, gpu=False):
    # (Unchanged - kept for completeness if needed later)
    """Loads the EasyOCR reader and caches it."""
    try:
        reader = easyocr.Reader(langs, gpu=gpu)
        return reader
    except Exception as e:
        st.error(f"Error loading EasyOCR Reader: {e}")
        return None

# --- Initialize Clients ---
mongo_client = get_mongo_client(MONGO_URI)
llm, embeddings, openai_client = initialize_openai_langchain(OPENAI_API_KEY)
vector_store = get_vector_store(mongo_client, embeddings)
cross_encoder = get_cross_encoder()
easyocr_reader = get_easyocr_reader() # Initialize even if not used in UI

# --- Helper Functions ---
# (Functions extract_text_from_pdf_stream, chunk_text, process_and_store_pdf
#  are kept here, even though not called by the UI,
#  in case you want to re-enable uploads later or use them differently)
# ... (copy the full helper functions from two responses ago here) ...
def extract_text_from_pdf_stream(pdf_stream):
    """
    Extracts text from a PDF file stream.
    Uses PyMuPDF for text extraction and falls back to EasyOCR for image-based pages.
    """
    full_doc_text = ""
    ocr_pages_count = 0
    tmp_pdf_path = None

    if easyocr_reader is None:
        st.warning("EasyOCR reader not loaded. OCR functionality will be skipped.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_stream.read())
            tmp_pdf_path = tmpfile.name

        doc = fitz.open(tmp_pdf_path)
        if len(doc) == 0:
             st.warning("Uploaded PDF appears to be empty.")
             return ""

        page_texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text").strip()

            if easyocr_reader and (not page_text or len(page_text) < OCR_MIN_TEXT_LENGTH):
                ocr_attempted_for_page = False
                try:
                    # st.write(f"Page {page_num+1}: Low text detected, attempting OCR...") # Debug info removed for cleaner output
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    np_image = np.array(pil_image)
                    ocr_results = easyocr_reader.readtext(np_image)
                    ocr_attempted_for_page = True
                    ocr_text = " ".join([res[1] for res in ocr_results])
                    if ocr_text.strip():
                        page_texts.append(ocr_text.strip())
                        ocr_pages_count += 1
                        # st.write(f"Page {page_num+1}: OCR successful.") # Debug info removed
                    elif page_text:
                        page_texts.append(page_text)
                except Exception as ocr_err:
                     st.warning(f"OCR failed for page {page_num + 1}: {ocr_err}. Falling back to get_text() if available.")
                     if page_text:
                         page_texts.append(page_text)
            else:
                page_texts.append(page_text)

        doc.close()
        os.remove(tmp_pdf_path)

        full_doc_text = "\n\n".join(filter(None, page_texts))

        if ocr_pages_count > 0:
            st.info(f"Used OCR to extract text from {ocr_pages_count} page(s).")

        if not full_doc_text.strip():
             st.warning("Extracted text is empty or only whitespace after processing.")
             return ""
        return full_doc_text

    except fitz.fitz.FileDataError:
        st.error("Error opening PDF: File may be corrupted or password-protected.")
        return None
    except Exception as e:
        st.error(f"Error extracting text from PDF stream: {e}")
        return None
    finally:
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            try: os.remove(tmp_pdf_path)
            except OSError: pass

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits text into manageable chunks."""
    if not text or not text.strip(): return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error during text chunking: {e}")
        return []

def process_and_store_pdf(pdf_stream, filename, country, year=None, source_url=None):
    """Processes an uploaded PDF stream and stores it."""
    # This function is currently unused by the UI but kept for completeness
    if not vector_store:
        st.error("Vector Store not initialized. Cannot process PDF.")
        return False

    status_placeholder = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0, text="Starting...")
    # ... (rest of the function remains the same as previously) ...
    try:
        status_placeholder.info(f"Processing '{filename}'...")
        progress_bar.progress(5, text="Extracting text (incl. OCR)...")
        with st.spinner("Extracting text (incl. OCR)..."):
            full_text = extract_text_from_pdf_stream(pdf_stream)

        if full_text is None:
            status_placeholder.error("Text extraction failed."); progress_bar.empty(); return False
        if not full_text:
             st.warning(f"No text found in '{filename}'. Skipping storage.")
             status_placeholder.empty(); progress_bar.empty(); return False

        progress_bar.progress(20, text="Chunking text...")
        text_chunks = chunk_text(full_text)
        if not text_chunks:
            st.error("No text chunks generated."); status_placeholder.empty(); progress_bar.empty(); return False

        progress_bar.progress(30, text="Preparing documents...")
        documents = []
        for i, chunk in enumerate(text_chunks):
            metadata = { "source": filename, "country": country.strip(), "chunk_index": i, "year": year, "source_url": source_url }
            filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
            documents.append(Document(page_content=chunk, metadata=filtered_metadata))

        if not documents:
            st.error("No documents created."); status_placeholder.empty(); progress_bar.empty(); return False

        progress_bar.progress(40, text=f"Embedding & storing {len(documents)} chunks...")
        with st.spinner(f"Adding {len(documents)} chunks to Vector Store..."):
            batch_size = 50
            inserted_ids = []
            total_docs = len(documents)
            for i in range(0, total_docs, batch_size):
                 batch = documents[i:i + batch_size]
                 ids = vector_store.add_documents(batch)
                 inserted_ids.extend(ids)
                 progress_percentage = min(40 + int(((i + len(batch)) / total_docs) * 60), 100)
                 progress_text = f"Embedding & Storing Chunks... {progress_percentage}%"
                 progress_bar.progress(progress_percentage / 100, text=progress_text)

        st.sidebar.success(f"Processed '{filename}' ({len(inserted_ids)} chunks).")
        progress_bar.empty(); status_placeholder.empty()
        st.spinner()
        return True

    except Exception as e:
        st.error(f"Error processing/storing PDF: {e}"); st.exception(e)
        status_placeholder.empty()
        if 'progress_bar' in locals(): progress_bar.empty()
        st.spinner()
        return False

# --- Custom Prompt Template ---
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

# Sidebar Status and Config Display
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
    if easyocr_reader: st.success("EasyOCR Reader Loaded") # Kept status for info
    else: st.warning("EasyOCR Reader Failed / Not Used") # Changed to warning
    st.divider()
    st.subheader("Configuration")
    if embeddings: st.write(f"**Embedding:** `{EMBEDDING_MODEL}`")
    if llm: st.write(f"**LLM:** `{LLM_MODEL}`")
    if cross_encoder: st.write(f"**Re-ranker:** `{CROSS_ENCODER_MODEL}`")
    # if easyocr_reader: st.write(f"**OCR Languages:** `{OCR_LANGUAGES}`") # Can hide this if not used
    st.divider()
    st.info("Database is updated via external process (e.g., Colab notebook).")

# --- Removed Sidebar PDF Upload Section ---


# --- Main Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources associated with previous assistant messages (with key fix)
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            message_id = message.get("id")
            if message_id:
                expander_key = f"expander_{message_id}"
                with st.expander("Show sources used", key=expander_key):
                    for i, doc in enumerate(message["sources"]):
                        source = doc.metadata.get('source', 'N/A')
                        country = doc.metadata.get('country', 'N/A')
                        year = doc.metadata.get('year', '')
                        chunk_idx = doc.metadata.get('chunk_index', '')
                        label = f"Source {i+1} | Src: {source} | Ctry: {country} | Yr: {year} | Idx: {chunk_idx}"
                        source_key = f"src_hist_{message_id}_{i}"
                        st.text_area(label, doc.page_content, height=100, key=source_key)


# Get user input using chat_input
if prompt := st.chat_input("Ask a question about the documents..." if status_ok else "System initializing..."):
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

            # --- Generate unique ID for this assistant message *before* creating UI elements ---
            asst_msg_id = str(uuid.uuid4())

            # Update placeholder with the actual answer
            response_placeholder.markdown(answer)

            # Display sources for *this* response in an expander (using corrected key logic)
            if sources:
                expander_key = f"expander_{asst_msg_id}" # Use the message ID for key
                with st.expander("Show sources used for this response", key=expander_key):
                    for i, doc in enumerate(sources):
                         source = doc.metadata.get('source', 'N/A')
                         country = doc.metadata.get('country', 'N/A')
                         year = doc.metadata.get('year', '')
                         chunk_idx = doc.metadata.get('chunk_index', '')
                         label = f"Source {i+1} | Src: {source} | Ctry: {country} | Yr: {year} | Idx: {chunk_idx}"
                         source_key = f"src_{asst_msg_id}_{i}" # Use message ID for key
                         st.text_area(label, doc.page_content, height=100, key=source_key)

        # Add assistant response and sources to chat history (using the same ID)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "id": asst_msg_id
            })
        # Rerun needed to potentially clear the chat_input after processing,
        # but let's test without it first as it might cause issues with expander state.
        # st.rerun()