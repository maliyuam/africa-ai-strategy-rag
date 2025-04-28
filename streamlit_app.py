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
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import tempfile
import pandas
import uuid # To generate unique keys for messages

# --- Configuration & Secrets ---
load_dotenv()

st.set_page_config(page_title="African AI Strategies Chat", layout="wide")
st.title("💬 African AI & ICT Strategies Chatbot")
st.caption("Chat about AI/ICT strategies (Handles Scanned PDFs | Re-ranking Accuracy)")

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
# (get_mongo_client, initialize_openai_langchain, get_vector_store, get_cross_encoder, get_easyocr_reader)
# (These cached functions remain unchanged from the previous EasyOCR version)
# ... (copy the full cached functions here) ...
@st.cache_resource(show_spinner="Connecting to MongoDB...")
def get_mongo_client(uri):
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000) # Add timeout
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
    try:
        if not api_key: raise ValueError("OpenAI API Key is empty.")
        llm = ChatOpenAI(openai_api_key=api_key, model_name=LLM_MODEL, temperature=0.1) # Even lower temp
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=EMBEDDING_MODEL)
        openai_client = OpenAI(api_key=api_key) # For potential direct use
        return llm, embeddings, openai_client
    except Exception as e:
        st.error(f"Error initializing OpenAI components: {e}")
        st.info("Check your OpenAI API Key.")
        return None, None, None

@st.cache_resource(show_spinner="Connecting to Vector Store...")
def get_vector_store(_mongo_client, _embeddings):
    if _mongo_client is None or _embeddings is None:
        st.error("Cannot initialize Vector Store: Dependencies missing.")
        return None
    try:
        db = _mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]
        vector_store = MongoDBAtlasVectorSearch(collection=collection, embedding=_embeddings, index_name=VECTOR_INDEX_NAME)
        try: # Health check
            vector_store.similarity_search("test query health check", k=1)
            # st.sidebar.success(f"Vector Store Index '{VECTOR_INDEX_NAME}' OK.") # Keep sidebar cleaner
        except Exception as index_e:
            st.sidebar.warning(f"Vector Index '{VECTOR_INDEX_NAME}' check failed: {index_e}.")
        return vector_store
    except Exception as e:
        st.error(f"Error initializing MongoDBAtlasVectorSearch: {e}")
        st.warning(f"Ensure Vector Index '{VECTOR_INDEX_NAME}' is active in Atlas.")
        return None

@st.cache_resource(show_spinner="Loading Re-ranking Model...")
def get_cross_encoder(model_name=CROSS_ENCODER_MODEL):
    """Loads the CrossEncoder model and caches it."""
    try:
        model = CrossEncoder(model_name)
        # st.sidebar.success(f"Re-ranking model ({model_name}) loaded.") # Keep sidebar cleaner
        return model
    except Exception as e:
        st.error(f"Error loading CrossEncoder model '{model_name}': {e}")
        st.info("Model might need to be downloaded. Ensure internet connection.")
        return None

# --- NEW: Cache EasyOCR Reader ---
@st.cache_resource(show_spinner="Loading OCR Model...")
def get_easyocr_reader(langs=OCR_LANGUAGES, gpu=False):
    """Loads the EasyOCR reader and caches it."""
    try:
        # You might need to adjust gpu=True if you have a compatible setup
        reader = easyocr.Reader(langs, gpu=gpu)
        # st.sidebar.success(f"EasyOCR Reader loaded ({', '.join(langs)}, GPU={gpu}).") # Keep sidebar clean
        return reader
    except Exception as e:
        st.error(f"Error loading EasyOCR Reader: {e}")
        st.info("Ensure PyTorch is installed correctly. Try CPU mode (gpu=False).")
        return None

# --- Initialize Clients ---
mongo_client = get_mongo_client(MONGO_URI)
llm, embeddings, openai_client = initialize_openai_langchain(OPENAI_API_KEY)
vector_store = get_vector_store(mongo_client, embeddings)
cross_encoder = get_cross_encoder()
easyocr_reader = get_easyocr_reader()

# --- Helper Functions ---
# (extract_text_from_pdf_stream, chunk_text, process_and_store_pdf remain unchanged from EasyOCR version)
# ... (copy the full functions here) ...
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
                try:
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    np_image = np.array(pil_image)
                    ocr_results = easyocr_reader.readtext(np_image)
                    ocr_text = " ".join([res[1] for res in ocr_results])
                    if ocr_text.strip():
                        page_texts.append(ocr_text.strip())
                        ocr_pages_count += 1
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
            st.info(f"Used OCR to extract text from {ocr_pages_count} page(s).") # Show info in main area

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
    if not vector_store:
        st.error("Vector Store not initialized. Cannot process PDF.")
        return False

    status_placeholder = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0, text="Starting...")

    try:
        status_placeholder.info(f"Processing '{filename}'...")
        progress_bar.progress(5, text="Extracting text (incl. OCR)...")
        with st.spinner("Extracting text (incl. OCR)..."):
            full_text = extract_text_from_pdf_stream(pdf_stream)
            if full_text is None:
                status_placeholder.empty(); progress_bar.empty(); return False
            if not full_text:
                 st.warning(f"No text found in '{filename}'. Skipping storage.")
                 status_placeholder.empty(); progress_bar.empty(); return False

        progress_bar.progress(20, text="Chunking text...")
        with st.spinner("Chunking text..."):
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
                 progress_percentage = min(40 + (i + batch_size) / total_docs * 60, 100)
                 progress_text = f"Embedding & Storing Chunks... {int(progress_percentage)}%"
                 progress_bar.progress(int(progress_percentage)/100, text=progress_text)

        st.success(f"Successfully added {len(inserted_ids)} chunks from '{filename}'.")
        progress_bar.empty(); status_placeholder.empty(); st.spinner()
        return True

    except Exception as e:
        st.error(f"An error occurred during PDF processing/storage: {e}"); st.exception(e)
        status_placeholder.empty();
        if 'progress_bar' in locals(): progress_bar.empty()
        st.spinner()
        return False


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
# (Unchanged - takes query, returns answer and sources)
def get_rag_response(query):
    """Performs RAG, Re-ranking, and LLM call."""
    if not all([vector_store, llm, cross_encoder]):
        return "Error: Core AI components are not ready.", []
    try:
        initial_docs = vector_store.similarity_search(query, k=INITIAL_RETRIEVAL_K)
        if not initial_docs:
            return "I couldn't find any potentially relevant documents based on your query.", []
        rerank_pairs = [(query, doc.page_content) for doc in initial_docs]
        scores = cross_encoder.predict(rerank_pairs)
        docs_with_scores = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        re_ranked_docs = [doc for doc, score in docs_with_scores[:FINAL_RANKED_K]]
        context_str = "\n\n---\n\n".join([doc.page_content for doc in re_ranked_docs])
        formatted_prompt = CUSTOM_PROMPT.format(context=context_str, question=query)
        response = llm.invoke(formatted_prompt)
        answer = response.content
        return answer, re_ranked_docs
    except Exception as e:
        st.error(f"An error occurred during the RAG process: {e}")
        st.exception(e)
        return "Sorry, an error occurred while processing your request.", []


# --- UI Components ---

# Sidebar (Unchanged)
with st.sidebar:
    st.divider()
    st.subheader("System Status")
    # Display status of components...
    if mongo_client: st.success("MongoDB Connected")
    else: st.error("MongoDB Disconnected")
    if embeddings: st.success("Embeddings Initialized")
    else: st.error("Embeddings Failed")
    if llm: st.success("LLM Initialized")
    else: st.error("LLM Failed")
    if vector_store: st.success("Vector Store Ready")
    else: st.error("Vector Store Failed")
    if cross_encoder: st.success("Re-ranker Loaded")
    else: st.error("Re-ranker Failed")
    if easyocr_reader: st.success("EasyOCR Reader Loaded")
    else: st.error("EasyOCR Reader Failed")
    st.divider()
    st.subheader("Configuration")
    if embeddings: st.info(f"Embedding: `{EMBEDDING_MODEL}`")
    if llm: st.info(f"LLM: `{LLM_MODEL}`")
    if cross_encoder: st.info(f"Re-ranker: `{CROSS_ENCODER_MODEL}`")
    if easyocr_reader: st.info(f"OCR Languages: `{OCR_LANGUAGES}`")
    st.divider()
    st.header("📄 Add New Document")
    if 'uploader_key' not in st.session_state:
        st.session_state['uploader_key'] = 0
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", key=f"uploader_{st.session_state['uploader_key']}")
    if uploaded_file is not None:
        file_key_suffix = uploaded_file.id
        country_name = st.text_input("Enter Country:", key=f"country_{file_key_suffix}")
        doc_year_str = st.text_input("Enter Year (optional):", key=f"year_{file_key_suffix}")
        doc_source_url = st.text_input("Enter Source URL (optional):", key=f"url_{file_key_suffix}")
        if st.button("Process Document", key=f"process_{file_key_suffix}"):
            if not all([vector_store, llm, embeddings, easyocr_reader]):
                 st.error("Backend components not ready.")
            elif not country_name: st.warning("Please enter the country name.")
            else:
                year_val = int(doc_year_str) if doc_year_str.isdigit() else None
                url_val = doc_source_url if doc_source_url else None
                success = process_and_store_pdf(uploaded_file, uploaded_file.name, country_name, year_val, url_val)
                if success:
                    st.session_state['uploader_key'] += 1; st.rerun()
    st.divider()

# --- Main Chat Interface ---

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources associated with previous assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Show sources used"):
                for i, doc in enumerate(message["sources"]):
                    source = doc.metadata.get('source', 'N/A')
                    country = doc.metadata.get('country', 'N/A')
                    year = doc.metadata.get('year', '')
                    chunk_idx = doc.metadata.get('chunk_index', '')
                    label = f"Source {i+1} | Src: {source} | Ctry: {country} | Yr: {year} | Idx: {chunk_idx}"
                    # Use unique key based on message ID and index
                    source_key = f"src_hist_{message['id']}_{i}"
                    st.text_area(label, doc.page_content, height=100, key=source_key)


# Get user input using chat_input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response and display it
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        answer, sources = get_rag_response(prompt) # RAG function call
        message_placeholder.markdown(answer)

        # Display sources for *this* response in an expander
        if sources:
            with st.expander("Show sources used for this response"):
                for i, doc in enumerate(sources):
                     source = doc.metadata.get('source', 'N/A')
                     country = doc.metadata.get('country', 'N/A')
                     year = doc.metadata.get('year', '')
                     chunk_idx = doc.metadata.get('chunk_index', '')
                     label = f"Source {i+1} | Src: {source} | Ctry: {country} | Yr: {year} | Idx: {chunk_idx}"
                     # Use unique key based on current prompt and index
                     source_key = f"src_curr_{len(st.session_state.messages)}_{i}"
                     st.text_area(label, doc.page_content, height=100, key=source_key)


    # Add assistant response and sources to history
    # Generate a unique ID for message keying
    msg_id = str(uuid.uuid4())
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "id": msg_id
        })