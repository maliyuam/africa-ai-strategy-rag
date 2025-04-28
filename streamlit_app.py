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
st.caption("Chat about AI/ICT strategies")

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
INITIAL_RETRIEVAL_K = 50
FINAL_RANKED_K = 30
# Memory constants
CONVERSATION_HISTORY_LENGTH = 15 # Number of user/assistant turn pairs to include
# Other constants
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
OCR_LANGUAGES = ['en']
OCR_MIN_TEXT_LENGTH = 50

# --- Caching Expensive Initializations --- (No changes needed here) ---
@st.cache_resource(show_spinner="Connecting to MongoDB...")
def get_mongo_client(uri):
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
    try:
        if not api_key: raise ValueError("OpenAI API Key is empty.")
        llm = ChatOpenAI(openai_api_key=api_key, model_name=LLM_MODEL, temperature=0.1, request_timeout=120)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=EMBEDDING_MODEL)
        openai_client = OpenAI(api_key=api_key)
        return llm, embeddings, openai_client
    except Exception as e:
        st.error(f"Error initializing OpenAI components: {e}")
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
        except Exception as index_e:
            st.sidebar.warning(f"Vector Index '{VECTOR_INDEX_NAME}' check failed: {index_e}.")
        return vector_store
    except Exception as e:
        st.error(f"Error initializing MongoDBAtlasVectorSearch: {e}")
        return None

@st.cache_resource(show_spinner="Loading Re-ranking Model...")
def get_cross_encoder(model_name=CROSS_ENCODER_MODEL):
    """Loads the CrossEncoder model and caches it."""
    try:
        model = CrossEncoder(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading CrossEncoder model '{model_name}': {e}")
        return None

@st.cache_resource(show_spinner="Loading OCR Model...")
def get_easyocr_reader(langs=OCR_LANGUAGES, gpu=False):
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
easyocr_reader = get_easyocr_reader()

# --- Helper Functions --- (No changes needed in extract/chunk/process funcs) ---
def extract_text_from_pdf_stream(pdf_stream):
    """
    Extracts text from a PDF file stream.
    Uses PyMuPDF for text extraction and falls back to EasyOCR for image-based pages.
    """
    full_doc_text = ""
    ocr_pages_count = 0
    tmp_pdf_path = None

    if easyocr_reader is None:
        st.warning("EasyOCR reader not loaded. OCR functionality will be skipped for images.")

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
            page_text = page.get_text("text", sort=True).strip() # Added sort=True for better reading order

            if easyocr_reader and (not page_text or len(page_text) < OCR_MIN_TEXT_LENGTH):
                ocr_attempted_for_page = False
                try:
                    pix = page.get_pixmap(dpi=300) # Higher DPI for better OCR
                    img_bytes = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    np_image = np.array(pil_image)
                    ocr_results = easyocr_reader.readtext(np_image)
                    ocr_attempted_for_page = True
                    ocr_text = " ".join([res[1] for res in ocr_results])

                    if ocr_text.strip() and len(ocr_text.strip()) > len(page_text):
                         if page_text:
                             page_texts.append(page_text + "\n<OCR_TEXT>\n" + ocr_text.strip())
                         else:
                            page_texts.append(ocr_text.strip())
                         ocr_pages_count += 1
                    elif page_text:
                        page_texts.append(page_text)

                except Exception as ocr_err:
                     st.warning(f"OCR failed for page {page_num + 1}: {ocr_err}. Using only text from get_text() if available.")
                     if page_text:
                         page_texts.append(page_text)
            elif page_text:
                page_texts.append(page_text)

        doc.close()
        os.remove(tmp_pdf_path)

        full_doc_text = "\n\n".join(filter(None, page_texts)) # Join pages

        if ocr_pages_count > 0:
            st.info(f"Used OCR to potentially enhance text extraction from {ocr_pages_count} page(s).")

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

# --- Prompt Template with History ---
prompt_template = """
You are a highly specialized AI assistant expert in analyzing and synthesizing information from African AI and ICT strategy documents. Your primary function is to answer questions accurately and concisely based *exclusively* on the provided context, considering the recent chat history for context. Maintain a professional and objective tone.

Chat History (Recent Turns):
--- BEGIN HISTORY ---
{chat_history}
--- END HISTORY ---

Retrieved Context from Documents:
--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---
**(Note: The context may contain text in languages other than English.)**

Current Question: {question}

Instructions for answering:
1.  **Review History:** Consider the recent `Chat History` to understand the background and flow of the conversation, especially if the `Current Question` refers to previous points (e.g., using 'it', 'that', 'those').
2.  **Analyze Context:** Carefully read all provided `Retrieved Context` chunks between the BEGIN and END markers. This context from the documents is the primary source for your answer's factual content.
3.  **Prioritize Retrieved Context:** Base your answer *primarily* on the information found within the `Retrieved Context`. Use the `Chat History` mainly to interpret the `Current Question` correctly.
4.  **Answer ONLY from Retrieved Context:** Your answer's substance must come *strictly* from the `Retrieved Context`. Do NOT use information *only* present in the `Chat History` as the factual basis for your answer, unless it's defining the current question itself. Do NOT use external knowledge or make assumptions.
5.  **Synthesize if Necessary:** If multiple context chunks provide relevant pieces of information, synthesize them into a coherent and unified answer based on the retrieved context.
6.  **Direct Answer & Evidence:** Start with a direct answer to the question based on the retrieved context if possible. Support your answer by referencing the key information or evidence found in the context.
7.  **Handle Insufficient Retrieved Context:**
    * If the retrieved context is relevant but does *not* fully answer the question, clearly state what information *is* available in the context and explicitly mention what parts of the question cannot be answered based on the provided documents.
    * If the retrieved context does *not* contain *any* relevant information to answer the question (even considering the chat history for question interpretation), state clearly: "Based on the provided documents, I cannot answer this question."
8.  **Handle Conflicting Retrieved Context:** If different parts of the retrieved context present conflicting information, acknowledge this discrepancy as found within the provided text.
9.  **Ensure English Output:** Regardless of the original language(s) in the context, formulate and write your entire final answer exclusively in **English**.
10. **Do Not Add Citation Numbers:** Do not add bracketed citation numbers like [1], [2] within your answer text. Focus solely on answering the question based on the provided context.

Answer:"""
CUSTOM_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["chat_history", "context", "question"] # Added chat_history
)

# --- RAG Function (incorporates re-ranking and history) ---
def get_rag_response(query: str, chat_history: str): # Added chat_history parameter
    """Performs RAG, Re-ranking, and LLM call, considering chat history."""
    if not all([vector_store, llm, cross_encoder]):
        missing = [comp_name for comp, comp_name in zip([vector_store, llm, cross_encoder], ["Vector Store", "LLM", "Re-ranker"]) if comp is None]
        return f"Error: Cannot perform query. Missing components: {', '.join(missing)}.", []
    try:
        with st.spinner(f"Searching for top {INITIAL_RETRIEVAL_K} candidates..."):
            # Perform similarity search based on the current query
            initial_docs = vector_store.similarity_search(query, k=INITIAL_RETRIEVAL_K)

        if not initial_docs:
            try:
                doc_count = vector_store.collection.count_documents({})
                if doc_count == 0:
                     return "The document collection appears to be empty. Please add documents.", []
            except Exception:
                pass
            return "I couldn't find any relevant sections in the documents based on your query.", []

        with st.spinner(f"Re-ranking {len(initial_docs)} candidates to select top {FINAL_RANKED_K}..."):
            rerank_pairs = [(query, doc.page_content) for doc in initial_docs]
            scores = cross_encoder.predict(rerank_pairs)
            docs_with_scores = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
            re_ranked_docs = [doc for doc, score in docs_with_scores[:FINAL_RANKED_K]]

        if not re_ranked_docs:
             return "Found some potentially relevant sections, but their relevance score was too low after re-ranking.", []

        # Prepare context string from re-ranked documents
        context_str = "\n\n---\n\n".join([doc.page_content for doc in re_ranked_docs])

        # Format the prompt with history, context, and question
        formatted_prompt = CUSTOM_PROMPT.format(
            chat_history=chat_history, # Pass formatted history
            context=context_str,
            question=query
        )

        with st.spinner(f"Generating answer in English based on {len(re_ranked_docs)} sources and history..."):
            response = llm.invoke(formatted_prompt)
            answer = response.content

        return answer, re_ranked_docs
    except Exception as e:
        st.error(f"An error occurred during the RAG process: {e}"); st.exception(e)
        if "context length" in str(e).lower():
             st.warning(f"The context might be too long for the LLM ({LLM_MODEL}). Consider reducing FINAL_RANKED_K (currently {FINAL_RANKED_K}) or CONVERSATION_HISTORY_LENGTH (currently {CONVERSATION_HISTORY_LENGTH}).")
        return "Sorry, an error occurred while processing your request.", []

# --- Helper Function to Format History ---
def format_chat_history(messages: list, k: int = CONVERSATION_HISTORY_LENGTH) -> str:
    """Formats the last k turns of chat history for the prompt."""
    if not messages:
        return "No history yet."

    # Get the last k * 2 messages (k user + k assistant turns)
    history_messages = messages[-(k*2):]

    formatted_history = []
    for msg in history_messages:
        role = "Human" if msg["role"] == "user" else "AI"
        formatted_history.append(f"{role}: {msg['content']}")

    return "\n".join(formatted_history)


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
    if easyocr_reader: st.success("EasyOCR Reader Loaded")
    else: st.warning("EasyOCR Reader Failed / Not Used")
    st.divider()
    st.subheader("Configuration")
    if embeddings: st.write(f"**Embedding:** `{EMBEDDING_MODEL}`")
    if llm: st.write(f"**LLM:** `{LLM_MODEL}`")
    if cross_encoder: st.write(f"**Re-ranker:** `{CROSS_ENCODER_MODEL}`")
    st.write(f"**Context Chunks:** `{FINAL_RANKED_K}` (From top {INITIAL_RETRIEVAL_K})")
    st.write(f"**History Length:** `{CONVERSATION_HISTORY_LENGTH}` turns")
    st.divider()
    st.info("Database is updated via external process (e.g., Colab notebook).")

# --- Main Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources associated with previous assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            message_id = message.get("id")
            if message_id:
                # FIX 1 applied: Removed key argument
                with st.expander(f"Show Sources Used ({len(message['sources'])}):"):
                    for i, doc in enumerate(message["sources"]):
                        source = doc.metadata.get('source', 'N/A')
                        country = doc.metadata.get('country', 'N/A')
                        year = doc.metadata.get('year', '')
                        chunk_idx = doc.metadata.get('chunk_index', '')
                        # Updated Label for Citation Style
                        label = f"[{i+1}] Src: {source} | Ctry: {country} | Yr: {year} | Chunk: {chunk_idx}"
                        source_key = f"src_hist_{message_id}_{i}"
                        st.text_area(label, doc.page_content, height=100, key=source_key, help="Original content from the source document.")


# Get user input using chat_input
if prompt := st.chat_input("Ask a question about the documents..." if status_ok else "System initializing..."):
    if not status_ok:
        st.warning("Please wait for all system components to initialize (check sidebar).")
    else:
        # Add user message to history and display it FIRST
        user_msg_id = str(uuid.uuid4())
        st.session_state.messages.append({"role": "user", "content": prompt, "id": user_msg_id})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare history FOR the RAG call (excluding the current prompt itself)
        history_for_prompt = format_chat_history(st.session_state.messages[:-1]) # Pass all messages *before* the current one

        # Get and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")

            # Call RAG function with current prompt and formatted history
            answer, sources = get_rag_response(prompt, history_for_prompt)

            # Generate unique ID for this assistant message
            asst_msg_id = str(uuid.uuid4())

            # Update placeholder with the actual answer
            response_placeholder.markdown(answer)

            # Display sources for *this* response in an expander
            if sources:
                # FIX 2 applied: Removed key argument
                with st.expander(f"Show Sources Used for this Response ({len(sources)}):"):
                    for i, doc in enumerate(sources):
                        source = doc.metadata.get('source', 'N/A')
                        country = doc.metadata.get('country', 'N/A')
                        year = doc.metadata.get('year', '')
                        chunk_idx = doc.metadata.get('chunk_index', '')
                        # Updated Label for Citation Style
                        label = f"[{i+1}] Src: {source} | Ctry: {country} | Yr: {year} | Chunk: {chunk_idx}"
                        source_key = f"src_{asst_msg_id}_{i}"
                        st.text_area(label, doc.page_content, height=100, key=source_key, help="Original content from the source document.")

        # Add assistant response and sources to chat history AFTER generating it
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "id": asst_msg_id
            })
        # Optional: rerun may cause state issues with expanders/history display
        st.rerun()