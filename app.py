import streamlit as st
import os
import time
from typing import Dict, List

# Import RAG components
try:
    from test import RAGSystem  # Assumes classes are defined in test.py under if __name__ == '__main__' guard
except ImportError as e:
    st.error(f"Failed to import RAGSystem from test.py: {e}")
    st.stop()

# Import DB connection helper
try:
    from test_tesseract import get_db_connection
except ImportError as e:
    st.warning(f"Could not import get_db_connection: {e}")

from config import DATA_DIR

st.set_page_config(page_title="RAG based QnA Chatbot", page_icon="💬", layout="wide")

st.title("💬 SLT Circulars Chatbot")
st.caption("Enter a question about the circulars. The system retrieves relevant chunks and generates an answer with citations.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    force_recreate = st.checkbox("Recreate Vector Collection", value=False, help="Drops and recreates the Qdrant collection before initialization.")
    show_relevant = st.checkbox("Show relevant documents", value=True)
    if st.button("Initialize / Refresh System", type="primary"):
        st.session_state["init_requested"] = True

# Session state initialization
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # Each: {role: user|assistant, content: str, meta: dict}
if "init_log" not in st.session_state:
    st.session_state.init_log = []
if "init_requested" not in st.session_state:
    st.session_state.init_requested = False

# Helper: create required PostgreSQL tables (idempotent)
def ensure_postgres_tables():
    if 'get_db_connection' not in globals():
        return
    try:
        conn = get_db_connection()
        with conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_files (
                    filename VARCHAR(255) PRIMARY KEY,
                    file_hash VARCHAR(64),
                    extracted_text TEXT,
                    extraction_date TIMESTAMP,
                    UNIQUE (filename)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    filename VARCHAR(255) PRIMARY KEY,
                    metadata JSONB,
                    extraction_date TIMESTAMP,
                    FOREIGN KEY (filename) REFERENCES pdf_files(filename)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON pdf_files(file_hash)")
        st.session_state.init_log.append("PostgreSQL tables verified.")
    except Exception as e:
        st.session_state.init_log.append(f"Failed to verify PostgreSQL tables: {e}")

# Initialization workflow
if st.session_state.init_requested:
    st.session_state.init_requested = False
    st.session_state.init_log = []
    with st.spinner("Initializing RAG system (this may take a while on first run)..."):
        ensure_postgres_tables()
        try:
            rag = RAGSystem(data_dir=DATA_DIR)
            # Expose vector_db.initialize_collection for control
            try:
                rag.vector_db.initialize_collection(force_recreate=force_recreate)
                st.session_state.init_log.append("Vector collection initialized.")
            except Exception as e:
                st.session_state.init_log.append(f"Vector collection init failed: {e}")
            # Build / update documents & embeddings
            try:
                rag.initialize_system()  # Assumes it will skip existing already processed PDFs
                st.session_state.init_log.append("System document/embedding initialization complete.")
            except Exception as e:
                st.session_state.init_log.append(f"RAG initialization failed: {e}")
                st.error(f"Initialization error: {e}")
            else:
                st.session_state.rag_system = rag
        except Exception as e:
            st.session_state.init_log.append(f"Failed to construct RAGSystem: {e}")
            st.error(f"Failed to construct RAGSystem: {e}")

# Display initialization log
if st.session_state.init_log:
    with st.expander("Initialization Log", expanded=False):
        for line in st.session_state.init_log:
            st.write(line)

if st.session_state.rag_system is None:
    st.info("Click 'Initialize / Refresh System' in the sidebar to start.")
    st.stop()

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_area("Your question", placeholder="Ask something about the circulars...", height=100)
    submitted = st.form_submit_button("Send")

if submitted and user_query.strip():
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.spinner("Retrieving and generating answer..."):
        start_time = time.time()
        try:
            result = st.session_state.rag_system.query(user_query.strip())
            elapsed = time.time() - start_time
            answer = result.get("answer", "No answer returned")
            sources = result.get("sources", [])
            rel_docs = result.get("relevant_documents", [])
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": {"sources": sources, "relevant_documents": rel_docs, "latency_s": round(elapsed, 2)}
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error processing query: {e}",
                "meta": {"sources": [], "relevant_documents": []}
            })

# Render conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        chat = st.chat_message("assistant")
        chat.markdown(msg["content"])
        meta = msg.get("meta", {})
        if meta.get("sources"):
            chat.caption("Sources: " + ", ".join(meta["sources"]))
        if show_relevant and meta.get("relevant_documents"):
            with chat.expander("Relevant documents"):
                for d in meta["relevant_documents"]:
                    doc_id = d.get("id") or d.get("metadata", {}).get("circular_number") or "(no id)"
                    score = d.get("score")
                    text_snippet = d.get("text", "")[:300].replace("\n", " ") + ("..." if len(d.get("text", "")) > 300 else "")
                    st.markdown(f"**ID:** {doc_id} | **Score:** {score:.4f}" if score is not None else f"**ID:** {doc_id}")
                    st.markdown(f"**Snippet:** {text_snippet}")
                    md = d.get("metadata", {})
                    if md:
                        st.code(md, language="json")

st.markdown("---")
st.caption("RAG Chatbot • Vector search + LLM generated answers with citations.")