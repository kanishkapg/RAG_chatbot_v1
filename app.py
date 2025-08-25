import streamlit as st
import json
import os
from test import RAGSystem
from config import DATA_DIR

st.set_page_config(layout="wide", page_title="RAG Chatbot")

st.title("RAG based QnA Chatbot")

# Initialize system on first load
@st.cache_resource
def load_system():
    rag_system = RAGSystem(data_dir=DATA_DIR)
    rag_system.initialize_system()
    return rag_system

rag_system = load_system()

# Create tabs for Chat and Graph Visualization
chat_tab, graph_tab = st.tabs(["Chat", "Document Graph"])

with chat_tab:
    st.subheader("Chat with your Documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.caption(f"Sources: {', '.join(message['sources'])}")
    
    # User input
    if query := st.chat_input("Ask a question about the documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get response from system
        with st.spinner("Thinking..."):
            result = rag_system.query(query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            if result["sources"]:
                st.caption(f"Sources: {', '.join(result['sources'])}")
        
        # Add assistant message to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )

with graph_tab:
    st.subheader("Document Relationship Graph")
    
    # Generate new visualization on demand
    if st.button("Generate/Refresh Graph Visualization"):
        with st.spinner("Generating graph visualization..."):
            viz_path = rag_system.generate_visualization()
            st.success(f"Graph visualization generated at {viz_path}")
    
    # Display the graph if it exists
    viz_path = "document_graph.html"
    if os.path.exists(viz_path):
        with open(viz_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800)
    else:
        st.info("No graph visualization available. Click the button above to generate one.")
    
    # Circular status checker
    st.subheader("Check Circular Status")
    circular_number = st.text_input("Enter circular number (e.g., 10/2022):")
    if st.button("Check Status") and circular_number:
        status = rag_system.find_effective_circular(circular_number)
        if status:
            if status["is_effective"]:
                st.success(f"Circular {status['original']} is still effective")
            else:
                st.warning(f"Circular {status['original']} has been replaced by: {', '.join(status['replaced_by'])}")
        else:
            st.error("Circular not found in the database")