import streamlit as st
import os
import random
from document_handler import DocumentHandler
import sys

# Temporary fix to add rag as package to sys path
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))


from init_rag import initialize_rag

rag = initialize_rag()

# Set page configuration
st.set_page_config(page_title="RAG Chat App", page_icon="📚", layout="wide")

# Initialize session state variables if they don't exist
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}  # {filename: document_content}
if "focused_doc" not in st.session_state:
    st.session_state.focused_doc = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize DocumentHandler
doc_handler = DocumentHandler()

def display_sources(sources):
    """Display reference sources with expandable chunks."""
    for i, source in enumerate(sources):
        with st.expander(f"Source {i + 1}: {source['document']}"):
            st.markdown(f"_{source['text']}_")


# Main page layout
st.title("📚 RAG Chat Application")

# Sidebar for document upload and management
with st.sidebar:
    st.header("Document Management")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents", accept_multiple_files=True, type=["pdf", "md", "txt"]
    )

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if file is already processed
            if uploaded_file.name not in st.session_state.uploaded_docs:
                content = doc_handler.parse_document(uploaded_file)
                rag.ingest(file=uploaded_file) # Uploaded file is a file not a filename
                if content:
                    st.session_state.uploaded_docs[uploaded_file.name] = content
                    st.success(f"Successfully uploaded: {uploaded_file.name}")
                else:
                    st.error(f"Failed to process: {uploaded_file.name}")

    # Document selection
    st.subheader("Your Documents")
    if st.session_state.uploaded_docs:
        for doc_name in st.session_state.uploaded_docs.keys():
            if st.button(f"📄 {doc_name}", key=f"btn_{doc_name}"):
                st.session_state.focused_doc = doc_name
    else:
        st.info("No documents uploaded yet.")

# Document viewer in the right sidebar
if st.session_state.uploaded_docs and st.session_state.focused_doc:
    with st.sidebar.expander("Document Preview", expanded=True):
        st.subheader(st.session_state.focused_doc)
        st.text_area(
            "Content",
            value=st.session_state.uploaded_docs[st.session_state.focused_doc],
            height=400,
            disabled=True,
        )

# Main chat interface
st.subheader("Chat")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # Display sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            st.divider()
            st.markdown("**References:**")
            display_sources(message["sources"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Simulate assistant response with RAG sources
    # response = f"This would be a RAG response about: {prompt}"
    rag_response, rag_context = rag.query(prompt)
    response = rag_response.content

    # Generate simulated source references
    sources = []
    if st.session_state.uploaded_docs:
        # Get documents to reference (preferring the focused one)
        docs_to_reference = (
            [st.session_state.focused_doc]
            if st.session_state.focused_doc
            else list(st.session_state.uploaded_docs.keys())
        )

        for context in rag_context:
            sources.append({
                "document": doc_name,
                "text": context
            })

    # Add assistant response with sources to chat history
    assistant_msg = {"role": "assistant", "content": response, "sources": sources}
    st.session_state.chat_history.append(assistant_msg)

    # Display assistant response with sources
    with st.chat_message("assistant"):
        st.write(response)
        if sources:
            st.divider()
            st.markdown("**References:**")
            display_sources(sources)
