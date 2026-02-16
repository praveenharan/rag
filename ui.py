__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from rag_utils import process_files, ask_llm



st.set_page_config(page_title="Advanced RAG", page_icon=":sparkles:", layout="wide")
st.title("RAG | by Praveen HARAN")

st.sidebar.header("Upload files")
uploaded_files = st.sidebar.file_uploader("Choose files to upload", accept_multiple_files=True)
chunk_size = st.sidebar.slider("Chunk size", min_value=100, max_value=1000, value=500, step=50)
chunk_overlap = st.sidebar.slider("Chunk overlap", min_value=0, max_value=chunk_size, value=50, step=10)
top_k = st.sidebar.slider("Top K", min_value=1, max_value=20, value=5, step=1)

if st.sidebar.button("Process files"):
    if uploaded_files:
        with st.spinner("Processing files..."):
            process_files(uploaded_files, chunk_size, chunk_overlap)
        st.success(f"Processing {len(uploaded_files)} files with chunk size {chunk_size}, overlap {chunk_overlap}, and top K {top_k}.")
    else:
        st.warning("Please upload at least one file to process.")

st.subheader("Chat with your Files")
query = st.text_input("Enter your query here")
if st.button("Ask RAG"):
    if query:
        st.info(f"Querying with: {query}")
        answer, _ = ask_llm(query, top_k)
        
        st.subheader("Answer")
        st.markdown(answer)
    else:
        st.warning("Please enter a query to ask.")
