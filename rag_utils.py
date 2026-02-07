import os
import json
import tempfile


# Modern Modular Imports
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader 
from langchain_classic.chains import RetrievalQA

from dotenv import load_dotenv

# Initialize sentence transformer embeddings (free)
load_dotenv(dotenv_path=".env") 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # or "gemini-1.5-pro", etc.
    temperature=0.1,
    api_key=os.getenv("GOOGLE_API_KEY")
)

VECTOR_STORE_PATH = "vector_store"
HISTORY_FILE = os.path.join(VECTOR_STORE_PATH, "conversation_history.json")

# Ensure the vector store directory exists
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def process_files(files, chunk_size = 1000, chunk_overlap = 100):
    docs = []
    # Store file temp
    for file in files:
        file_ext = os.path.splitext(file.name)[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
    
    # txt extraction
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            
        elif file_ext == ".csv":
            loader = CSVLoader(tmp_path)
            
        elif file_ext == ".txt":
            loader = TextLoader(tmp_path)
            
        else:
            continue
        docs.extend(loader.load())
            
    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    
    # Vector store
# Corrected Vector Store Ingestion
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=VECTOR_STORE_PATH,
        collection_name="data_chunks"
    )
    return vector_db

# ask_llm function to query the vector store and get answers from the LLM
def ask_llm(query, top_k=3):
    vectordb = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings, collection_name="data_chunks")
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    qa = RetrievalQA.from_chain_type(llm=llm, 
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True)
    
    # Modern LangChain uses invoke()
    result = qa.invoke({"query": query})
    
    answer = result['result']
    # Get the source documents list
    source_docs = result.get('source_documents', [])
    
    # Extract unique source names (filenames) from the metadata
    sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in source_docs]))
    
    return answer, sources