import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import config # For KNOWLEDGE_BASE_FILE

# Configuration for the vector store path
VECTOR_STORE_DIR = "faiss_index_store"  # Directory to store FAISS index
INDEX_NAME = "bignalytics_index"    # Name for the FAISS index file within the directory

def create_and_save_vector_store(embedding_model):
    """
    Creates a FAISS vector store from the knowledge base specified in config.py
    and saves it to disk.
    Args:
        embedding_model: The embedding model instance to use.
    Returns:
        The created FAISS vectorstore instance, or None if creation fails.
    """
    print(f"--- Creating new vector store from knowledge base: {config.KNOWLEDGE_BASE_FILE} ---")
    try:
        loader = TextLoader(config.KNOWLEDGE_BASE_FILE)
        documents = loader.load()
        if not documents:
            print(f"--- No documents found in {config.KNOWLEDGE_BASE_FILE}. Cannot create vector store. ---")
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        print(f"--- Generating embeddings and creating FAISS index. This may take a while... ---")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
        
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)
            print(f"--- Created directory: {VECTOR_STORE_DIR} ---")
            
        index_path = os.path.join(VECTOR_STORE_DIR, INDEX_NAME)
        vectorstore.save_local(index_path)
        print(f"--- Vector store created and saved to: {index_path} ---")
        return vectorstore
    except Exception as e:
        print(f"--- Error creating and saving vector store: {e} ---")
        return None

def load_vector_store(embedding_model):
    """
    Loads an existing FAISS vector store from disk.
    Args:
        embedding_model: The embedding model instance to use for loading.
    Returns:
        The loaded FAISS vectorstore instance if found and loaded successfully, otherwise None.
    """
    index_path = os.path.join(VECTOR_STORE_DIR, INDEX_NAME)
    if os.path.exists(index_path) and os.path.isdir(index_path): # FAISS.save_local creates a directory
        print(f"--- Attempting to load existing vector store from: {index_path} ---")
        try:
            # allow_dangerous_deserialization is needed for FAISS.load_local with langchain_community >= 0.0.30
            vectorstore = FAISS.load_local(
                index_path, 
                embeddings=embedding_model, 
                allow_dangerous_deserialization=True
            )
            print(f"--- Vector store loaded successfully from: {index_path} ---")
            return vectorstore
        except Exception as e:
            print(f"--- Error loading vector store from {index_path}: {e}. Will attempt to recreate. ---")
            return None
    else:
        print(f"--- No existing vector store found at: {index_path} (or it's not a directory) ---")
        return None
