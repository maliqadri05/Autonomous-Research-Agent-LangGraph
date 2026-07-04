"""
FAISS Vector Store Implementation with Document Processing
"""
import os
from typing import List, Optional
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from config import Config


class VectorStoreManager:
    """Manages FAISS vector store for document retrieval"""
    
    def __init__(self):
        print(f"Loading embeddings model: {Config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': Config.DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embeddings model loaded")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store: Optional[FAISS] = None
        self._load_or_create_store()
    
    def _load_or_create_store(self):
        """Load existing vector store or create a new one"""
        if os.path.exists(Config.VECTOR_STORE_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    Config.VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✓ Loaded existing vector store from {Config.VECTOR_STORE_PATH}")
            except Exception as e:
                print(f"⚠ Could not load vector store: {e}. Creating new one.")
                self.vector_store = None
        else:
            print("Creating new vector store...")
            self.vector_store = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        if not documents:
            print("No documents to add")
            return
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            print("✓ Created new vector store")
        else:
            self.vector_store.add_documents(chunks)
            print(f"✓ Added {len(chunks)} chunks to existing vector store")
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Add raw texts to the vector store"""
        if not texts:
            return
        
        # Create documents from texts
        documents = [
            Document(page_content=text, metadata=metadatas[i] if metadatas else {})
            for i, text in enumerate(texts)
        ]
        self.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if self.vector_store is None:
            print("⚠ Vector store is empty")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            print(f"✓ Found {len(results)} relevant documents")
            return results
        except Exception as e:
            print(f"⚠ Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with relevance scores"""
        if self.vector_store is None:
            print("⚠ Vector store is empty")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            print(f"✓ Found {len(results)} relevant documents with scores")
            return results
        except Exception as e:
            print(f"⚠ Error during similarity search: {e}")
            return []
    
    def save(self) -> None:
        """Save the vector store to disk"""
        if self.vector_store is None:
            print("⚠ No vector store to save")
            return
        
        os.makedirs(os.path.dirname(Config.VECTOR_STORE_PATH), exist_ok=True)
        self.vector_store.save_local(Config.VECTOR_STORE_PATH)
        print(f"✓ Saved vector store to {Config.VECTOR_STORE_PATH}")
    
    def get_retriever(self, k: int = 5):
        """Get a retriever for the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Add documents first.")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def clear(self) -> None:
        """Clear the vector store"""
        self.vector_store = None
        if os.path.exists(Config.VECTOR_STORE_PATH):
            import shutil
            shutil.rmtree(Config.VECTOR_STORE_PATH)
        print("✓ Cleared vector store")
