"""
Configuration module for the Autonomous Research Agent
"""
import os
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the research agent"""
    
    # API Keys (optional for open-source models)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional if using HF only
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Optional for private models
    
    # LangSmith Configuration
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "research-agent")
    
    # Model Configuration - Hugging Face Open Source
    LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))
    
    # Auto-detect device: cuda if available, else cpu
    _device_env = os.getenv("DEVICE", "auto")
    if _device_env == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = _device_env
    
    # Vector Store Configuration
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Research Configuration
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        # Hugging Face models don't require API keys for open-source models
        print(f"✓ Using LLM: {cls.LLM_MODEL}")
        print(f"✓ Using Embeddings: {cls.EMBEDDING_MODEL}")
        print(f"✓ Device: {cls.DEVICE}")
        
        # Warn if running on CPU with large model
        if cls.DEVICE == "cpu" and "7B" in cls.LLM_MODEL:
            print("⚠️  WARNING: Running 7B model on CPU will be slow!")
            print("   Consider using a smaller model like 'google/flan-t5-base'")
            print("   Set in .env: LLM_MODEL=google/flan-t5-base")
        
        return True

# Validate configuration on import
Config.validate()
