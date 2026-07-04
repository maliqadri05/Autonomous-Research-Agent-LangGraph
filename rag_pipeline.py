"""
RAG Pipeline Implementation using LCEL (LangChain Expression Language)
"""
from typing import List, Dict, Any
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline, AutoConfig
import torch
from config import Config
from vector_store import VectorStoreManager


def load_hf_model(max_tokens: int = None, temperature: float = None):
    """Helper function to load Hugging Face model (auto-detect seq2seq or causal)"""
    if max_tokens is None:
        max_tokens = Config.MAX_NEW_TOKENS
    if temperature is None:
        temperature = Config.TEMPERATURE
    
    tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
    config = AutoConfig.from_pretrained(Config.LLM_MODEL)
    is_seq2seq = config.is_encoder_decoder if hasattr(config, 'is_encoder_decoder') else False
    
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            Config.LLM_MODEL,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
            device_map="auto" if Config.DEVICE == "cuda" else None,
        )
        task = "text2text-generation"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
            device_map="auto" if Config.DEVICE == "cuda" else None,
            load_in_8bit=True if Config.DEVICE == "cuda" else False
        )
        task = "text-generation"
    
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        max_length=2048,  # Allow longer input sequences
        truncation=True,  # Truncate if too long
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
    )
    
    return HuggingFacePipeline(pipeline=pipe)


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline using LCEL"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        print(f"Loading LLM: {Config.LLM_MODEL}")
        self.llm = self._load_huggingface_llm()
        print("✓ LLM loaded")
        self.chain = None  # Lazy initialization
    
    def _load_huggingface_llm(self):
        """Load Hugging Face model as LLM (auto-detect seq2seq or causal)"""
        config = AutoConfig.from_pretrained(Config.LLM_MODEL)
        is_seq2seq = config.is_encoder_decoder if hasattr(config, 'is_encoder_decoder') else False
        model_type = "seq2seq" if is_seq2seq else "causal LM"
        print(f"  Detected {model_type} model")
        return load_hf_model()
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for context"""
        if not docs:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            formatted.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n".join(formatted)
    
    def _create_rag_chain(self):
        """Create RAG chain using LCEL"""
        
        # Define the prompt template (simplified for FLAN-T5)
        template = """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Create the RAG chain using LCEL
        retriever = self.vector_store.get_retriever(k=Config.MAX_SEARCH_RESULTS)
        
        rag_chain = (
            {
                "context": retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def query(self, question: str) -> str:
        """Query the RAG pipeline"""
        try:
            # Lazy initialization of chain
            if self.chain is None:
                self.chain = self._create_rag_chain()
            
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            return f"Error during RAG query: {str(e)}"
    
    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """Query and return both answer and source documents"""
        try:
            # Lazy initialization of chain
            if self.chain is None:
                self.chain = self._create_rag_chain()
            
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(
                question,
                k=Config.MAX_SEARCH_RESULTS
            )
            
            # Generate answer
            answer = self.chain.invoke(question)
            
            return {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ]
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": []
            }


class SummarizationChain:
    """Specialized chain for document summarization"""
    
    def __init__(self):
        print("Initializing Summarization Chain...")
        self.llm = load_hf_model(max_tokens=1024, temperature=0.3)
        print("✓ Summarization Chain initialized")
    
    def summarize(self, text: str, summary_type: str = "concise") -> str:
        """Summarize text with different styles"""
        
        # Truncate input text to avoid token limit
        max_input_chars = 1200  # ~300 tokens
        if len(text) > max_input_chars:
            text = text[:max_input_chars] + "..."
        
        templates = {
            "concise": """Summarize this text in 2-3 sentences:

{text}

Summary:""",
            "detailed": """Provide a detailed summary of this text:

{text}

Summary:""",
            "bullet": """List the main points from this text:

{text}

Points:"""
        }
        
        template = templates.get(summary_type, templates["concise"])
        prompt = PromptTemplate.from_template(template)
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            summary = chain.invoke({"text": text})
            return summary
        except Exception as e:
            return f"Error during summarization: {str(e)}"


class SynthesisChain:
    """Chain for synthesizing multiple pieces of information"""
    
    def __init__(self):
        print("Initializing Synthesis Chain...")
        self.llm = load_hf_model()
        print("✓ Synthesis Chain initialized")
    
    def synthesize(self, topic: str, information_pieces: List[str]) -> str:
        """Synthesize multiple pieces of information into a coherent report"""
        
        # Truncate each piece to avoid token limit
        truncated_pieces = [
            f"Information Source {i+1}:\n{info[:800]}"  # Limit each piece to 800 chars
            for i, info in enumerate(information_pieces)
        ]
        
        # Limit total combined info
        combined_info = "\n\n".join(truncated_pieces)
        if len(combined_info) > 1500:  # ~400 tokens for FLAN-T5
            combined_info = combined_info[:1500] + "..."
        
        template = """Synthesize research findings on {topic}:

{information}

Provide a comprehensive summary:"""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            synthesis = chain.invoke({
                "topic": topic,
                "information": combined_info
            })
            return synthesis
        except Exception as e:
            return f"Error during synthesis: {str(e)}"
