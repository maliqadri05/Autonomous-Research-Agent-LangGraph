"""
Research Agents Implementation - Search, Summarize, and Synthesize
"""
from typing import List, Dict, Any, Optional
import arxiv
import wikipedia
from duckduckgo_search import DDGS
from langchain.schema import Document
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline, AutoConfig
import torch
from config import Config
from rag_pipeline import SummarizationChain, SynthesisChain, load_hf_model
from vector_store import VectorStoreManager


class SearchAgent:
    """Agent responsible for searching and gathering information"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        print("Initializing Search Agent LLM...")
        self.llm = load_hf_model(max_tokens=512, temperature=Config.TEMPERATURE)
        print("✓ Search Agent initialized")
    
    def search_arxiv(self, query: str, max_results: int = 3) -> List[Document]:
        """Search arXiv for academic papers"""
        print(f"🔍 Searching arXiv for: {query}")
        documents = []
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                doc = Document(
                    page_content=f"Title: {result.title}\n\nAbstract: {result.summary}",
                    metadata={
                        "source": "arxiv",
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "published": str(result.published),
                        "url": result.entry_id
                    }
                )
                documents.append(doc)
                print(f"  ✓ Found: {result.title}")
            
            return documents
        except Exception as e:
            print(f"  ⚠ arXiv search error: {e}")
            return []
    
    def search_wikipedia(self, query: str, max_results: int = 2) -> List[Document]:
        """Search Wikipedia for general knowledge"""
        print(f"🔍 Searching Wikipedia for: {query}")
        documents = []
        
        try:
            search_results = wikipedia.search(query, results=max_results)
            
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    doc = Document(
                        page_content=f"Title: {page.title}\n\n{page.summary}",
                        metadata={
                            "source": "wikipedia",
                            "title": page.title,
                            "url": page.url
                        }
                    )
                    documents.append(doc)
                    print(f"  ✓ Found: {page.title}")
                except Exception as e:
                    print(f"  ⚠ Could not fetch page '{title}': {e}")
                    continue
            
            return documents
        except Exception as e:
            print(f"  ⚠ Wikipedia search error: {e}")
            return []
    
    def search_web(self, query: str, max_results: int = 5) -> List[Document]:
        """Search the web using DuckDuckGo"""
        print(f"🔍 Searching web for: {query}")
        documents = []
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
                for result in results:
                    doc = Document(
                        page_content=f"Title: {result['title']}\n\n{result['body']}",
                        metadata={
                            "source": "web",
                            "title": result['title'],
                            "url": result['href']
                        }
                    )
                    documents.append(doc)
                    print(f"  ✓ Found: {result['title']}")
            
            return documents
        except Exception as e:
            print(f"  ⚠ Web search error: {e}")
            return []
    
    def comprehensive_search(self, query: str) -> List[Document]:
        """Perform comprehensive search across all sources"""
        print(f"\n{'='*60}")
        print(f"🔍 COMPREHENSIVE SEARCH: {query}")
        print(f"{'='*60}")
        
        all_documents = []
        
        # Search arXiv
        arxiv_docs = self.search_arxiv(query, max_results=2)
        all_documents.extend(arxiv_docs)
        
        # Search Wikipedia
        wiki_docs = self.search_wikipedia(query, max_results=2)
        all_documents.extend(wiki_docs)
        
        # Search Web
        web_docs = self.search_web(query, max_results=3)
        all_documents.extend(web_docs)
        
        # Add to vector store
        if all_documents:
            self.vector_store.add_documents(all_documents)
            print(f"\n✓ Total documents found: {len(all_documents)}")
            print(f"✓ Added to vector store for retrieval")
        else:
            print("\n⚠ No documents found")
        
        return all_documents
    
    def generate_search_queries(self, research_topic: str) -> List[str]:
        """Generate multiple search queries from a research topic"""
        print(f"🤔 Generating search queries for: {research_topic}")
        
        template = """Generate 3 search queries about: {topic}

Queries:"""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"topic": research_topic})
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            
            # Fallback to direct topic if LLM fails to generate queries
            if not queries or len(queries) == 0:
                queries = [research_topic]
                print(f"  Using direct topic as query")
            else:
                print(f"  ✓ Generated {len(queries)} search queries")
            
            return queries
        except Exception as e:
            print(f"  ⚠ Error generating queries: {e}")
            return [research_topic]


class SummarizationAgent:
    """Agent responsible for summarizing information"""
    
    def __init__(self):
        self.summarizer = SummarizationChain()
    
    def summarize_documents(self, documents: List[Document], style: str = "detailed") -> str:
        """Summarize a list of documents"""
        print(f"\n📝 Summarizing {len(documents)} documents ({style} style)...")
        
        if not documents:
            return "No documents to summarize."
        
        # Combine documents with truncation to avoid token limits
        combined_parts = []
        total_chars = 0
        max_chars = 1500  # Limit to ~400 tokens for FLAN-T5
        
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content[:500]  # Limit each document to 500 chars
            part = f"Source: {source}\n{content}"
            
            if total_chars + len(part) > max_chars:
                break
            
            combined_parts.append(part)
            total_chars += len(part)
        
        combined_text = "\n\n---\n\n".join(combined_parts)
        
        summary = self.summarizer.summarize(combined_text, summary_type=style)
        print("✓ Summarization complete")
        return summary
    
    def summarize_text(self, text: str, style: str = "concise") -> str:
        """Summarize raw text"""
        return self.summarizer.summarize(text, summary_type=style)


class SynthesisAgent:
    """Agent responsible for synthesizing information"""
    
    def __init__(self):
        self.synthesizer = SynthesisChain()
    
    def synthesize_research(self, topic: str, summaries: List[str]) -> str:
        """Synthesize research findings into a coherent report"""
        print(f"\n🎯 Synthesizing research on: {topic}")
        print(f"   Processing {len(summaries)} information sources...")
        
        if not summaries:
            return "No information to synthesize."
        
        synthesis = self.synthesizer.synthesize(topic, summaries)
        print("✓ Synthesis complete")
        return synthesis


class ResearchCoordinator:
    """Coordinates the research agents"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.search_agent = SearchAgent(vector_store)
        self.summarization_agent = SummarizationAgent()
        self.synthesis_agent = SynthesisAgent()
        self.vector_store = vector_store
    
    def conduct_research(self, topic: str) -> Dict[str, Any]:
        """Conduct comprehensive research on a topic"""
        print(f"\n{'='*60}")
        print(f"🚀 STARTING RESEARCH: {topic}")
        print(f"{'='*60}\n")
        
        results = {
            "topic": topic,
            "queries": [],
            "documents": [],
            "summaries": [],
            "synthesis": ""
        }
        
        # Step 1: Generate search queries
        queries = self.search_agent.generate_search_queries(topic)
        results["queries"] = queries
        
        # Step 2: Search for each query
        all_docs = []
        for query in queries:
            docs = self.search_agent.comprehensive_search(query)
            all_docs.extend(docs)
        
        results["documents"] = all_docs
        
        # Step 3: Summarize findings
        if all_docs:
            summary = self.summarization_agent.summarize_documents(
                all_docs,
                style="detailed"
            )
            results["summaries"] = [summary]
        
        # Step 4: Synthesize everything
        if results["summaries"]:
            synthesis = self.synthesis_agent.synthesize_research(
                topic,
                results["summaries"]
            )
            results["synthesis"] = synthesis
        
        print(f"\n{'='*60}")
        print("✓ RESEARCH COMPLETE")
        print(f"{'='*60}\n")
        
        return results
