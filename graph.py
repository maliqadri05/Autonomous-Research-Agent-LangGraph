"""
LangGraph Multi-Agent Workflow for Autonomous Research
"""
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from vector_store import VectorStoreManager
from agents import SearchAgent, SummarizationAgent, SynthesisAgent
from rag_pipeline import RAGPipeline


class ResearchState(TypedDict):
    """State schema for the research workflow"""
    topic: str
    queries: List[str]
    documents: List[Document]
    summaries: List[str]
    synthesis: str
    final_report: str
    iteration: int
    max_iterations: int


class ResearchGraph:
    """LangGraph-based multi-agent research workflow"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.search_agent = SearchAgent(vector_store)
        self.summarization_agent = SummarizationAgent()
        self.synthesis_agent = SynthesisAgent()
        self.rag_pipeline = RAGPipeline(vector_store)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self.generate_queries_node)
        workflow.add_node("search", self.search_node)
        workflow.add_node("summarize", self.summarize_node)
        workflow.add_node("synthesize", self.synthesize_node)
        workflow.add_node("generate_report", self.generate_report_node)
        
        # Define edges
        workflow.set_entry_point("generate_queries")
        workflow.add_edge("generate_queries", "search")
        workflow.add_edge("search", "summarize")
        workflow.add_edge("summarize", "synthesize")
        workflow.add_edge("synthesize", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def generate_queries_node(self, state: ResearchState) -> ResearchState:
        """Node: Generate search queries from research topic"""
        print(f"\n{'='*60}")
        print("NODE: Generate Queries")
        print(f"{'='*60}")
        
        topic = state["topic"]
        queries = self.search_agent.generate_search_queries(topic)
        
        state["queries"] = queries
        state["iteration"] = state.get("iteration", 0) + 1
        
        print(f"Generated {len(queries)} queries:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        
        return state
    
    def search_node(self, state: ResearchState) -> ResearchState:
        """Node: Search for information using generated queries"""
        print(f"\n{'='*60}")
        print("NODE: Search")
        print(f"{'='*60}")
        
        all_documents = []
        
        for query in state["queries"]:
            # Search across multiple sources
            docs = self.search_agent.comprehensive_search(query)
            all_documents.extend(docs)
        
        # Remove duplicates based on content similarity
        unique_docs = self._remove_duplicate_docs(all_documents)
        
        state["documents"] = unique_docs
        print(f"\n✓ Total unique documents collected: {len(unique_docs)}")
        
        return state
    
    def summarize_node(self, state: ResearchState) -> ResearchState:
        """Node: Summarize collected documents"""
        print(f"\n{'='*60}")
        print("NODE: Summarize")
        print(f"{'='*60}")
        
        documents = state["documents"]
        
        if not documents:
            print("⚠ No documents to summarize")
            state["summaries"] = []
            return state
        
        # Group documents by source for better summarization
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        summaries = []
        for source, docs in docs_by_source.items():
            print(f"\n📝 Summarizing {len(docs)} documents from {source}...")
            summary = self.summarization_agent.summarize_documents(docs, style="detailed")
            summaries.append(f"[{source.upper()}]\n{summary}")
        
        state["summaries"] = summaries
        print(f"\n✓ Created {len(summaries)} summaries")
        
        return state
    
    def synthesize_node(self, state: ResearchState) -> ResearchState:
        """Node: Synthesize information into coherent insights"""
        print(f"\n{'='*60}")
        print("NODE: Synthesize")
        print(f"{'='*60}")
        
        topic = state["topic"]
        summaries = state["summaries"]
        
        if not summaries:
            print("⚠ No summaries to synthesize")
            state["synthesis"] = "Insufficient information for synthesis."
            return state
        
        synthesis = self.synthesis_agent.synthesize_research(topic, summaries)
        state["synthesis"] = synthesis
        
        return state
    
    def generate_report_node(self, state: ResearchState) -> ResearchState:
        """Node: Generate final research report"""
        print(f"\n{'='*60}")
        print("NODE: Generate Report")
        print(f"{'='*60}")
        
        topic = state["topic"]
        synthesis = state["synthesis"]
        documents = state["documents"]
        
        # Create comprehensive report
        report_parts = [
            f"# Research Report: {topic}",
            f"\n## Executive Summary",
            synthesis,
            f"\n## Key Findings",
            self._extract_key_findings(synthesis),
            f"\n## Sources",
            self._format_sources(documents),
            f"\n## Methodology",
            f"- Queries generated: {len(state['queries'])}",
            f"- Documents analyzed: {len(documents)}",
            f"- Sources consulted: arXiv, Wikipedia, Web Search"
        ]
        
        final_report = "\n".join(report_parts)
        state["final_report"] = final_report
        
        print("✓ Final report generated")
        
        return state
    
    def _remove_duplicate_docs(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity"""
        if not documents:
            return []
        
        unique_docs = []
        seen_contents = set()
        
        for doc in documents:
            # Use first 100 chars as fingerprint
            fingerprint = doc.page_content[:100].strip()
            if fingerprint not in seen_contents:
                seen_contents.add(fingerprint)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _extract_key_findings(self, synthesis: str) -> str:
        """Extract key findings from synthesis"""
        # Simple extraction - in production, use LLM for this
        lines = synthesis.split("\n")
        key_points = [line for line in lines if line.strip() and len(line) > 50]
        return "\n".join(f"- {line.strip()}" for line in key_points[:5])
    
    def _format_sources(self, documents: List[Document]) -> str:
        """Format source citations"""
        sources = []
        for doc in documents:
            metadata = doc.metadata
            source_type = metadata.get("source", "Unknown")
            title = metadata.get("title", "Untitled")
            url = metadata.get("url", "No URL")
            sources.append(f"- [{source_type}] {title}\n  {url}")
        
        return "\n".join(sources)
    
    def run(self, topic: str, max_iterations: int = 1) -> ResearchState:
        """Execute the research workflow"""
        print(f"\n{'#'*60}")
        print(f"🚀 AUTONOMOUS RESEARCH AGENT ACTIVATED")
        print(f"Topic: {topic}")
        print(f"{'#'*60}\n")
        
        initial_state = ResearchState(
            topic=topic,
            queries=[],
            documents=[],
            summaries=[],
            synthesis="",
            final_report="",
            iteration=0,
            max_iterations=max_iterations
        )
        
        # Execute the graph
        final_state = self.graph.invoke(initial_state)
        
        print(f"\n{'#'*60}")
        print("✓ RESEARCH WORKFLOW COMPLETE")
        print(f"{'#'*60}\n")
        
        return final_state
    
    def interactive_query(self, question: str) -> str:
        """Query the knowledge base interactively using RAG"""
        print(f"\n💬 Query: {question}")
        response = self.rag_pipeline.query(question)
        return response
