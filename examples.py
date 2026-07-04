"""
Example Usage of the Autonomous Research Agent
"""
from main import AutonomousResearchAgent


def example_1_basic_research():
    """Example 1: Basic research on a topic"""
    print("="*60)
    print("EXAMPLE 1: Basic Research")
    print("="*60 + "\n")
    
    agent = AutonomousResearchAgent()
    
    # Conduct research
    topic = "Transformer Architecture in Deep Learning"
    result = agent.research(topic)
    
    # Print the report
    print("\n" + result["final_report"])


def example_2_multiple_topics():
    """Example 2: Research multiple related topics"""
    print("="*60)
    print("EXAMPLE 2: Multiple Topics Research")
    print("="*60 + "\n")
    
    agent = AutonomousResearchAgent()
    
    topics = [
        "Retrieval-Augmented Generation",
        "FAISS Vector Database",
        "LangChain Framework"
    ]
    
    for topic in topics:
        print(f"\n{'='*60}")
        print(f"Researching: {topic}")
        print(f"{'='*60}\n")
        
        agent.research(topic, save_report=True)


def example_3_research_and_query():
    """Example 3: Research a topic then ask questions"""
    print("="*60)
    print("EXAMPLE 3: Research + Interactive Queries")
    print("="*60 + "\n")
    
    agent = AutonomousResearchAgent()
    
    # Research the topic
    topic = "Graph Neural Networks"
    print(f"📚 Researching: {topic}\n")
    agent.research(topic)
    
    # Ask follow-up questions
    questions = [
        "What are the main applications of Graph Neural Networks?",
        "How do GNNs differ from traditional neural networks?",
        "What are the current challenges in GNN research?"
    ]
    
    print("\n" + "="*60)
    print("💬 Q&A SESSION")
    print("="*60 + "\n")
    
    for question in questions:
        print(f"Q: {question}")
        answer = agent.query(question)
        print(f"A: {answer}\n")
        print("-"*60 + "\n")


def example_4_custom_workflow():
    """Example 4: Custom research workflow"""
    print("="*60)
    print("EXAMPLE 4: Custom Workflow")
    print("="*60 + "\n")
    
    from vector_store import VectorStoreManager
    from agents import ResearchCoordinator
    from rag_pipeline import RAGPipeline
    
    # Initialize components
    vector_store = VectorStoreManager()
    coordinator = ResearchCoordinator(vector_store)
    rag = RAGPipeline(vector_store)
    
    # Custom research process
    topic = "Agentic AI Systems"
    
    # Step 1: Conduct research
    print(f"Step 1: Researching {topic}")
    results = coordinator.conduct_research(topic)
    
    # Step 2: Save vector store
    vector_store.save()
    
    # Step 3: Query specific aspects
    print("\nStep 2: Querying specific aspects")
    queries = [
        "What are agentic AI systems?",
        "How do they differ from traditional AI?",
        "What frameworks support building agents?"
    ]
    
    for query in queries:
        print(f"\n🔍 {query}")
        answer = rag.query(query)
        print(f"✓ {answer[:200]}...\n")


def example_5_comparative_research():
    """Example 5: Comparative research between topics"""
    print("="*60)
    print("EXAMPLE 5: Comparative Research")
    print("="*60 + "\n")
    
    agent = AutonomousResearchAgent()
    
    # Research multiple topics
    topics = [
        "LangGraph vs CrewAI",
        "FAISS vs Pinecone",
        "GPT-4 vs Claude Sonnet"
    ]
    
    for topic in topics:
        print(f"\n📊 Comparative Analysis: {topic}")
        result = agent.research(topic)
        print("\n" + result["synthesis"])
        print("\n" + "="*60)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   AUTONOMOUS RESEARCH AGENT - EXAMPLES                 ║
    ║   Multi-Agent Research Automation with LangGraph       ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    print("\nAvailable Examples:")
    print("1. Basic Research")
    print("2. Multiple Topics Research")
    print("3. Research + Interactive Queries")
    print("4. Custom Workflow")
    print("5. Comparative Research")
    
    choice = input("\nSelect example (1-5) or press Enter to run Example 1: ").strip()
    
    examples = {
        "1": example_1_basic_research,
        "2": example_2_multiple_topics,
        "3": example_3_research_and_query,
        "4": example_4_custom_workflow,
        "5": example_5_comparative_research,
        "": example_1_basic_research
    }
    
    example_func = examples.get(choice, example_1_basic_research)
    
    print("\n" + "="*60)
    print("🚀 Starting Example...")
    print("="*60 + "\n")
    
    try:
        example_func()
        print("\n✓ Example completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
