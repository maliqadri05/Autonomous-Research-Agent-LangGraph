"""
Main Application - Autonomous Research Agent
"""
import argparse
from vector_store import VectorStoreManager
from graph import ResearchGraph
from rag_pipeline import RAGPipeline


class AutonomousResearchAgent:
    """Main class for the Autonomous Research Agent"""
    
    def __init__(self):
        print("🤖 Initializing Autonomous Research Agent...")
        self.vector_store = VectorStoreManager()
        self.research_graph = ResearchGraph(self.vector_store)
        print("✓ Agent initialized successfully!\n")
    
    def research(self, topic: str, save_report: bool = True) -> dict:
        """Conduct autonomous research on a topic"""
        # Run the research workflow
        result = self.research_graph.run(topic)
        
        # Save the report if requested
        if save_report and result.get("final_report"):
            self._save_report(topic, result["final_report"])
        
        # Save the vector store
        self.vector_store.save()
        
        return result
    
    def query(self, question: str) -> str:
        """Query the research knowledge base"""
        return self.research_graph.interactive_query(question)
    
    def interactive_mode(self):
        """Start interactive Q&A mode"""
        print("\n" + "="*60)
        print("💬 INTERACTIVE MODE")
        print("="*60)
        print("Ask questions about the researched topics.")
        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                answer = self.query(question)
                print(f"\nAgent: {answer}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"⚠ Error: {e}\n")
    
    def _save_report(self, topic: str, report: str):
        """Save research report to file"""
        import os
        from datetime import datetime
        
        # Create reports directory
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:50]
        filename = f"{reports_dir}/{safe_topic}_{timestamp}.md"
        
        # Save report
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n💾 Report saved: {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent - Multi-agent research automation system"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic to investigate"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start in interactive query mode"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the research report"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = AutonomousResearchAgent()
        
        if args.interactive and not args.topic:
            # Pure interactive mode
            agent.interactive_mode()
        
        elif args.topic:
            # Research mode
            print(f"📚 Starting research on: {args.topic}\n")
            result = agent.research(args.topic, save_report=not args.no_save)
            
            # Display results
            print("\n" + "="*60)
            print("📊 RESEARCH RESULTS")
            print("="*60)
            print(result["final_report"])
            
            # Start interactive mode if requested
            if args.interactive:
                agent.interactive_mode()
        
        else:
            # No topic provided
            parser.print_help()
            print("\nExample usage:")
            print("  python main.py \"Large Language Models\"")
            print("  python main.py \"Quantum Computing\" --interactive")
            print("  python main.py --interactive")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
