"""
Gradio Web Interface for Autonomous Research Agent
Simple, clean UI perfect for quick deployment
"""
import gradio as gr
import os
from main import AutonomousResearchAgent
from datetime import datetime

# Initialize agent
print("🚀 Initializing Autonomous Research Agent...")
agent = None

def initialize_agent():
    """Initialize the agent (lazy loading)"""
    global agent
    if agent is None:
        agent = AutonomousResearchAgent()
    return agent

def research_topic(topic, progress=gr.Progress()):
    """Conduct research on a topic"""
    if not topic or topic.strip() == "":
        return "⚠️ Please enter a research topic.", "", []
    
    try:
        progress(0, desc="Initializing agent...")
        research_agent = initialize_agent()
        
        progress(0.2, desc="Generating search queries...")
        result = research_agent.research(topic, save_report=True)
        
        progress(0.4, desc="Searching multiple sources...")
        progress(0.6, desc="Summarizing findings...")
        progress(0.8, desc="Synthesizing research...")
        progress(1.0, desc="Complete!")
        
        # Extract information
        synthesis = result.get("synthesis", "No synthesis available")
        queries = result.get("queries", [])
        docs = result.get("documents", [])
        
        # Format sources
        sources = []
        seen_titles = set()
        for doc in docs[:10]:  # Show top 10 sources
            title = doc.metadata.get("title", "Untitled")
            if title not in seen_titles:
                seen_titles.add(title)
                source = doc.metadata.get("source", "unknown")
                url = doc.metadata.get("url", "")
                sources.append(f"**[{source.upper()}]** {title}\n{url}\n")
        
        sources_text = "\n".join(sources) if sources else "No sources available"
        
        return (
            f"✅ **Research Complete!**\n\n{synthesis}",
            f"**Generated Queries:**\n" + "\n".join([f"• {q}" for q in queries]),
            sources_text
        )
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

def ask_question(question, progress=gr.Progress()):
    """Ask a question about researched topics"""
    if not question or question.strip() == "":
        return "⚠️ Please enter a question."
    
    try:
        progress(0.3, desc="Searching knowledge base...")
        research_agent = initialize_agent()
        
        progress(0.6, desc="Generating answer...")
        answer = research_agent.query(question)
        
        progress(1.0, desc="Complete!")
        
        return f"**Answer:**\n\n{answer}"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def clear_database():
    """Clear the vector store"""
    try:
        research_agent = initialize_agent()
        research_agent.vector_store.clear()
        return "✅ Database cleared successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Custom CSS
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0;
}
#subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-top: 0;
}
.source-box {
    background-color: #f8f9fa;
    border-left: 4px solid #667eea;
    padding: 10px;
    margin: 5px 0;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("<h1 id='title'>🤖 Autonomous Research Agent</h1>")
    gr.Markdown("<p id='subtitle'>Powered by Mistral-7B, LangGraph & RAG | 100% Open Source 🤗</p>")
    
    with gr.Tabs():
        
        # Tab 1: Research
        with gr.Tab("🔍 Research"):
            gr.Markdown("""
            ### Conduct Autonomous Research
            Enter any topic and the agent will:
            - Search arXiv, Wikipedia, and the web
            - Analyze and summarize findings
            - Store knowledge for future queries
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    research_input = gr.Textbox(
                        label="Research Topic",
                        placeholder="e.g., Graph Neural Networks, Quantum Computing, Transformers in NLP",
                        lines=2
                    )
                with gr.Column(scale=1):
                    research_btn = gr.Button("🚀 Start Research", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    synthesis_output = gr.Markdown(label="Research Synthesis")
                
            with gr.Row():
                with gr.Column():
                    queries_output = gr.Markdown(label="Search Queries Generated")
                with gr.Column():
                    sources_output = gr.Markdown(label="Sources")
            
            research_btn.click(
                fn=research_topic,
                inputs=[research_input],
                outputs=[synthesis_output, queries_output, sources_output]
            )
        
        # Tab 2: Q&A
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("""
            ### Interactive Q&A
            Ask questions about topics you've researched. The agent will:
            - Search its knowledge base (vector store)
            - Retrieve relevant information
            - Generate context-aware answers
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What are the main applications of GNNs?",
                        lines=3
                    )
                with gr.Column(scale=1):
                    ask_btn = gr.Button("💡 Get Answer", variant="primary", size="lg")
            
            answer_output = gr.Markdown(label="Answer")
            
            ask_btn.click(
                fn=ask_question,
                inputs=[question_input],
                outputs=[answer_output]
            )
            
            gr.Markdown("""
            ---
            **💡 Tip:** Research topics first using the Research tab, then ask questions here!
            """)
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Agent
            
            ### 🤖 Technology Stack
            - **LLM:** Mistral-7B-Instruct-v0.2 (7B parameters, open source)
            - **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
            - **Vector Store:** FAISS (Fast AI Similarity Search)
            - **Framework:** LangChain + LangGraph
            - **RAG:** Retrieval-Augmented Generation
            
            ### 🔍 How It Works
            1. **Research Phase:**
               - Searches arXiv, Wikipedia, and web
               - Collects relevant documents
               - Stores in vector database
               
            2. **Processing Phase:**
               - Converts text to embeddings (384D vectors)
               - Summarizes findings
               - Synthesizes information
               
            3. **Q&A Phase:**
               - Semantic search in vector store
               - Retrieves relevant context
               - Generates accurate answers
            
            ### ✨ Features
            - ✅ 100% Open Source (no API keys!)
            - ✅ Runs locally (privacy-first)
            - ✅ Multi-source research
            - ✅ Persistent knowledge base
            - ✅ Context-aware answers
            
            ### 🔗 Links
            - [GitHub Repository](https://github.com/yourusername/autonomous-research-agent)
            - [Documentation](https://github.com/yourusername/autonomous-research-agent#readme)
            - [Report Issues](https://github.com/yourusername/autonomous-research-agent/issues)
            
            ### 📊 System Status
            - **Model:** Mistral-7B (loaded)
            - **Vector Store:** Active
            - **Device:** {device_info}
            """.format(device_info=os.getenv("DEVICE", "cuda")))
    
    # Footer
    gr.Markdown("""
    ---
    <div style='text-align: center; color: #666;'>
    Made with ❤️ using Hugging Face Transformers | 
    <a href='https://github.com/yourusername/autonomous-research-agent'>View on GitHub</a>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Gradio Web Interface")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True for public URL
        show_error=True
    )
