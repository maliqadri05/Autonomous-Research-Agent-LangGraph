"""
Streamlit Web Interface for Autonomous Research Agent
More customizable UI with advanced features
"""
import streamlit as st
import os
from main import AutonomousResearchAgent
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "research_history" not in st.session_state:
    st.session_state.research_history = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

def initialize_agent():
    """Initialize the agent"""
    if st.session_state.agent is None:
        with st.spinner("🤖 Initializing AI models... This may take a minute..."):
            st.session_state.agent = AutonomousResearchAgent()
    return st.session_state.agent

# Header
st.markdown('<h1 class="main-header">🤖 Autonomous Research Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Mistral-7B, LangGraph & RAG | 100% Open Source 🤗</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("### 🔧 System Status")
    device = os.getenv("DEVICE", "cuda")
    st.info(f"**Device:** {device}")
    st.info(f"**Model:** Mistral-7B-Instruct-v0.2")
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    if st.session_state.research_history:
        st.metric("Topics Researched", len(st.session_state.research_history))
    if st.session_state.qa_history:
        st.metric("Questions Asked", len(st.session_state.qa_history))
    
    st.markdown("---")
    
    st.markdown("### 🗑️ Database")
    if st.button("Clear Vector Store", type="secondary"):
        try:
            agent = initialize_agent()
            agent.vector_store.clear()
            st.success("✅ Database cleared!")
            st.session_state.research_history = []
            st.session_state.qa_history = []
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This agent autonomously:
    - 🔍 Searches multiple sources
    - 📚 Stores knowledge
    - 🤖 Answers questions
    - 📝 Generates reports
    
    [GitHub](https://github.com/yourusername/repo) | 
    [Docs](https://github.com/yourusername/repo#readme)
    """)

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Research", "💬 Q&A", "📜 History", "📖 Guide"])

# Tab 1: Research
with tab1:
    st.header("🔍 Conduct Research")
    st.markdown("Enter any topic and let the AI agent do autonomous research for you!")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        research_topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Graph Neural Networks, Quantum Computing, Transformers in NLP",
            key="research_topic"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        research_button = st.button("🚀 Start Research", type="primary", key="research_btn")
    
    if research_button and research_topic:
        try:
            agent = initialize_agent()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Generating search queries...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            status_text.text("🌐 Searching arXiv, Wikipedia, and web...")
            progress_bar.progress(40)
            
            # Conduct research
            result = agent.research(research_topic, save_report=True)
            
            progress_bar.progress(60)
            status_text.text("📝 Summarizing findings...")
            time.sleep(0.5)
            
            progress_bar.progress(80)
            status_text.text("🎯 Synthesizing research...")
            time.sleep(0.5)
            
            progress_bar.progress(100)
            status_text.text("✅ Research complete!")
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Add to history
            st.session_state.research_history.append({
                "topic": research_topic,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result": result
            })
            
            # Display results
            st.success(f"✅ Research on '{research_topic}' completed successfully!")
            
            # Synthesis
            st.markdown("### 📊 Research Synthesis")
            st.markdown(result.get("synthesis", "No synthesis available"))
            
            # Expanders for details
            with st.expander("🔎 Search Queries Generated"):
                queries = result.get("queries", [])
                for i, query in enumerate(queries, 1):
                    st.markdown(f"{i}. {query}")
            
            with st.expander("📚 Sources Found"):
                docs = result.get("documents", [])
                for doc in docs[:10]:
                    title = doc.metadata.get("title", "Untitled")
                    source = doc.metadata.get("source", "unknown")
                    url = doc.metadata.get("url", "")
                    
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>[{source.upper()}]</strong> {title}<br>
                        <a href="{url}" target="_blank">{url}</a>
                    </div>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Tab 2: Q&A
with tab2:
    st.header("💬 Ask Questions")
    st.markdown("Ask anything about the topics you've researched. The agent uses RAG to provide accurate answers.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "Your Question",
            placeholder="e.g., What are the main applications of Graph Neural Networks?",
            height=100,
            key="question"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ask_button = st.button("💡 Get Answer", type="primary", key="ask_btn")
    
    if ask_button and question:
        try:
            agent = initialize_agent()
            
            with st.spinner("🤔 Searching knowledge base and generating answer..."):
                answer = agent.query(question)
            
            # Add to history
            st.session_state.qa_history.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Display answer
            st.markdown("### 💡 Answer")
            st.markdown(answer)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Tip: Make sure to research topics first in the Research tab!")

# Tab 3: History
with tab3:
    st.header("📜 Research & Q&A History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Research History")
        if st.session_state.research_history:
            for item in reversed(st.session_state.research_history):
                with st.expander(f"📌 {item['topic']} - {item['timestamp']}"):
                    st.markdown(item['result'].get('synthesis', 'No synthesis available')[:500] + "...")
        else:
            st.info("No research history yet. Start researching in the Research tab!")
    
    with col2:
        st.subheader("💬 Q&A History")
        if st.session_state.qa_history:
            for item in reversed(st.session_state.qa_history):
                with st.expander(f"❓ {item['question'][:50]}... - {item['timestamp']}"):
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(f"**A:** {item['answer'][:300]}...")
        else:
            st.info("No Q&A history yet. Start asking questions in the Q&A tab!")

# Tab 4: Guide
with tab4:
    st.header("📖 User Guide")
    
    st.markdown("""
    ## How to Use This Agent
    
    ### 1️⃣ Research a Topic
    1. Go to the **Research** tab
    2. Enter any topic you want to research
    3. Click "Start Research"
    4. Wait for the agent to search, analyze, and synthesize information
    
    ### 2️⃣ Ask Questions
    1. After researching, go to the **Q&A** tab
    2. Ask specific questions about your research
    3. Get context-aware answers from the knowledge base
    
    ### 3️⃣ View History
    - Check the **History** tab to see past research and questions
    
    ---
    
    ## How It Works
    
    ### 🔍 Research Process
    ```
    Your Topic
        ↓
    Generate Search Queries (LLM)
        ↓
    Search Multiple Sources
    ├─ arXiv (academic papers)
    ├─ Wikipedia (encyclopedia)
    └─ Web (recent articles)
        ↓
    Convert to Vectors (Embeddings)
        ↓
    Store in FAISS Database
        ↓
    Summarize & Synthesize (LLM)
        ↓
    Research Report
    ```
    
    ### 💡 Q&A Process
    ```
    Your Question
        ↓
    Convert to Vector (Embeddings)
        ↓
    Search Vector Database (FAISS)
        ↓
    Retrieve Relevant Documents
        ↓
    Generate Answer (LLM + Context)
        ↓
    Context-Aware Answer
    ```
    
    ---
    
    ## Technology Stack
    
    - **LLM:** Mistral-7B-Instruct-v0.2 (7B params)
    - **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
    - **Vector Store:** FAISS
    - **Framework:** LangChain + LangGraph
    - **Architecture:** RAG (Retrieval-Augmented Generation)
    
    ---
    
    ## Tips
    
    - 💡 Research broad topics first, then ask specific questions
    - 💡 The more you research, the more knowledge the agent accumulates
    - 💡 Clear the database if you want to start fresh
    - 💡 Wait for research to complete before asking questions
    
    ---
    
    ## Privacy & Open Source
    
    ✅ **100% Open Source** - No proprietary APIs  
    ✅ **Runs Locally** - Your data never leaves your machine  
    ✅ **No API Keys** - Free to use  
    ✅ **Transparent** - All code is open and auditable  
    
    ---
    
    [📖 Full Documentation](https://github.com/yourusername/repo) | 
    [🐛 Report Issues](https://github.com/yourusername/repo/issues) | 
    [⭐ Star on GitHub](https://github.com/yourusername/repo)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
Made with ❤️ using Hugging Face Transformers | 
<a href='https://github.com/yourusername/autonomous-research-agent'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)
