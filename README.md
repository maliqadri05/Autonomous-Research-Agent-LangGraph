# 🤖 Autonomous Research Agent

> A multi-agent research automation system powered by **LangGraph**, **RAG**, **FAISS**, and **Hugging Face LLMs**. Automates academic research, document retrieval, summarization, and knowledge synthesis using fully open-source AI models.

---

## ✨ Features

- 🤖 Multi-agent workflow with **LangGraph**
- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Multi-source research (arXiv, Wikipedia, Web)
- 🧠 Semantic search with **FAISS**
- 📝 Automatic summarization & report generation
- 💬 Interactive knowledge base Q&A
- 🌐 Web interfaces (Gradio, Streamlit, FastAPI)
- 🔒 100% local inference using Hugging Face models

---

## 🏗️ Architecture

```text
                 User Query
                      │
                      ▼
            Research Coordinator
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
 Search Agent   Summary Agent   Synthesis Agent
      │               │               │
      └───────────────┼───────────────┘
                      ▼
               FAISS Vector Store
                      │
                      ▼
                 RAG Pipeline
                      │
                      ▼
             Final Research Report
```

---

## 🛠️ Tech Stack

- LangGraph
- LangChain (LCEL)
- FAISS
- Hugging Face Transformers
- Sentence Transformers
- PyTorch
- Python

---

## 🚀 Installation

```bash
git clone <repository-url>
cd Autonomous_Research_Agent

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

(Optional)

```bash
cp .env.example .env
```

---

## ▶️ Usage

### Research

```bash
python main.py "Large Language Models"
```

### Interactive Mode

```bash
python main.py --interactive
```

### Research + Interactive

```bash
python main.py "Transformer Architecture" --interactive
```

### Web Applications

```bash
python app_gradio.py

streamlit run app_streamlit.py

python app_fastapi.py
```

---

## 📂 Project Structure

```text
Autonomous_Research_Agent/
│
├── main.py
├── agents.py
├── graph.py
├── rag_pipeline.py
├── vector_store.py
├── config.py
│
├── app_gradio.py
├── app_streamlit.py
├── app_fastapi.py
│
├── requirements.txt
├── requirements-web.txt
├── .env.example
│
├── data/
│   └── vector_store/
│
└── reports/
```

---

## ⚙️ Workflow

1. Generate search queries
2. Search multiple knowledge sources
3. Store documents in FAISS
4. Retrieve relevant context
5. Summarize findings
6. Synthesize knowledge
7. Generate research report
8. Answer follow-up questions via RAG

---

## 📊 Core Components

| Component | Purpose |
|----------|---------|
| **Search Agent** | Collects information from multiple sources |
| **Summarization Agent** | Creates concise document summaries |
| **Synthesis Agent** | Combines knowledge into coherent insights |
| **FAISS** | Semantic vector database |
| **RAG Pipeline** | Context-aware question answering |
| **LangGraph** | Multi-agent orchestration |

---

## 📈 Highlights

- Fully open-source AI stack
- No OpenAI API required
- Local inference support
- Persistent vector database
- Modular architecture
- Extensible agent workflow

---

## 📌 Requirements

- Python 3.9+
- 16 GB RAM (recommended)
- NVIDIA GPU (optional but recommended)
- Internet connection for document retrieval

---

## 🔮 Future Improvements

- PDF ingestion
- Citation graph visualization
- Multi-language research
- Additional LLM providers
- Collaborative research workspace
- Model quantization support

---

## 🙏 Acknowledgements

Built using:

- LangChain
- LangGraph
- Hugging Face
- FAISS
- PyTorch

---

## 📄 License

This project is released under the **MIT License**.

---

<div align="center">

**Built for autonomous research, semantic search, and AI-powered knowledge discovery.**

⭐ If you found this project useful, consider giving it a star.

</div
