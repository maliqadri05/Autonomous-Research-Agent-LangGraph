"""
FastAPI Backend for Autonomous Research Agent
REST API for frontend integration
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from main import AutonomousResearchAgent
from datetime import datetime
import os

# Initialize FastAPI
app = FastAPI(
    title="Autonomous Research Agent API",
    description="REST API for AI-powered autonomous research",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None

def get_agent():
    """Get or initialize agent"""
    global agent
    if agent is None:
        agent = AutonomousResearchAgent()
    return agent

# Request/Response Models
class ResearchRequest(BaseModel):
    topic: str
    save_report: bool = True

class QuestionRequest(BaseModel):
    question: str

class ResearchResponse(BaseModel):
    success: bool
    topic: str
    synthesis: str
    queries: List[str]
    sources: List[Dict]
    timestamp: str

class AnswerResponse(BaseModel):
    success: bool
    question: str
    answer: str
    timestamp: str

class StatusResponse(BaseModel):
    status: str
    device: str
    model: str
    vector_store_active: bool

# Endpoints

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Autonomous Research Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

@app.get("/health", response_model=Dict)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    try:
        research_agent = get_agent()
        return StatusResponse(
            status="ready",
            device=os.getenv("DEVICE", "cuda"),
            model="mistralai/Mistral-7B-Instruct-v0.2",
            vector_store_active=research_agent.vector_store is not None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Conduct research on a topic"""
    try:
        if not request.topic or request.topic.strip() == "":
            raise HTTPException(status_code=400, detail="Topic cannot be empty")
        
        research_agent = get_agent()
        
        # Conduct research
        result = research_agent.research(request.topic, save_report=request.save_report)
        
        # Format sources
        sources = []
        for doc in result.get("documents", [])[:10]:
            sources.append({
                "title": doc.metadata.get("title", "Untitled"),
                "source": doc.metadata.get("source", "unknown"),
                "url": doc.metadata.get("url", ""),
                "content_preview": doc.page_content[:200] + "..."
            })
        
        return ResearchResponse(
            success=True,
            topic=request.topic,
            synthesis=result.get("synthesis", "No synthesis available"),
            queries=result.get("queries", []),
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about researched topics"""
    try:
        if not request.question or request.question.strip() == "":
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        research_agent = get_agent()
        
        # Get answer
        answer = research_agent.query(request.question)
        
        return AnswerResponse(
            success=True,
            question=request.question,
            answer=answer,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/database")
async def clear_database():
    """Clear the vector store"""
    try:
        research_agent = get_agent()
        research_agent.vector_store.clear()
        return {"success": True, "message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/stats")
async def get_database_stats():
    """Get vector store statistics"""
    try:
        research_agent = get_agent()
        # This is a simplified version - implement actual stats
        return {
            "vector_store_path": os.getenv("VECTOR_STORE_PATH", "./data/vector_store"),
            "exists": os.path.exists(os.getenv("VECTOR_STORE_PATH", "./data/vector_store"))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting FastAPI Server")
    print("="*60)
    print("\n📖 API Documentation: http://localhost:8000/docs")
    print("📊 Health Check: http://localhost:8000/health")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        "app_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
