# Quick launch script for web interfaces
# Choose which interface to launch

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   AUTONOMOUS RESEARCH AGENT - WEB LAUNCHER" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Choose your frontend:" -ForegroundColor Yellow
Write-Host "1. Gradio (Simplest - Quick demos)" -ForegroundColor White
Write-Host "2. Streamlit (Customizable - Interactive apps)" -ForegroundColor White
Write-Host "3. FastAPI (REST API - Custom frontends)" -ForegroundColor White
Write-Host "4. Exit" -ForegroundColor Gray
Write-Host ""

$choice = Read-Host "Enter choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Launching Gradio..." -ForegroundColor Green
        Write-Host "Installing Gradio if needed..." -ForegroundColor Gray
        pip install gradio --quiet
        Write-Host ""
        Write-Host "Starting Gradio on http://localhost:7860" -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
        Write-Host ""
        python app_gradio.py
    }
    "2" {
        Write-Host ""
        Write-Host "Launching Streamlit..." -ForegroundColor Green
        Write-Host "Installing Streamlit if needed..." -ForegroundColor Gray
        pip install streamlit --quiet
        Write-Host ""
        Write-Host "Starting Streamlit on http://localhost:8501" -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
        Write-Host ""
        streamlit run app_streamlit.py
    }
    "3" {
        Write-Host ""
        Write-Host "Launching FastAPI..." -ForegroundColor Green
        Write-Host "Installing FastAPI if needed..." -ForegroundColor Gray
        pip install fastapi uvicorn --quiet
        Write-Host ""
        Write-Host "Starting FastAPI on http://localhost:8000" -ForegroundColor Green
        Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
        Write-Host ""
        python app_fastapi.py
    }
    "4" {
        Write-Host ""
        Write-Host "Goodbye!" -ForegroundColor Cyan
        exit
    }
    default {
        Write-Host ""
        Write-Host "Invalid choice. Please run again." -ForegroundColor Red
    }
}
