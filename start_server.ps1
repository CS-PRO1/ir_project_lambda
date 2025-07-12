# IR Search Engine - Server Startup Script
# This script starts the FastAPI backend server

Write-Host "🚀 Starting IR Search Engine Server..." -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found. Please run the following commands first:" -ForegroundColor Red
    Write-Host "   python -m venv venv" -ForegroundColor Yellow
    Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "   pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Blue
& ".\venv\Scripts\Activate.ps1"

# Check if requirements are installed
Write-Host "🔍 Checking dependencies..." -ForegroundColor Blue
try {
    python -c "import fastapi, uvicorn, streamlit, transformers, torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Some dependencies are missing. Installing requirements..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
} catch {
    Write-Host "❌ Error checking dependencies. Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Start the server
Write-Host "🌐 Starting FastAPI server..." -ForegroundColor Green
Write-Host "   Server will be available at: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "   API documentation at: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start uvicorn server
uvicorn ir_search_engine.core.server:app --host 127.0.0.1 --port 8000 --reload 