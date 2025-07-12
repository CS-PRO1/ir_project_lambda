# IR Search Engine - UI Startup Script
# This script starts the Streamlit web interface

Write-Host "🎨 Starting IR Search Engine UI..." -ForegroundColor Green

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
    python -c "import streamlit, requests, fastapi" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Some dependencies are missing. Installing requirements..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
} catch {
    Write-Host "❌ Error checking dependencies. Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Check if server is running
Write-Host "🔗 Checking server connection..." -ForegroundColor Blue
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/" -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Server is running!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Server might not be running. Make sure to start the server first:" -ForegroundColor Yellow
        Write-Host "   .\start_server.ps1" -ForegroundColor Cyan
        Write-Host ""
    }
} catch {
    Write-Host "⚠️  Cannot connect to server. Make sure the server is running:" -ForegroundColor Yellow
    Write-Host "   .\start_server.ps1" -ForegroundColor Cyan
    Write-Host "   Or run in a separate terminal window" -ForegroundColor Yellow
    Write-Host ""
}

# Start Streamlit UI
Write-Host "🌐 Starting Streamlit UI..." -ForegroundColor Green
Write-Host "   UI will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop the UI" -ForegroundColor Yellow
Write-Host ""

# Start streamlit app
streamlit run ir_search_engine.core.client_app 