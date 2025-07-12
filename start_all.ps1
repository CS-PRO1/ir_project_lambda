# IR Search Engine - Complete Startup Script
# This script starts both the server and UI in separate windows

Write-Host "🚀 Starting IR Search Engine (Server + UI)..." -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found. Please run the following commands first:" -ForegroundColor Red
    Write-Host "   python -m venv venv" -ForegroundColor Yellow
    Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "   pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "📦 Starting server in new window..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\start_server.ps1"

Write-Host "⏳ Waiting 10 seconds for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "🎨 Starting UI in new window..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\start_ui.ps1"

Write-Host "✅ Both components started!" -ForegroundColor Green
Write-Host "   Server: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "   UI: http://localhost:8501" -ForegroundColor Cyan
Write-Host "   API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to close this window..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 