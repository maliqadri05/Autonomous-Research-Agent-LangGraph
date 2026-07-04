# Setup script for Autonomous Research Agent
# Run this script to set up your environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Autonomous Research Agent Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Skipping..." -ForegroundColor Gray
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✓ pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
pip install -r requirements.txt --quiet
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Create .env file if it doesn't exist
Write-Host "`nSetting up environment variables..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host ".env file already exists. Skipping..." -ForegroundColor Gray
} else {
    Copy-Item ".env.example" ".env"
    Write-Host "✓ Created .env file from template" -ForegroundColor Green
    Write-Host ""
    Write-Host "ℹ Using open-source Hugging Face models (no API keys needed!)" -ForegroundColor Cyan
    Write-Host "  Default: Mistral-7B-Instruct-v0.2" -ForegroundColor Gray
    Write-Host "  You can customize models in .env file" -ForegroundColor Gray
}

# Create data directories
Write-Host "`nCreating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "reports" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. (Optional) Edit .env to customize models" -ForegroundColor White
Write-Host "2. Run: python main.py 'Your Research Topic'" -ForegroundColor White
Write-Host "3. Or try: python examples.py" -ForegroundColor White
Write-Host ""
Write-Host "ℹ First run will download models (~14GB for Mistral-7B)" -ForegroundColor Cyan
Write-Host "  Models are cached for future use" -ForegroundColor Gray
Write-Host ""
Write-Host "📖 Model guide: See HUGGINGFACE_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "For help:" -ForegroundColor Yellow
Write-Host "  python main.py --help" -ForegroundColor White
Write-Host ""
Write-Host "Happy researching! 🚀" -ForegroundColor Green
Write-Host ""
