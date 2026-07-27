@echo off
setlocal

python -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat || exit /b 1
python -m pip install --upgrade pip || exit /b 1
python -m pip install -e . || exit /b 1

where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama is required: https://ollama.com/download
  exit /b 1
)

ollama pull nomic-embed-text || exit /b 1
ollama pull qwen2.5:7b || exit /b 1
echo Setup complete. Run run.bat or run_web.bat
