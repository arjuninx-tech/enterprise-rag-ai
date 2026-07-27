@echo off
echo Starting On-Prem RAG Assistant (Web Mode)...
echo Open your browser at: http://localhost:5000
echo.
set PYTHONPATH=src
if exist .venv\Scripts\python.exe set "PYTHON=.venv\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -m onprem_rag.server
