@echo off
echo Starting On-Prem RAG Assistant (Desktop)...
if exist .venv\Scripts\python.exe set "PYTHON=.venv\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"
set PYTHONPATH=src
"%PYTHON%" -m onprem_rag.desktop
