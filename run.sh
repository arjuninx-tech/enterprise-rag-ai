#!/usr/bin/env bash
echo "Starting On-Prem RAG Assistant..."
PYTHON_BIN=python3
if [ -x .venv/bin/python ]; then PYTHON_BIN=.venv/bin/python; fi
PYTHONPATH=src "$PYTHON_BIN" -m onprem_rag.desktop
