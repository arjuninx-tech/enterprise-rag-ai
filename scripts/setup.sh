#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is required: https://ollama.com/download"
  exit 1
fi

ollama pull nomic-embed-text
ollama pull qwen2.5:7b
echo "Setup complete. Run ./run.sh or ./run_web.sh"
