# Troubleshooting Guide

---

## Startup Issues

### "Ollama not found" / "Cannot connect to Ollama"

**Symptom:** Red error in the sidebar: *Cannot connect to Ollama at http://localhost:11434*

**Fix:**
1. Open a terminal and run: `ollama serve`
2. Wait for the message: *Listening on 127.0.0.1:11434*
3. Refresh the application page

If Ollama isn't installed: download from [https://ollama.com](https://ollama.com)

---

### "model 'qwen2.5:7b' not found (404)"

**Symptom:** Error when asking a question, or the sidebar shows ⬇️ next to the selected model.

**Fix:** Pull the model from a terminal (internet required once):
```bash
ollama pull qwen2.5:7b
```

For other models:
```bash
ollama pull qwen3:8b
ollama pull gemma3:4b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

After pulling, refresh the page — no app restart needed.

---

### "streamlit: command not found"

**Symptom:** Running `run.bat`, `run.sh`, or the browser launcher fails.

**Fix:** Your virtual environment is not activated.

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Then re-run the command.

---

### App starts but immediately shows an error page

**Fix:** Clear the Python cache and restart:

```bat
# Windows
run.bat

# macOS/Linux
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
./run.sh
```

---

## Knowledge Base Issues

### "The knowledge base is empty" after uploading documents

**Symptom:** You've added documents to `data/documents/` but the assistant says it has no documents.

**Fix:** Click **Build Index** in the Knowledge Base window. Documents are not searchable until indexing completes.

---

### Rebuild is stuck / progress bar not moving

**Symptom:** Progress bar appears but stays at 0% or stops partway through.

**Causes and fixes:**

1. **Ollama is not running.** Open a terminal, run `ollama serve`, then retry.
2. **nomic-embed-text not pulled.** Run `ollama pull nomic-embed-text` then retry.
3. **Very large PDF.** This is normal—page extraction, structural splitting, and embedding take time on CPU. The progress bar will move as passages are embedded.
4. **Scanned/image PDF.** The loader cannot extract text from image-based PDFs. Use a text-based PDF or convert it first (e.g., with Adobe Acrobat's OCR export).

---

### "Collection does not exist" error

**Symptom:** Error shown after a knowledge base rebuild.

**Fix:** This is a stale cache issue — it self-resolves on the next query. If it persists:
1. Stop the app
2. Delete `data/vector_db/`
3. Restart and rebuild

---

### Answers are wrong or cite the wrong document

**Possible causes:**

- **Passage size too small:** Short passages lose context. Set `CHUNK_SIZE` in `.env`, then build the index again.
- **Focused retrieval too narrow:** `TOP_K` controls focused questions, while `SUMMARY_TOP_K` controls summaries and broad explanations.
- **Multiple conflicting documents:** If you have old and new versions of the same document, remove the old one and rebuild.
- **Scanned PDF:** Text extraction failed silently. Check the terminal output during rebuild for warnings.

---

## Performance Issues

### Responses are very slow

**Tips:**

1. **Switch to a smaller model.** Gemma 3 4B is the fastest option in the sidebar — try it for speed.
2. **Enable GPU acceleration.** If you have an NVIDIA GPU or Apple Silicon, Ollama uses it automatically. Check: `ollama ps` — it should show your GPU in the output.
3. **Reduce context.** Lower `TOP_K`, `SUMMARY_TOP_K`, or `OLLAMA_NUM_CTX` in `.env` for shorter prompts and lower memory use.

---

### High RAM usage

Ollama loads the full model into RAM (or VRAM). Approximate memory requirements:

| Model | RAM needed |
|---|---|
| Gemma 3 4B | ~4 GB |
| Qwen 2.5 7B | ~6 GB |
| Qwen 3 8B | ~7 GB |
| Llama 3.1 8B | ~7 GB |

If your machine has less than 8 GB RAM, use Gemma 3 4B only.

---

## Windows-Specific Issues

### `run.bat` not recognised in PowerShell

**Symptom:** `run.bat : The term 'run.bat' is not recognized...`

**Fix:** Use `.\` prefix in PowerShell:
```powershell
.\run.bat
```
Or use Command Prompt (`cmd`) instead of PowerShell.

---

### `stop.bat` doesn't fully stop the app

**Fix:** Run in a Command Prompt (not PowerShell), or kill the process manually:
```powershell
Stop-Process -Name "streamlit" -Force
```

---

### `UnicodeDecodeError` during document loading

**Symptom:** Error during rebuild mentioning `'charmap' codec`

**Fix:** This affects `.txt` files with non-ASCII characters. Convert the file to UTF-8 encoding using Notepad++ or VS Code: open the file → *File > Save with Encoding > UTF-8*.

---

## Still stuck?

Open a [GitHub Issue](../.github/ISSUE_TEMPLATE/bug_report.md) with:
- The full error message from the terminal
- Your OS and Python version
- The document type causing the issue

---

## Useful Commands Reference

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# List installed models
ollama list

# Pull a model
ollama pull qwen2.5:7b

# Check which models are loaded in memory
ollama ps

# Clear Python cache (macOS/Linux)
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Rebuild knowledge base from command line
python -m onprem_rag.services.ingest
```
