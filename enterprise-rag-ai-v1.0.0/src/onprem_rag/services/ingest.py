"""
ingest.py
─────────
Reads all approved documents, splits into chunks, generates embeddings
via Ollama (nomic-embed-text), and stores everything in ChromaDB.

Run directly to (re)build the knowledge base:
    python -m onprem_rag.services.ingest
"""

import sys
import re
from pathlib import Path

import chromadb
import ollama

from onprem_rag.config import (
    CHROMA_PATH,
    CHROMA_COLLECTION,
    DOCS_PATH,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
)
from onprem_rag.services.document_loader import load_all_documents
from onprem_rag.utils.logger import log_error
from onprem_rag.services.rag_engine import reset_collection_cache

EMBED_BATCH_SIZE = 32


def get_source_documents(docs_path: Path | None = None) -> list[Path]:
    """Return supported source documents that are available for indexing."""
    docs_path = Path(docs_path or DOCS_PATH)
    if not docs_path.exists():
        return []
    return sorted(
        file_path
        for file_path in docs_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in {
            ".pdf", ".docx", ".txt", ".md", ".markdown"
        }
    )


def delete_knowledge_document(filename: str) -> dict:
    """Delete one managed source document and its indexed chunks."""
    safe_name = Path(filename).name
    if safe_name != filename or Path(safe_name).suffix.lower() not in {
        ".pdf", ".docx", ".txt", ".md", ".markdown"
    }:
        raise ValueError("Invalid knowledge-base document name")

    source_path = DOCS_PATH / safe_name
    removed_source = False
    if source_path.exists():
        source_path.unlink()
        removed_source = True

    removed_chunks = 0
    if CHROMA_PATH.exists():
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            collection_names = {
                collection.name for collection in client.list_collections()
            }
            if CHROMA_COLLECTION in collection_names:
                collection = client.get_collection(CHROMA_COLLECTION)
                matches = collection.get(
                    where={"document_name": safe_name},
                    include=[],
                )
                removed_chunks = len(matches.get("ids", []))
                if removed_chunks:
                    collection.delete(where={"document_name": safe_name})
        except Exception as exc:
            raise RuntimeError(
                f"The source file was removed, but its index entries could not be deleted: {exc}"
            ) from exc

    reset_collection_cache()
    return {
        "ok": True,
        "filename": safe_name,
        "removed_source": removed_source,
        "removed_chunks": removed_chunks,
    }


def clear_knowledge_base() -> dict:
    """Delete all managed source documents and the complete vector index."""
    removed_files = []
    for source_path in get_source_documents():
        source_path.unlink()
        removed_files.append(source_path.name)

    if CHROMA_PATH.exists():
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            # A missing collection is already a valid cleared state.
            pass

    reset_collection_cache()
    return {"ok": True, "removed_files": removed_files}


# ── Text chunker ──────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        break_pos = text.rfind("\n\n", start, end)
        if break_pos == -1 or break_pos <= start:
            for sep in (". ", "! ", "? ", "\n"):
                pos = text.rfind(sep, start, end)
                if pos > start:
                    break_pos = pos + len(sep)
                    break
            else:
                break_pos = end
        chunk = text[start:break_pos].strip()
        if chunk:
            chunks.append(chunk)
        # Apply overlap only when the current chunk is larger than the
        # requested overlap. Advancing by one character for short sentences
        # creates thousands of near-duplicate micro-chunks.
        if break_pos - start > overlap:
            overlap_start = break_pos - overlap
            next_boundary = text.find(" ", overlap_start, break_pos)
            start = next_boundary + 1 if next_boundary >= 0 else overlap_start
        else:
            start = break_pos

    return [c for c in chunks if c.strip()]


_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)+)\s*[-.:)]?\s+"
    r"(?P<title>[A-Za-z][^\n]{1,140}?)\s*$"
)


def _clean_heading_title(value: str) -> str:
    """Remove table-of-contents leaders and trailing page numbers."""
    value = re.sub(r"\.{3,}\s*\d+\s*$", "", value)
    return value.strip(" .:-")


def _structured_chunks(
    document: dict,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split a document into passages with real page and section metadata."""
    pages = document.get("pages") or []
    if not pages:
        return [
            {
                "text": text,
                "page_start": "",
                "page_end": "",
                "section_number": "",
                "section_title": "",
                "content_type": "body",
            }
            for text in _split_text(document["text"], chunk_size, overlap)
        ]

    sections = []
    current = {
        "lines": [],
        "pages": [],
        "section_number": "",
        "section_title": "Front matter",
        "content_type": "front_matter",
    }

    def flush():
        text = "\n".join(current["lines"]).strip()
        if text:
            sections.append({
                "text": text,
                "pages": list(current["pages"]),
                "section_number": current["section_number"],
                "section_title": current["section_title"],
                "content_type": current["content_type"],
            })

    for page in pages:
        page_number = int(page["page_number"])
        for raw_line in page["text"].splitlines():
            line = raw_line.strip()
            if not line:
                continue
            heading = _NUMBERED_HEADING.match(line)
            if heading:
                title = _clean_heading_title(heading.group("title"))
                # Ignore contents entries with dot leaders; they describe a
                # later section rather than starting one on the current page.
                if title and "..." not in line:
                    flush()
                    current = {
                        "lines": [f"{heading.group('number')} {title}"],
                        "pages": [page_number],
                        "section_number": heading.group("number"),
                        "section_title": title,
                        "content_type": "body",
                    }
                    continue
            current["lines"].append(line)
            if page_number not in current["pages"]:
                current["pages"].append(page_number)
    flush()

    # Short policies, objectives, and matrices may contain useful prose without
    # numbered headings. If no body section was detected, the complete document
    # is substantive content rather than front matter.
    if sections and not any(
        section["content_type"] == "body" for section in sections
    ):
        for section in sections:
            section["content_type"] = "body"
            section["section_title"] = ""

    passages = []
    for section in sections:
        for text in _split_text(section["text"], chunk_size, overlap):
            passages.append({
                "text": text,
                "page_start": min(section["pages"]) if section["pages"] else "",
                "page_end": max(section["pages"]) if section["pages"] else "",
                "section_number": section["section_number"],
                "section_title": section["section_title"],
                "content_type": section["content_type"],
            })
    return passages


# ── Batch embedding ───────────────────────────────────────────────────────────

def _get_embeddings_batch(texts: list) -> list:
    """
    Embed a list of texts in one call. Uses ollama.embed() for batch support.
    Falls back to individual calls if the installed client is older.
    """
    client = ollama.Client(host=OLLAMA_BASE_URL)
    try:
        response = client.embed(model=EMBED_MODEL, input=texts)
        return response["embeddings"]
    except (AttributeError, TypeError, KeyError):
        return [
            client.embeddings(model=EMBED_MODEL, prompt=t)["embedding"]
            for t in texts
        ]


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _get_chroma_collection(reset: bool = False):
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest_documents(
    docs_path: Path = DOCS_PATH,
    reset: bool = True,
    progress_callback=None,
    chunk_progress_callback=None,
) -> dict:
    """
    Load, chunk, embed, and store all documents in ChromaDB.

    Args:
        docs_path:               Folder containing approved documents.
        reset:                   If True, clears and rebuilds the collection.
        progress_callback:       Optional callable(message: str) for text updates.
        chunk_progress_callback: Optional callable(current: int, total: int)
                                 for progress bar updates.
    Returns:
        {'documents': int, 'chunks': int, 'errors': list}
    """
    docs_path = Path(docs_path)

    def _log(msg: str):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    # 1. Verify Ollama
    _log("Connecting to Ollama...")
    try:
        ollama.Client(host=OLLAMA_BASE_URL).list()
    except Exception as e:
        msg = (
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running. Error: {e}"
        )
        log_error("ingest — Ollama connection check", e)
        raise ConnectionError(msg)

    # 2. Load documents
    if not docs_path.exists() or not any(docs_path.iterdir()):
        _log("No documents found.")
        return {"documents": 0, "chunks": 0, "errors": []}

    _log("Reading documents...")
    all_docs = load_all_documents(docs_path)

    if not all_docs:
        _log("No documents could be loaded.")
        return {"documents": 0, "chunks": 0, "errors": []}

    # 3. Split all docs into chunks upfront so we know the total
    _log(f"Loaded {len(all_docs)} document(s) — splitting into sections...")
    doc_chunks = []
    for doc in all_docs:
        chunks = _structured_chunks(doc)
        for idx, passage in enumerate(chunks):
            doc_chunks.append((doc, passage, idx, len(chunks)))

    total = len(doc_chunks)
    _log(f"Found {total} sections across {len(all_docs)} document(s). Starting embedding...")

    # 4. Init ChromaDB
    collection = _get_chroma_collection(reset=reset)

    # 5. Embed in batches and store
    errors = []
    stored = 0
    skipped = 0

    _log("Embedding sections (this may take a few minutes for large documents)...")

    i = 0
    while i < total:
        batch = doc_chunks[i: i + EMBED_BATCH_SIZE]

        # In update (non-reset) mode, skip already-existing chunks
        if not reset:
            filtered = []
            for item in batch:
                doc, passage, idx, _ = item
                chunk_id = f"{doc['metadata']['document_name']}::chunk_{idx}"
                if not collection.get(ids=[chunk_id])["ids"]:
                    filtered.append(item)
                else:
                    skipped += 1
            batch = filtered

        if batch:
            batch_texts = [item[1]["text"] for item in batch]
            try:
                embeddings = _get_embeddings_batch(batch_texts)
            except Exception as e:
                for item in batch:
                    doc, passage, idx, _ = item
                    errors.append(f"Embedding failed for chunk {idx} of '{doc['metadata']['document_name']}': {e}")
                log_error("ingest — batch embedding", e)
                i += EMBED_BATCH_SIZE
                if chunk_progress_callback:
                    chunk_progress_callback(min(i, total), total)
                continue

            ids, vecs, docs_out, metas = [], [], [], []
            for item, embedding in zip(batch, embeddings):
                doc, passage, idx, n_chunks = item
                doc_name = doc["metadata"]["document_name"]
                chunk_id = f"{doc_name}::chunk_{idx}"
                meta = {k: str(v) for k, v in doc["metadata"].items()}
                meta["chunk_index"] = str(idx)
                meta["chunk_count"] = str(n_chunks)
                for key in (
                    "page_start",
                    "page_end",
                    "section_number",
                    "section_title",
                    "content_type",
                ):
                    meta[key] = str(passage[key])
                ids.append(chunk_id)
                vecs.append(embedding)
                docs_out.append(passage["text"])
                metas.append(meta)

            collection.upsert(ids=ids, embeddings=vecs, documents=docs_out, metadatas=metas)
            stored += len(batch)

        i += EMBED_BATCH_SIZE
        done = min(i, total)
        if chunk_progress_callback:
            chunk_progress_callback(done, total)

    _log(f"All done! {len(all_docs)} document(s) indexed, {stored} sections stored.")
    if errors:
        _log(f"  {len(errors)} section(s) had errors and were skipped.")

    reset_collection_cache()
    return {"documents": len(all_docs), "chunks": stored, "errors": errors}


def rebuild_index(progress_callback=None, chunk_progress_callback=None) -> dict:
    """Full rebuild: clear ChromaDB and re-ingest all approved documents."""
    if not get_source_documents():
        raise ValueError(
            "No supported documents are available. Upload at least one PDF, "
            "DOCX, TXT, or Markdown file before rebuilding the knowledge base."
        )
    return ingest_documents(
        reset=True,
        progress_callback=progress_callback,
        chunk_progress_callback=chunk_progress_callback,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        result = rebuild_index()
        if result["errors"]:
            print(f"Completed with {len(result['errors'])} error(s).")
            sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)
