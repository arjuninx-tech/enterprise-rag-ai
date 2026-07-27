"""
rag_engine.py - RAG Engine for On-Prem RAG Assistant
Retrieves relevant chunks from ChromaDB and calls Ollama LLM.
Supports both streaming and non-streaming responses.
"""

import chromadb
import ollama
import re

from onprem_rag.config import (
    CHROMA_PATH,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    SUMMARY_TOP_K,
    TOP_K,
    MAX_COSINE_DISTANCE,
)
from onprem_rag.services.prompts import (
    build_rag_prompt,
    build_prompt_with_attachments,
    build_system_prompt,
    profile_fallback,
)
from onprem_rag.utils.logger import log_question, log_error


# ── Cached ChromaDB client ────────────────────────────────────────────────────
_chroma_client = None
_chroma_collection = None


def _get_collection():
    global _chroma_client, _chroma_collection
    if not CHROMA_PATH.exists():
        return None
    try:
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        if _chroma_collection is None:
            _chroma_collection = _chroma_client.get_collection(CHROMA_COLLECTION)
        return _chroma_collection
    except Exception:
        _chroma_client = None
        _chroma_collection = None
        return None


def reset_collection_cache():
    global _chroma_client, _chroma_collection
    _chroma_client = None
    _chroma_collection = None


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_question(question):
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.embeddings(model=EMBED_MODEL, prompt=question)
    return response["embedding"]


# ── Vector search ─────────────────────────────────────────────────────────────

_BROAD_QUERY_TERMS = (
    "summarize",
    "summary",
    "overview",
    "entire",
    "whole",
    "all documents",
    "across documents",
    "compare",
    "explain",
    "learn",
)

_AMBIGUOUS_REQUESTS = {
    "help",
    "help me",
    "assist",
    "assist me",
    "please help",
    "what can you do",
}


def clarification_response(question: str) -> str | None:
    """Return a useful clarification prompt for requests with no actionable subject."""
    normalized = " ".join(question.lower().strip().rstrip("?.!").split())
    if normalized in _AMBIGUOUS_REQUESTS:
        return (
            "What would you like help with—summarizing a document, finding a "
            "requirement, comparing documents, or extracting specific information?"
        )
    return None


def _retrieval_limit(question: str) -> int:
    """Use wider evidence coverage for broad synthesis requests."""
    normalized = question.lower()
    if any(term in normalized for term in _BROAD_QUERY_TERMS):
        return max(TOP_K, SUMMARY_TOP_K)
    return TOP_K


def _normalize_grounded_answer(
    answer: str,
    fallback: str,
    has_evidence: bool,
) -> tuple[str, bool]:
    """Remove a contradictory fallback prefix when a grounded answer follows it."""
    stripped_answer = answer.strip()
    fallback_text = fallback.strip()
    if not stripped_answer.lower().startswith(fallback_text.lower()):
        return stripped_answer, True

    continuation = stripped_answer[len(fallback_text):]
    continuation = re.sub(r"^[\s\-–—:;,.]+", "", continuation)
    continuation = re.sub(
        r"^(however|but|nevertheless|still)\b[\s,:;-]*",
        "",
        continuation,
        flags=re.IGNORECASE,
    ).strip()

    # A meaningful continuation backed by retrieved passages is preferable to
    # discarding the complete answer because the model repeated the fallback
    # preamble. Very short continuations remain classified as unavailable.
    if has_evidence and len(continuation.split()) >= 8:
        return continuation, True
    return fallback_text, False


_SEARCH_STOP_WORDS = {
    "a", "about", "and", "are", "can", "could", "do", "for", "from",
    "help", "i", "in", "is", "it", "learn", "like", "me", "of", "please",
    "the", "to", "u", "what", "with",
}


def _search_terms(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", value.lower())
        if token not in _SEARCH_STOP_WORDS and len(token) > 1
    }


def _document_alias_terms(document_name: str) -> set[str]:
    """Derive searchable title terms and natural acronyms such as OSP."""
    stem = re.sub(r"\.[^.]+$", "", document_name)
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", stem)
        if len(word) > 1 and word.lower() not in _SEARCH_STOP_WORDS
    ]
    aliases = set(words)
    if len(words) >= 2:
        acronym = "".join(word[0] for word in words)
        if 2 <= len(acronym) <= 10:
            aliases.add(acronym)
    return aliases


def _lexical_score(question: str, text: str, metadata: dict) -> float:
    """Score exact terminology, identifiers, headings, and document titles."""
    query_terms = _search_terms(question)
    if not query_terms:
        return 0.0
    text_terms = _search_terms(text)
    name_terms = _document_alias_terms(metadata.get("document_name", ""))
    heading_terms = _search_terms(
        f"{metadata.get('section_number', '')} {metadata.get('section_title', '')}"
    )
    score = (
        len(query_terms & text_terms)
        + 5 * len(query_terms & heading_terms)
        + 8 * len(query_terms & name_terms)
    )
    identifiers = {term for term in query_terms if len(term) <= 10}
    score += 8 * len(identifiers & name_terms)
    return float(score)


def _source_label(document_name: str, metadata: dict) -> str:
    """Build a citation from user-meaningful document structure."""
    parts = [document_name]
    section_number = metadata.get("section_number", "").strip()
    section_title = metadata.get("section_title", "").strip()
    if section_number:
        section = f"Section {section_number}"
        if section_title:
            section += f": {section_title}"
        parts.append(section)
    elif section_title and section_title != "Front matter":
        parts.append(section_title)
    page_start = metadata.get("page_start", "").strip()
    page_end = metadata.get("page_end", "").strip()
    if page_start:
        page_label = f"page {page_start}"
        if page_end and page_end != page_start:
            page_label = f"pages {page_start}–{page_end}"
        parts.append(page_label)
    return " — ".join(parts)


def _search_chunks_vector_only(collection, embedding, question: str = ""):
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(_retrieval_limit(question), collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc_text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Chroma cosine distance increases as relevance decreases. Rejecting
        # weak matches prevents unrelated questions from receiving arbitrary context.
        if distance > MAX_COSINE_DISTANCE:
            continue
        doc_name = meta.get("document_name", "Unknown Document")
        doc_type = meta.get("document_type", "")
        standard = meta.get("standard", "")
        chunk_idx = meta.get("chunk_index", "")

        source_parts = [doc_name]
        if standard and standard not in ("Unknown", ""):
            source_parts.append(standard)
        if doc_type and doc_type not in ("Unknown", ""):
            source_parts.append(doc_type)
        if chunk_idx:
            source_parts.append(f"section {int(chunk_idx) + 1}")

        chunks.append({
            "text": doc_text,
            "distance": distance,
            "source": " — ".join(source_parts),
            "document_name": doc_name,
            "metadata": meta,
        })
    return chunks


# ── Source formatting ─────────────────────────────────────────────────────────

def _search_chunks(collection, embedding, question: str = ""):
    """Combine semantic and exact-term rankings into grounded passages."""
    limit = min(_retrieval_limit(question), collection.count())
    candidate_count = min(max(limit * 4, 20), collection.count())
    results = collection.query(
        query_embeddings=[embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    candidates = {}
    vector_rank = {}
    for rank, (chunk_id, text, meta, distance) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), 1):
        if distance <= MAX_COSINE_DISTANCE:
            candidates[chunk_id] = {
                "text": text,
                "metadata": meta,
                "distance": distance,
            }
            vector_rank[chunk_id] = rank

    all_records = collection.get(include=["documents", "metadatas"])
    lexical = []
    for chunk_id, text, meta in zip(
        all_records["ids"],
        all_records["documents"],
        all_records["metadatas"],
    ):
        score = _lexical_score(question, text, meta)
        # One incidental shared word is not enough to establish relevance.
        # Strong semantic candidates are already included above; lexical-only
        # candidates must match at least two weighted terms.
        if score >= 2:
            lexical.append((chunk_id, score, text, meta))
            candidates.setdefault(chunk_id, {
                "text": text,
                "metadata": meta,
                "distance": 1.0,
            })
    lexical.sort(key=lambda item: (-item[1], item[0]))
    lexical_rank = {item[0]: rank for rank, item in enumerate(lexical, 1)}
    lexical_scores = {item[0]: item[1] for item in lexical}

    query_terms = _search_terms(question)
    title_match_counts = {
        document_name: len(
            query_terms & _document_alias_terms(document_name)
        )
        for document_name in {
            meta.get("document_name", "")
            for meta in all_records["metadatas"]
        }
    }
    strongest_title_match = max(title_match_counts.values(), default=0)
    matched_documents = {
        document_name
        for document_name, match_count in title_match_counts.items()
        if match_count == strongest_title_match and match_count > 0
    }
    if matched_documents:
        scoped = {
            chunk_id: candidate
            for chunk_id, candidate in candidates.items()
            if candidate["metadata"].get("document_name") in matched_documents
        }
        filtered = {}
        for document_name in matched_documents:
            document_candidates = {
                chunk_id: candidate
                for chunk_id, candidate in scoped.items()
                if candidate["metadata"].get("document_name") == document_name
            }
            body_candidates = {
                chunk_id: candidate
                for chunk_id, candidate in document_candidates.items()
                if candidate["metadata"].get("content_type") != "front_matter"
            }
            filtered.update(body_candidates or document_candidates)
        candidates = filtered

    ranked = []
    for chunk_id, candidate in candidates.items():
        score = 0.0
        if chunk_id in vector_rank:
            score += 1 / (60 + vector_rank[chunk_id])
        if chunk_id in lexical_rank:
            score += 1 / (60 + lexical_rank[chunk_id])
            score += min(lexical_scores[chunk_id], 30) / 300
        content_type = candidate["metadata"].get("content_type", "")
        if content_type == "body":
            score += 0.02
        elif content_type == "front_matter":
            score -= 0.02
        ranked.append((score, chunk_id, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    chunks = []
    for _, _, candidate in ranked[:limit]:
        meta = candidate["metadata"]
        doc_name = meta.get("document_name", "Unknown Document")
        chunks.append({
            "text": candidate["text"],
            "distance": candidate["distance"],
            "source": _source_label(doc_name, meta),
            "document_name": doc_name,
            "metadata": meta,
        })
    return chunks


def _format_sources(chunks):
    seen = set()
    sources = []
    for chunk in chunks:
        key = chunk["source"]
        if key not in seen:
            seen.add(key)
            meta = chunk["metadata"]
            sources.append({
                "document_name": chunk["document_name"],
                "document_type": meta.get("document_type", ""),
                "standard":      meta.get("standard", ""),
                "version":       meta.get("version", ""),
                "status":        meta.get("status", ""),
                "source_label":  chunk["source"],
                "page_start":    meta.get("page_start", ""),
                "page_end":      meta.get("page_end", ""),
                "section_number": meta.get("section_number", ""),
                "section_title": meta.get("section_title", ""),
            })
    return sources


# ── Retrieval (embed + search, called separately so UI can show progress) ─────

def retrieve(question: str) -> dict:
    question = question.strip()
    if not question:
        return {"chunks": [], "found": False, "error": None, "collection_empty": False}
    if clarification_response(question):
        return {"chunks": [], "found": False, "error": None, "collection_empty": False}

    collection = _get_collection()
    if collection is None:
        return {"chunks": [], "found": False, "error": None, "collection_empty": True}

    try:
        count = collection.count()
    except Exception:
        reset_collection_cache()
        return {"chunks": [], "found": False, "error": None, "collection_empty": True}

    if count == 0:
        return {"chunks": [], "found": False, "error": None, "collection_empty": True}

    try:
        embedding = _embed_question(question)
    except Exception as e:
        log_error("rag_engine — embed question", e)
        return {"chunks": [], "found": False, "error": str(e), "collection_empty": False}

    try:
        chunks = _search_chunks(collection, embedding, question)
    except Exception as e:
        log_error("rag_engine — vector search", e)
        return {"chunks": [], "found": False, "error": str(e), "collection_empty": False}

    return {"chunks": chunks, "found": bool(chunks), "error": None, "collection_empty": False}


# ── Streaming ask ─────────────────────────────────────────────────────────────

def ask_stream(
    question,
    chunks=None,
    model=None,
    attached_context: str = "",
    agent_profile: dict | None = None,
):
    """
    Stream an LLM answer.

    Parameters
    ----------
    question          : user's question / request
    chunks            : pre-retrieved KB chunks (None = auto-retrieve)
    model             : Ollama model tag override
    attached_context  : formatted text from in-memory attached documents;
                        when non-empty the attachment prompt + system prompt
                        are used instead of the standard RAG prompt.
    """
    question = question.strip()
    active_model = model or LLM_MODEL
    result_holder = {}
    has_attachments = bool(attached_context)
    fallback_response = profile_fallback(agent_profile)

    def _run():
        if not question:
            yield "Please enter a question."
            result_holder.update({"answer": "Please enter a question.", "sources": [], "found": False, "error": None})
            return

        clarification = clarification_response(question)
        if clarification:
            yield clarification
            result_holder.update({
                "answer": clarification,
                "sources": [],
                "found": False,
                "error": None,
                "model": active_model,
            })
            return

        resolved_chunks = chunks

        # ── KB retrieval (skip if we have attachments and no explicit chunks) ──
        if resolved_chunks is None and not has_attachments:
            collection = _get_collection()
            if collection is None:
                msg = "The knowledge base is empty. Please upload documents and click Build Index first."
                yield msg
                result_holder.update({"answer": msg, "sources": [], "found": False, "error": None})
                return
            try:
                if collection.count() == 0:
                    msg = "The knowledge base is empty. Please upload documents and click Build Index first."
                    yield msg
                    result_holder.update({"answer": msg, "sources": [], "found": False, "error": None})
                    return
            except Exception:
                reset_collection_cache()
                msg = "The knowledge base is empty. Please upload documents and click Build Index first."
                yield msg
                result_holder.update({"answer": msg, "sources": [], "found": False, "error": None})
                return
            try:
                embedding = _embed_question(question)
                resolved_chunks = _search_chunks(collection, embedding, question)
            except Exception as e:
                msg = f"Could not connect to Ollama. Error: {e}"
                yield msg
                result_holder.update({"answer": msg, "sources": [], "found": False, "error": str(e)})
                return

        if not resolved_chunks and not has_attachments:
            yield fallback_response
            log_question(question, fallback_response, [], found=False)
            result_holder.update({"answer": fallback_response, "sources": [], "found": False, "error": None})
            return

        # ── Build prompt ───────────────────────────────────────────────────────
        if has_attachments:
            prompt        = build_prompt_with_attachments(question, attached_context, resolved_chunks)
            active_system = build_system_prompt(agent_profile)
        else:
            prompt        = build_rag_prompt(question, resolved_chunks)
            active_system = build_system_prompt(agent_profile)

        import time
        gen_start = time.time()
        try:
            client = ollama.Client(host=OLLAMA_BASE_URL)
            stream = client.chat(
                model=active_model,
                messages=[
                    {"role": "system", "content": active_system},
                    {"role": "user",   "content": prompt},
                ],
                stream=True,
                options={
                    "temperature": 0.1,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_predict": OLLAMA_NUM_PREDICT,
                },
            )
            full_answer = ""
            token_count = 0
            for chunk_resp in stream:
                token = chunk_resp["message"]["content"]
                full_answer += token
                token_count += 1
                yield token
        except Exception as e:
            msg = f"LLM call failed. Make sure Ollama is running and model '{active_model}' is pulled.\nError: {e}"
            yield msg
            log_error("rag_engine — LLM stream", e)
            result_holder.update({"answer": msg, "sources": [], "found": False, "error": str(e), "model": active_model})
            return

        gen_time = time.time() - gen_start
        tokens_per_sec = round(token_count / gen_time, 1) if gen_time > 0 else 0

        final_answer, found = _normalize_grounded_answer(
            full_answer,
            fallback_response,
            has_evidence=bool(resolved_chunks or attached_context),
        )
        sources = _format_sources(resolved_chunks or []) if found else []
        log_question(question, final_answer, sources, found=found)
        result_holder.update({
            "answer": final_answer,
            "sources": sources,
            "found": found,
            "error": None,
            "model": active_model,
            "gen_time": round(gen_time, 1),
            "token_count": token_count,
            "tokens_per_sec": tokens_per_sec,
        })

    class _StreamGen:
        def __init__(self, gen, holder):
            self._gen = gen
            self._holder = holder
        def __iter__(self):
            return self
        def __next__(self):
            return next(self._gen)
        @property
        def result(self):
            return self._holder

    return _StreamGen(_run(), result_holder)


# ── Non-streaming ask ─────────────────────────────────────────────────────────

def ask(question):
    gen = ask_stream(question)
    full = "".join(gen)
    return gen.result if gen.result else {"answer": full, "sources": [], "found": True, "error": None}


# ── Utility ───────────────────────────────────────────────────────────────────

def check_ollama_status():
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        models = client.list()
        model_names = [m["name"] for m in models.get("models", [])]
        missing = [m for m in [LLM_MODEL, EMBED_MODEL] if not any(m in n for n in model_names)]
        if missing:
            return False, f"Models not pulled: {', '.join(missing)}"
        return True, f"Ollama OK — models ready: {LLM_MODEL}, {EMBED_MODEL}"
    except Exception:
        return False, f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Start with: ollama serve"


def get_kb_stats():
    from onprem_rag.services.ingest import get_source_documents

    source_documents = [path.name for path in get_source_documents()]
    try:
        collection = _get_collection()
        if collection is None:
            return {
                "indexed": False,
                "chunk_count": 0,
                "documents": [],
                "source_documents": source_documents,
            }
        count = collection.count()
        if count == 0:
            return {
                "indexed": False,
                "chunk_count": 0,
                "documents": [],
                "source_documents": source_documents,
            }
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        doc_names = sorted({m.get("document_name", "?") for m in all_meta})
        return {
            "indexed": True,
            "chunk_count": count,
            "documents": doc_names,
            "source_documents": source_documents,
        }
    except Exception:
        reset_collection_cache()
        return {
            "indexed": False,
            "chunk_count": 0,
            "documents": [],
            "source_documents": source_documents,
        }
