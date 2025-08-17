# app/main.py
import os
import io
import uuid
import json
import pickle
import time
import magic
import fitz                 # PyMuPDF
from PIL import Image
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException, Query as FastAPIQuery
from pydantic import BaseModel
import requests
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ----------------------
# Config & persistence
# ----------------------
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embeddings model (change if needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is not set")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

# Local persistent store (simple)
DB_DIR = os.getenv("DOC_STORE_DIR", "/app/data")
os.makedirs(DB_DIR, exist_ok=True)
EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(DB_DIR, "metadata.pkl")

# In-memory structures (will load from disk if present)
embeddings: Optional[np.ndarray] = None
metadata: List[Dict[str, Any]] = []

def save_db():
    """Persist embeddings + metadata to disk."""
    global embeddings, metadata
    if embeddings is not None:
        np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

def load_db():
    """Load embeddings + metadata from disk into memory."""
    global embeddings, metadata
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
    else:
        embeddings = None
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = []

# load on startup
load_db()

# Simple nearest-neighbor wrapper (re-fits on upsert)
_knn: Optional[NearestNeighbors] = None
def fit_knn():
    """Fit in-memory NearestNeighbors on current embeddings."""
    global _knn, embeddings
    if embeddings is None or len(embeddings) == 0:
        _knn = None
        return
    _knn = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric="cosine")
    _knn.fit(embeddings)

def semantic_search(query_emb: np.ndarray, top_k: int = 4):
    """
    Returns list of dicts for top_k results with fields: score (cosine distance), id, text, source.
    Lower score == more similar.
    """
    global _knn, embeddings, metadata
    if _knn is None:
        return []
    distances, idxs = _knn.kneighbors(query_emb.reshape(1, -1), n_neighbors=min(top_k, len(embeddings)))
    distances = distances[0].tolist()
    idxs = idxs[0].tolist()
    results = []
    for d, i in zip(distances, idxs):
        item = metadata[i]
        results.append({"score": float(d), "id": item["id"], "text": item["text"], "source": item.get("source")})
    return results

# ----------------------
# Utility functions (OpenAI)
# ----------------------
def call_openai_chat(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    r = requests.post(OPENAI_CHAT_URL, json=payload, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI Chat API error: {r.status_code} {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

def call_openai_embeddings(texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings endpoint via requests (batch). Returns list of vectors."""
    results = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        payload = {"input": batch, "model": EMBEDDING_MODEL}
        r = requests.post(OPENAI_EMBED_URL, headers=HEADERS, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI Embeddings API error: {r.status_code} {r.text}")
        data = r.json()
        for item in data["data"]:
            results.append(item["embedding"])
        time.sleep(0.05)  # small delay to be polite
    return results

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200):
    """
    Chunk text into overlapping windows of size `max_chars` with `overlap` characters overlap.
    Returns list of chunk strings.
    """
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = end - overlap
    return chunks

# ----------------------
# File parsing
# ----------------------
def ocr_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        text = page.get_text("text")
        if text and text.strip():
            pages_text.append(text)
        else:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            ocr_text = ocr_image_bytes(img_bytes)
            pages_text.append(ocr_text)
    return pages_text

# ----------------------
# Upsert embeddings (store chunks + embeddings)
# ----------------------
def upsert_document_chunks(chunks: List[str], source: str):
    """
    Create embeddings for chunks, append to DB, persist, and re-fit knn.
    Returns list of ids inserted.
    """
    global embeddings, metadata
    if not chunks:
        return []
    vecs = call_openai_embeddings(chunks)  # list of lists
    vecs_np = np.array(vecs).astype("float32")
    ids = []
    for chunk_text, vec in zip(chunks, vecs_np):
        doc_id = str(uuid.uuid4())
        meta = {"id": doc_id, "text": chunk_text, "source": source, "ts": time.time()}
        metadata.append(meta)
        if embeddings is None:
            embeddings = vec.reshape(1, -1)
        else:
            embeddings = np.vstack([embeddings, vec.reshape(1, -1)])
        ids.append(doc_id)
    save_db()
    fit_knn()
    return ids

# ----------------------
# FastAPI app & endpoints
# ----------------------
app = FastAPI()

class QueryIn(BaseModel):
    message: str
    use_rag: Optional[bool] = False
    top_k: Optional[int] = 4

@app.on_event("startup")
def startup_event():
    fit_knn()

@app.post("/chat")
def chat(q: QueryIn):
    """
    Chat endpoint.
    If q.use_rag is True and we have docs, performs semantic retrieval and includes top_k contexts.
    """
    try:
        if q.use_rag and embeddings is not None and len(metadata) > 0:
            q_emb = call_openai_embeddings([q.message])[0]
            q_emb_np = np.array(q_emb).astype("float32")
            retrieved = semantic_search(q_emb_np, top_k=q.top_k or 4)
            context_texts = []
            for idx, item in enumerate(retrieved):
                context_texts.append(f"Source {idx+1} (score: {item['score']:.4f}):\n{item['text']}")
            context_str = "\n\n---\n\n".join(context_texts)
            system_prompt = (
                "You are a helpful assistant that answers user questions using the provided context from documents. "
                "If the context does not contain the answer, say you don't know and provide a brief relevant suggestion."
            )
            user_prompt = (
                f"Context:\n{context_str}\n\nUser question:\n{q.message}\n\n"
                "Answer concisely, cite the matching sources by number (e.g., Source 1), and provide a short summary of reasoning."
            )
            answer = call_openai_chat(system_prompt, user_prompt)
            return {"answer": answer, "retrieved": retrieved}
        else:
            system_prompt = "You are a helpful assistant."
            answer = call_openai_chat(system_prompt, q.message)
            return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document, extract text, chunk it, embed and store chunks.
    Returns summary + ids of stored chunks and basic analysis (via LLM).
    """
    content = await file.read()
    mtype = magic.from_buffer(content, mime=True)
    result: Dict[str, Any] = {"filename": file.filename, "content_type": mtype}
    try:
        # Determine file type and extract text
        if mtype == "application/pdf" or file.filename.lower().endswith(".pdf"):
            pages = extract_text_from_pdf_bytes(content)
            full_text = "\n\n---PAGE---\n\n".join(pages).strip()
            chunks = chunk_text(full_text, max_chars=1000, overlap=200)
            ids = upsert_document_chunks(chunks, source=file.filename)
            summary_prompt = "This is a PDF document. Please summarize key points in 3-6 sentences."
            summary = call_openai_chat("You summarize documents.", summary_prompt + "\n\n" + full_text[:20000])
            result.update({
                "kind": "pdf",
                "pages": len(pages),
                "chunks_stored": len(chunks),
                "chunk_ids": ids,
                "summary": summary
            })
            return result

        elif (mtype and mtype.startswith("image/")) or any(file.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]):
            text = ocr_image_bytes(content).strip()
            chunks = chunk_text(text, max_chars=800, overlap=200)
            ids = upsert_document_chunks(chunks, source=file.filename)
            summary_prompt = "Describe the image, and summarize any extracted text."
            summary = call_openai_chat("You describe images and summarize extracted text.", summary_prompt + "\n\n" + text[:5000])
            result.update({
                "kind": "image",
                "chunks_stored": len(chunks),
                "chunk_ids": ids,
                "extracted_text_preview": text[:4000],
                "summary": summary
            })
            return result

        elif mtype in ("text/plain",) or file.filename.lower().endswith((".txt", ".md", ".csv")):
            text = content.decode(errors="ignore")
            chunks = chunk_text(text, max_chars=1200, overlap=300)
            ids = upsert_document_chunks(chunks, source=file.filename)
            summary_prompt = "Summarize the following text file."
            summary = call_openai_chat("You summarize plain text files.", summary_prompt + "\n\n" + text[:20000])
            result.update({
                "kind": "text",
                "chunks_stored": len(chunks),
                "chunk_ids": ids,
                "summary": summary
            })
            return result

        else:
            # attempt to parse as PDF
            try:
                pages = extract_text_from_pdf_bytes(content)
                full_text = "\n\n---PAGE---\n\n".join(pages).strip()
                chunks = chunk_text(full_text, max_chars=1000, overlap=200)
                ids = upsert_document_chunks(chunks, source=file.filename)
                summary = call_openai_chat("You summarize documents.", "Parsed unknown file as PDF.\n\n" + full_text[:20000])
                result.update({
                    "kind": "pdf_inferred",
                    "pages": len(pages),
                    "chunks_stored": len(chunks),
                    "chunk_ids": ids,
                    "summary": summary
                })
                return result
            except Exception:
                raise HTTPException(status_code=400, detail="Unsupported file type or failed to process.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs/list")
def list_docs(limit: int = FastAPIQuery(20, ge=1, le=200)):
    """Return metadata list for stored chunks (limited)."""
    global metadata
    return {"count": len(metadata), "items": metadata[:limit]}

# ----------------------
# Agent endpoint (Agentic AI)
# ----------------------
from .agent import agentic_response  # noqa: E402 (import after definitions to avoid circulars)

@app.post("/agent")
def agent_chat(q: QueryIn):
    """
    Agent endpoint: delegates to agentic_response (tool-using LLM loop).
    """
    try:
        result = agentic_response(q.message, use_rag=q.use_rag, top_k=q.top_k or 4)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
