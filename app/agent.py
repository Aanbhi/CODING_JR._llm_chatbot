# app/agent.py
"""
Agentic AI module:
- Planner LLM decides which "tool" to use:
  - RAG_Search: perform semantic search over stored doc chunks and answer using retrieved context.
  - LLM_Response: answer directly with the LLM.
- One-step agent (plan -> act -> answer) with trace. Can be extended to multi-step loops.
- Avoids circular import with main.py by re-implementing minimal utilities and
  loading embeddings/metadata from disk at call time.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
import requests
import pickle

# ---- Shared config (must match main.py) ----
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is not set")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

DB_DIR = os.getenv("DOC_STORE_DIR", "/app/data")
EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(DB_DIR, "metadata.pkl")

# ---- Minimal OpenAI helpers ----
def _call_openai_chat(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    r = requests.post(OPENAI_CHAT_URL, json=payload, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI Chat API error: {r.status_code} {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

def _call_openai_embeddings(texts: List[str]) -> List[List[float]]:
    results = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"input": batch, "model": EMBEDDING_MODEL}
        r = requests.post(OPENAI_EMBED_URL, headers=HEADERS, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI Embeddings API error: {r.status_code} {r.text}")
        data = r.json()
        for item in data["data"]:
            results.append(item["embedding"])
        time.sleep(0.05)
    return results

# ---- DB loading + simple search (fresh each call to avoid circular imports) ----
def _load_db():
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(METADATA_FILE):
        return None, []
    emb = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "rb") as f:
        meta = pickle.load(f)
    return emb, meta

def _semantic_search_local(query_emb: np.ndarray, top_k: int = 4):
    emb, meta = _load_db()
    if emb is None or len(meta) == 0:
        return []
    n_neighbors = min(top_k, len(emb))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(emb)
    distances, idxs = knn.kneighbors(query_emb.reshape(1, -1), n_neighbors=n_neighbors)
    distances = distances[0].tolist()
    idxs = idxs[0].tolist()
    results = []
    for d, i in zip(distances, idxs):
        item = meta[i]
        results.append({"score": float(d), "id": item["id"], "text": item["text"], "source": item.get("source")})
    return results

# ---- Agent ----
PLANNER_SYSTEM_PROMPT = (
    "You are an AI agent with tools:\n"
    "- RAG_Search: search uploaded docs (vector search) and answer using retrieved context.\n"
    "- LLM_Response: answer directly with your own knowledge.\n\n"
    "Return STRICT JSON ONLY as one line. Use either:\n"
    '{ "action": "RAG_Search", "input": "<rewrite of user query for retrieval>" }\n'
    'OR { "action": "LLM_Response", "input": "<short direct answer instruction>" }\n'
    "If the user asks about uploaded files, prefer RAG_Search."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context if available. "
    "If the context does not contain the answer, say you don't know and offer a brief suggestion."
)

def agentic_response(user_message: str, use_rag: bool = True, top_k: int = 4) -> Dict[str, Any]:
    """
    One-step agent:
      1) Planner chooses an action in JSON.
      2) Execute action:
         - If RAG_Search and use_rag True: perform local vector search and answer using context.
         - If LLM_Response: answer directly.
      3) Return final answer and a trace of decisions.
    """
    # 1) Plan
    planner_raw = _call_openai_chat(PLANNER_SYSTEM_PROMPT, f"User: {user_message}")
    trace: List[Any] = [{"planner_raw": planner_raw}]

    # 2) Parse JSON plan (robustly)
    plan: Optional[Dict[str, Any]] = None
    try:
        plan = json.loads(planner_raw.strip())
    except Exception:
        # If the model failed JSON, fallback to direct LLM answer
        final_ans = _call_openai_chat(ANSWER_SYSTEM_PROMPT, f"User: {user_message}")
        trace.append({"note": "Planner JSON parse failed; used direct answer."})
        return {"answer": final_ans, "trace": trace}

    trace.append({"planner": plan})

    # 3) Execute plan
    if "action" in plan and plan["action"] == "RAG_Search" and use_rag:
        # compute embedding for retrieval input (can be the user msg or plan input)
        query_for_search = plan.get("input") or user_message
        q_emb = _call_openai_embeddings([query_for_search])[0]
        q_emb_np = np.array(q_emb, dtype="float32")
        retrieved = _semantic_search_local(q_emb_np, top_k=top_k)
        context_str = "\n\n---\n\n".join([r["text"] for r in retrieved]) if retrieved else ""
        user_prompt = (
            (f"Context:\n{context_str}\n\n" if context_str else "") +
            f"User question:\n{user_message}\n\n"
            "Answer concisely. Cite sources by number (Source 1, Source 2) when you use them."
        )
        final_ans = _call_openai_chat(ANSWER_SYSTEM_PROMPT, user_prompt)
        return {"answer": final_ans, "retrieved": retrieved, "trace": trace}

    # Otherwise default to LLM_Response (or if use_rag is False)
    if "action" in plan and plan["action"] == "LLM_Response":
        instruction = plan.get("input") or f"Answer the user's question: {user_message}"
        final_ans = _call_openai_chat(ANSWER_SYSTEM_PROMPT, instruction)
        return {"answer": final_ans, "trace": trace}

    # Fallback: direct answer
    final_ans = _call_openai_chat(ANSWER_SYSTEM_PROMPT, f"User: {user_message}")
    trace.append({"note": "Fallback direct answer."})
    return {"answer": final_ans, "trace": trace}
