"""
Agentic AI module:
- Planner agent (RAG vs direct LLM).
- Tool-calling agent (function calls: web_search, code_runner, data_converter).
- Both return final answer + trace.
"""

import os
import json
import time
import pickle
import requests
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.neighbors import NearestNeighbors

from .tools import TOOLS   # Task 5 tools

# ---- Shared config ----
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

# ---- OpenAI helpers ----
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

# ---- DB loading + semantic search ----
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
    results = []
    for d, i in zip(distances[0], idxs[0]):
        item = meta[i]
        results.append({"score": float(d), "id": item["id"], "text": item["text"], "source": item.get("source")})
    return results

# =========================
# 1. Planner Agent (RAG vs LLM)
# =========================
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

def planner_agent(user_message: str, use_rag: bool = True, top_k: int = 4) -> Dict[str, Any]:
    planner_raw = _call_openai_chat(PLANNER_SYSTEM_PROMPT, f"User: {user_message}")
    trace: List[Any] = [{"planner_raw": planner_raw}]
    try:
        plan = json.loads(planner_raw.strip())
    except Exception:
        final_ans = _call_openai_chat(ANSWER_SYSTEM_PROMPT, f"User: {user_message}")
        trace.append({"note": "Planner JSON parse failed"})
        return {"answer": final_ans, "trace": trace}

    trace.append({"planner": plan})

    if plan.get("action") == "RAG_Search" and use_rag:
        q_emb = _call_openai_embeddings([plan.get("input") or user_message])[0]
        q_emb_np = np.array(q_emb, dtype="float32")
        retrieved = _semantic_search_local(q_emb_np, top_k=top_k)
        context_str = "\n\n---\n\n".join([r["text"] for r in retrieved]) if retrieved else ""
        user_prompt = f"Context:\n{context_str}\n\nUser question:\n{user_message}"
        final_ans = _call_openai_chat(ANSWER_SYSTEM_PROMPT, user_prompt)
        return {"answer": final_ans, "retrieved": retrieved, "trace": trace}

    # Default to LLM response
    instruction = plan.get("input") or user_message
    final_ans = _call_openai_chat(ANSWER_SYSTEM_PROMPT, instruction)
    return {"answer": final_ans, "trace": trace}

# =========================
# 2. Tool Agent (function calling)
# =========================
TOOL_SYSTEM_PROMPT = """You are an agent that can answer OR call tools.
If a tool is needed, respond with JSON: {"tool": "<tool_name>", "args": {...}}
Available tools: web_search, code_runner, data_converter.
Otherwise, answer directly.
"""

def tool_agent(user_message: str) -> Dict[str, Any]:
    try:
        raw = _call_openai_chat(TOOL_SYSTEM_PROMPT, user_message, max_tokens=600)
    except Exception as e:
        return {"answer": f"LLM error: {e}", "trace": []}

    trace = [{"step": "llm_decision", "content": raw}]
    answer = raw

    try:
        parsed = json.loads(raw)
        if "tool" in parsed:
            tool = parsed["tool"]
            args = parsed.get("args", {})
            func = TOOLS.get(tool)
            if func:
                tool_result = func(**args)
                trace.append({"step": "tool_call", "tool": tool, "args": args, "result": tool_result})
                final = _call_openai_chat(
                    "You are a helpful assistant. Use the tool result to answer.",
                    f"User asked: {user_message}\n\nTool output:\n{tool_result}",
                    max_tokens=500
                )
                trace.append({"step": "final_answer", "content": final})
                answer = final
    except Exception:
        pass

    return {"answer": answer, "trace": trace}
