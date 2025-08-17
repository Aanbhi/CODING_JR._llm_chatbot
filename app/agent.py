
import json
from .main import call_openai_chat, semantic_search, call_openai_embeddings
import numpy as np

def agentic_response(user_message: str, use_rag: bool = True, top_k: int = 4):
    """
    Simple agent loop: decide whether to search docs (RAG) or just answer directly.
    """
    system_prompt = (
        "You are an AI agent with tools:\n"
        "- RAG_Search: search uploaded docs.\n"
        "- LLM_Response: call the LLM directly.\n"
        "Respond in strict JSON with keys {action, input} or {final_answer}."
    )
    # Step 1: planner
    planner_output = call_openai_chat(system_prompt, f"User asked: {user_message}")
    try:
        planner = json.loads(planner_output)
    except Exception:
        return {"answer": planner_output, "trace": ["Planner failed JSON parse. Returning raw."]}

    trace = [planner]
    if "action" in planner:
        if planner["action"] == "RAG_Search" and use_rag:
            # embed and search
            q_emb = call_openai_embeddings([planner["input"]])[0]
            q_emb_np = np.array(q_emb).astype("float32")
            retrieved = semantic_search(q_emb_np, top_k=top_k)
            context_str = "\n".join([r["text"] for r in retrieved])
            final_ans = call_openai_chat("You are an assistant.", f"Context:\n{context_str}\n\nUser: {user_message}")
            return {"answer": final_ans, "retrieved": retrieved, "trace": trace}
        elif planner["action"] == "LLM_Response":
            final_ans = call_openai_chat("You are an assistant.", planner["input"])
            return {"answer": final_ans, "trace": trace}
    elif "final_answer" in planner:
        return {"answer": planner["final_answer"], "trace": trace}

    return {"answer": "Agent could not decide.", "trace": trace}
