# app/streamlit_app.py
import streamlit as st
import requests
import os
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:10000")

st.set_page_config(page_title="LLM Chatbot + RAG", page_icon="🤖", layout="wide")
st.title("🤖 LLM Chatbot & Attachment Analyzer (with RAG)")

tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📎 Upload & Analyze", "📚 Stored Chunks"])

with tab1:
    st.subheader("Chat with the LLM")
    user_input = st.text_area("Your message:", height=120)
    use_rag = st.checkbox("Use RAG (retrieve from uploaded docs)", value=True)
    top_k = st.slider("Top-k retrieved passages", min_value=1, max_value=8, value=4)
    if st.button("Send", key="chatbtn"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            with st.spinner("Getting response..."):
                payload = {"message": user_input, "use_rag": bool(use_rag), "top_k": int(top_k)}
                try:
                    resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown("### Response")
                        st.write(data.get("answer", "No answer"))
                        if use_rag and "retrieved" in data:
                            st.markdown("### Retrieved Passages")
                            for i, r in enumerate(data["retrieved"], start=1):
                                st.markdown(f"**Source {i}** — score: {r['score']:.4f} — id: {r['id']}")
                                st.write(r["text"][:1000])
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab2:
    st.subheader("Upload a file for analysis and indexing")
    uploaded_file = st.file_uploader("Choose a file (PDF, image, or text)", type=None)
    if uploaded_file is not None:
        if st.button("Analyze & Index", key="uploadbtn"):
            with st.spinner("Uploading and analyzing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success("Upload complete")
                        st.json(data)
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab3:
    st.subheader("Stored chunks (preview)")
    col1, col2 = st.columns([1,3])
    with col1:
        limit = st.slider("How many items to list", min_value=1, max_value=200, value=10)
        if st.button("Refresh list"):
            try:
                resp = requests.get(f"{BACKEND_URL}/docs/list?limit={limit}", timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    st.write(f"Total chunks stored: {data.get('count')}")
                    items = data.get("items", [])
                    st.session_state["items"] = items
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
    with col2:
        items = st.session_state.get("items", [])
        if items:
            for it in items:
                st.markdown(f"**ID:** {it['id']}  — source: {it.get('source')} — ts: {time.ctime(it.get('ts',0))}")
                st.write(it['text'][:800])
                st.write("---")
        else:
            st.info("No items loaded. Click 'Refresh list' to load stored chunks.")
