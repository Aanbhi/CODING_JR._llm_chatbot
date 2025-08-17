# app/streamlit_app.py
import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:10000")

st.set_page_config(page_title="Chatbot Clone", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Upload & Analyze", "Stored Chunks", "Agentic AI", "Agent Tools"])

# Shared function for API calls
def call_api(endpoint, payload=None, files=None):
    try:
        if files:
            resp = requests.post(f"{BACKEND_URL}{endpoint}", files=files, timeout=120)
        else:
            resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Error: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None

# --- Chatbot Page ---
if page == "Chatbot":
    st.title("Chat with AI")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        resp = call_api("/chat", {"message": user_input, "use_rag": True, "top_k": 4})
        if resp:
            st.session_state.chat_history.append({"role": "assistant", "content": resp.get("answer", "")})

    # Render chat
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["content"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["content"])

# --- Upload Page ---
elif page == "Upload & Analyze":
    st.title("Upload & Analyze Documents")
    uploaded_file = st.file_uploader("Upload a file (PDF, image, or text)")
    if uploaded_file and st.button("Analyze & Index"):
        files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
        data = call_api("/upload", files=files)
        if data:
            st.json(data)

# --- Stored Chunks Page ---
elif page == "Stored Chunks":
    st.title("Stored Chunks")
    limit = st.slider("Limit", 1, 100, 10)
    if st.button("Refresh"):
        try:
            resp = requests.get(f"{BACKEND_URL}/docs/list?limit={limit}", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                st.write(f"Total: {data.get('count')}")
                for item in data.get("items", []):
                    st.markdown(f"**{item['id']}** - {item['text'][:200]}...")
                    st.write("---")
        except Exception as e:
            st.error(e)

# --- Agentic AI Page ---
elif page == "Agentic AI":
    st.title("Agentic AI")
    user_msg = st.chat_input("Ask the Agent...")
    if user_msg:
        resp = call_api("/agent", {"message": user_msg, "use_rag": True, "top_k": 4})
        if resp:
            with st.chat_message("assistant"):
                st.write(resp.get("answer", "No answer"))
            if "trace" in resp:
                st.json(resp["trace"])

# --- Agent Tools Page ---
elif page == "Agent Tools":
    st.title("Agent Tools")
    user_msg = st.chat_input("Ask with Tools...")
    if user_msg:
        resp = call_api("/agent", {"message": user_msg, "use_rag": False})
        if resp:
            with st.chat_message("assistant"):
                st.write(resp.get("answer", "No answer"))
            if "trace" in resp:
                st.json(resp["trace"])
