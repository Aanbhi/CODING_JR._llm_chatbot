# app/streamlit_app.py
import os
import time
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:10000")

# ---------- Page config ----------
st.set_page_config(page_title="LLM Chatbot + RAG + Agent", layout="wide")

# ---------- Custom Dark UI ----------
st.markdown(
    """
    <style>
      /* Full-page dark gradient */
      .stApp {
        background: radial-gradient(1200px 600px at 10% 10%, #111827 0%, #0b1220 40%, #0a0f1a 100%) !important;
        color: #E5E7EB !important;
      }
      /* Section cards */
      .section-card {
        background: linear-gradient(180deg, rgba(31,41,55,0.85) 0%, rgba(17,24,39,0.85) 100%);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.02);
        backdrop-filter: blur(6px);
      }
      .section-title {
        font-weight: 700 !important;
        letter-spacing: 0.2px;
        margin-bottom: 6px !important;
        color: #A5B4FC;
        text-shadow: 0 0 18px rgba(99,102,241,0.25);
      }
      .section-sub {
        color: #93C5FD;
        margin-top: -4px;
        margin-bottom: 10px;
        font-size: 0.95rem;
      }
      /* Inputs */
      .stTextArea textarea, .stTextInput input, .stSelectbox div, .stSlider > div {
        background: rgba(17,24,39,0.75) !important;
        color: #E5E7EB !important;
        border: 1px solid rgba(99,102,241,0.35) !important;
        border-radius: 12px !important;
      }
      /* Buttons */
      .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        color: white;
        border: 0;
        padding: 0.55rem 1.05rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 8px 18px rgba(5,150,105,0.25);
      }
      .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(6,182,212,0.32);
      }
      /* Success/info/error boxes tint */
      .stAlert {
        background: rgba(17,24,39,0.65) !important;
        border: 1px solid rgba(59,130,246,0.35) !important;
        border-radius: 12px !important;
      }
      /* JSON pretty box */
      .element-container .stJson {
        border-radius: 12px !important;
        border: 1px solid rgba(99,102,241,0.25) !important;
      }
      /* Small badges */
      .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.45);
        color: #E5E7EB;
        font-size: 12px;
        margin-left: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='section-title' style='font-size:2.0rem'>LLM Chatbot & Attachment Analyzer (RAG + Agent)</h1>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Single-page control panel · Dark, colorful, and fast</div>", unsafe_allow_html=True)

# ---------- Helpers ----------
def section_header(title, subtitle=None):
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-sub'>{subtitle}</div>", unsafe_allow_html=True)

def card():
    return st.container()

# ---------- Layout: two rows of cards ----------
row1_col1, row1_col2 = st.columns([1.2, 1])
row2_col1, row2_col2 = st.columns([1, 1])

# =============================================
# 1) Chatbot (RAG optional)
# =============================================
with row1_col1:
    with card():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_header("Chatbot", "Talk to the model. Optionally ground answers with your uploaded documents.")
        user_input = st.text_area("Your message", height=120, key="chat_message")
        c1, c2, c3 = st.columns([1, 1, 1.5])
        with c1:
            use_rag = st.checkbox("Use RAG", value=True)
        with c2:
            top_k = st.slider("Top-k", min_value=1, max_value=8, value=4, help="Number of document chunks to retrieve")
        with c3:
            st.write("")  # spacing
            send = st.button("Send", key="chatbtn")

        if send:
            if not user_input.strip():
                st.warning("Please enter a message.")
            else:
                with st.spinner("Getting response..."):
                    payload = {"message": user_input, "use_rag": bool(use_rag), "top_k": int(top_k)}
                    try:
                        resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.markdown("**Response**")
                            st.write(data.get("answer", "No answer"))
                            if use_rag and "retrieved" in data:
                                st.markdown("**Retrieved Passages**")
                                for i, r in enumerate(data["retrieved"], start=1):
                                    st.markdown(f"Source {i} <span class='badge'>score {r['score']:.4f}</span>  <span class='badge'>id {r['id']}</span>", unsafe_allow_html=True)
                                    st.write(r["text"][:1000])
                        else:
                            st.error(resp.text)
                    except Exception as e:
                        st.error(f"Request failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# 2) Upload & Analyze
# =============================================
with row1_col2:
    with card():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_header("Upload & Analyze", "PDFs, images (OCR), and text files are indexed for RAG.")
        uploaded_file = st.file_uploader("Choose a file (PDF, image, or text)", type=None, key="uploader")
        if st.button("Analyze & Index", key="uploadbtn"):
            if uploaded_file is None:
                st.warning("Please choose a file first.")
            else:
                with st.spinner("Uploading and analyzing..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                        resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success("Upload complete")
                            st.json(data)
                        else:
                            st.error(resp.text)
                    except Exception as e:
                        st.error(f"Request failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# 3) Stored Chunks
# =============================================
with row2_col1:
    with card():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_header("Stored Chunks", "Quickly preview indexed segments.")
        c1, c2 = st.columns([1, 1.8])
        with c1:
            limit = st.slider("How many to list", min_value=1, max_value=200, value=10)
            if st.button("Refresh list"):
                try:
                    resp = requests.get(f"{BACKEND_URL}/docs/list?limit={limit}", timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.write(f"Total chunks stored: {data.get('count')}")
                        st.session_state["items"] = data.get("items", [])
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(f"Request failed: {e}")
        with c2:
            items = st.session_state.get("items", [])
            if items:
                for it in items:
                    ts_fmt = time.ctime(it.get("ts", 0)) if it.get("ts") else "n/a"
                    st.markdown(f"**ID:** {it['id']}  •  source: `{it.get('source','n/a')}`  •  {ts_fmt}")
                    st.write(it["text"][:800])
                    st.write("---")
            else:
                st.info("No items loaded. Click “Refresh list” to load stored chunks.")
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# 4) Agentic AI (RAG aware)
# =============================================
with row2_col2:
    with card():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        section_header("Agentic AI", "Let the agent plan, retrieve, and answer.")
        agent_q = st.text_area("Enter your query", height=100, key="agent_q")
        if st.button("Ask Agent"):
            if agent_q.strip():
                with st.spinner("Agent thinking..."):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/agent",
                            json={"message": agent_q, "use_rag": True, "top_k": 4},
                            timeout=120,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.markdown("**Agent Answer**")
                            st.write(data.get("answer", "No answer"))
                            if "retrieved" in data and data["retrieved"]:
                                st.markdown("**Retrieved Passages (Agent)**")
                                for i, r in enumerate(data["retrieved"], start=1):
                                    st.markdown(f"Source {i} <span class='badge'>score {r['score']:.4f}</span>  <span class='badge'>id {r['id']}</span>", unsafe_allow_html=True)
                                    st.write(r["text"][:1000])
                            if "trace" in data:
                                st.markdown("**Agent Trace**")
                                st.json(data["trace"])
                        else:
                            st.error(resp.text)
                    except Exception as e:
                        st.error(f"Request failed: {e}")
            else:
                st.warning("Please enter a message for the agent.")
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# 5) Agent Tools (function-calling)
# =============================================
with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    section_header("Agent Tools", "Function-calling framework. The agent detects and uses tools when helpful.")
    tools_q = st.text_area("Enter your query", height=100, key="agent_tools")
    if st.button("Ask with Tools"):
        if tools_q.strip():
            with st.spinner("Agent reasoning..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/agent",
                        json={"message": tools_q, "use_rag": False},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown("**Agent Answer**")
                        st.write(data.get("answer", "No answer"))
                        if "trace" in data:
                            st.markdown("**Execution Trace**")
                            st.json(data["trace"])
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(f"Request failed: {e}")
        else:
            st.warning("Please enter a message.")
    st.markdown("</div>", unsafe_allow_html=True)
