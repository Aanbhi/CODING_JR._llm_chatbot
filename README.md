# LLM Chatbot + Attachment Analyzer

This repo contains a FastAPI backend and a Streamlit frontend that:  
- Provide a `/chat` endpoint (calls OpenAI).  
- Provide a `/upload` endpoint to accept PDFs, images, and text files, extract text (PyMuPDF/Tesseract), and analyze with OpenAI.  
- Streamlit UI that talks to the backend for chat and file uploads.

**Security:** Revoke any leaked OpenAI keys. Do NOT commit keys to source control. Use environment variables or Render secrets.

## Files
- `app/main.py` - FastAPI backend (chat + upload endpoints)
- `app/streamlit_app.py` - Streamlit GUI frontend
- `Dockerfile` - Container image that installs Tesseract/poppler and runs FastAPI + Streamlit
- `requirements.txt` - Python dependencies
- `.dockerignore` - Docker ignore file
- `README.md` - (this file)

## Local testing
1. Build the image:
```bash
docker build -t llm-chatbot .
```

2. Run (replace `sk-NEW` with your new key):
```bash
docker run -e OPENAI_API_KEY="sk-NEW" -e BACKEND_URL="http://localhost:10000" -p 10000:10000 -p 8501:8501 llm-chatbot
```

3. Open Streamlit UI: [http://localhost:8501](http://localhost:8501)

4. Test API: POST [http://localhost:10000/chat](http://localhost:10000/chat) or /upload

## Deploy to Render

1. Push this repo to GitHub.
2. Create a Render Web Service (Docker).
3. Add `OPENAI_API_KEY` as a secret environment variable in Render.
4. Deploy.
