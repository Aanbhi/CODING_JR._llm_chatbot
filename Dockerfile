FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     poppler-utils     tesseract-ocr     libtesseract-dev  && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=10000
EXPOSE 10000
EXPOSE 8501

CMD sh -c "gunicorn app.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:10000 --workers 1 --timeout 120 & streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
