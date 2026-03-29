FROM python:3.11-slim

WORKDIR /app

# Flush Python stdout/stderr immediately so logs appear in Azure without buffering
ENV PYTHONUNBUFFERED=1

# libgomp1 is required by PyTorch (used by sentence-transformers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer — only re-runs if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image so startup is instant.
# (~90 MB model, adds to image size but removes the download on every cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy application code
COPY app.py preprocess.py ./
COPY static/ ./static/

# Copy the pre-processed data files (generated locally by preprocess.py).
# info.pdf is NOT copied — it's only needed for preprocessing, not at runtime.
COPY pages_data.json embeddings.npy ./

# Copy brain viewer assets
COPY index.html brain.glb ./
COPY models/ ./models/

EXPOSE 8000

CMD ["gunicorn", "--bind=0.0.0.0:8000", "--timeout=600", "--workers=1", "--threads=4", "app:app"]
