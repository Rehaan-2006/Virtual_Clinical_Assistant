FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by unstructured's hi_res pipeline
# (poppler for pdf2image, tesseract for OCR fallback)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

COPY . .

# Don't copy your local data or processed_files log into the image
# These should stay on the host or in a volume

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]