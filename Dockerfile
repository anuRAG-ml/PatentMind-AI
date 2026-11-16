
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OCR and scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Download spaCy English language model
RUN python -m spacy download en_core_web_sm

# Copy application source code
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY src/ ./src/

# Create directories for mounted data and models
RUN mkdir -p /app/models /app/data

EXPOSE 8000 8501
