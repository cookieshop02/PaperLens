FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source code
COPY src/ ./src/

# Expose port
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]