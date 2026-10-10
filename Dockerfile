FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# 1. libraries first: this layer is reused whenever only the code or the model changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. the embedding model, downloaded while the image is built so containers need no internet to start
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models/fastembed_cache')"

# 3. the vector store (from git), the model (put in latest_model/ by the pipeline) and the code
COPY vectorstore/ ./vectorstore/
COPY latest_model/ ./models/
COPY Agent/ ./Agent/
COPY weather_backend/ ./weather_backend/

EXPOSE 8000
CMD ["uvicorn", "weather_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

