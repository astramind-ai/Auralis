# Deployment Guide

This guide covers deploying Auralis in production environments.

## Docker Deployment

### Basic Container

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Auralis
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Run server
CMD ["python", "server.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  tts:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MAX_BATCH_SIZE=4
```

## Server Configuration

### FastAPI Server

```python
from fastapi import FastAPI
from auralis import TTS

app = FastAPI()
tts = TTS(use_vllm=True)

@app.post("/generate")
async def generate_speech(text: str):
    audio = await tts.generate(text)
    return {"audio": audio.to_bytes()}
```

### Production Settings

```python
# server.py
import uvicorn
from auralis.common.logging import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
```

## Load Balancing

### Nginx Configuration

```nginx
upstream tts_servers {
    server tts1:8000;
    server tts2:8000;
    server tts3:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://tts_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

requests_total = Counter(
    'tts_requests_total', 
    'Total TTS requests'
)

generation_time = Histogram(
    'tts_generation_seconds', 
    'Time spent generating audio'
)
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_memory": get_gpu_memory_usage(),
        "model_loaded": tts.is_ready()
    }
```

## Error Handling

```python
from fastapi import HTTPException
from auralis.common.logging import setup_logger

logger = setup_logger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error processing request: {exc}")
    return {"error": str(exc)}
```

!!! tip "Production Checklist"
    - [ ] Configure logging
    - [ ] Setup monitoring
    - [ ] Implement health checks
    - [ ] Configure error handling
    - [ ] Set resource limits
    - [ ] Enable SSL/TLS
    - [ ] Setup backups

!!! warning "Common Issues"
    - Memory leaks from large batches
    - GPU OOM in production
    - Network timeouts
    - Insufficient error handling 