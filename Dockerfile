FROM python:3.11-slim

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source tree and model weights
COPY . .

# Ensure storage directories exist
RUN mkdir -p data/outputs data/uploads data/.ultralytics

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/app/data/.ultralytics \
    PORT=8000

# Expose default port
EXPOSE 8000

# Run with Uvicorn
CMD ["uvicorn", "src.web_server:app", "--host", "0.0.0.0", "--port", "8000"]
