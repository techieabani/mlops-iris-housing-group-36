# Use slim Python base for smaller image size
FROM python:3.10-slim AS base

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (if your project needs extra libs like gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a separate layer for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code last (to leverage Docker layer caching)
COPY . .

# Expose API port
EXPOSE 8000

# Start Uvicorn server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]