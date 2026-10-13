# ==============================================================================
# EduPredict – Student Academic Performance Prediction System
# Dockerfile for Production / Containerized Deployment
# ==============================================================================

# Base image: Official lightweight Python 3.11 Debian slim
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000

# Set working directory
WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and assets
COPY app.py /app/app.py
COPY templates/ /app/templates/
COPY static/ /app/static/
COPY model/ /app/model/
COPY models/ /app/models/

# Expose web server port
EXPOSE 5000

# Health check monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the Flask backend
CMD ["python", "app.py"]
