#!/bin/bash

# Startup script for Moments Face Tagging Service

echo "Starting Moments Face Tagging Service..."

# Set default environment variables if not set
export ENVIRONMENT=${ENVIRONMENT:-production}
export PORT=${PORT:-8080}
export GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-moments-38b77}
export FACE_RECOGNITION_ENABLED=${FACE_RECOGNITION_ENABLED:-true}
export BACKGROUND_PROCESSING_ENABLED=${BACKGROUND_PROCESSING_ENABLED:-true}

# Memory optimization settings
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_MMAP_MAX_=65536

# Create models directory
echo "Creating models directory..."
mkdir -p /app/models

# Start the application immediately - let it handle model download internally
echo "Starting FastAPI application..."
echo "Port: $PORT"
echo "Environment: $ENVIRONMENT"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Files in current directory: $(ls -la)"

# Start the application
exec python main.py

