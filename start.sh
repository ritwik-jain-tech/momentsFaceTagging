#!/bin/bash

# Startup script for Moments Face Tagging Service

echo "Starting Moments Face Tagging Service..."

# Set default environment variables if not set
export ENVIRONMENT=${ENVIRONMENT:-production}
export PORT=${PORT:-8080}
export GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-moments-38b77}
export FACE_RECOGNITION_ENABLED=${FACE_RECOGNITION_ENABLED:-true}
export BACKGROUND_PROCESSING_ENABLED=${BACKGROUND_PROCESSING_ENABLED:-true}

# Download InsightFace models if not present
echo "Checking InsightFace models..."
python -c "
import insightface
import os
model_path = os.getenv('MODEL_PATH', '/app/models')
if not os.path.exists(model_path):
    os.makedirs(model_path)
print('InsightFace models ready')
"

# Start the application
echo "Starting FastAPI application..."
exec python main.py

