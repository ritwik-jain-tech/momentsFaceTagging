#!/bin/bash

# Deployment script for Moments Face Tagging Service on Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "Google Cloud SDK is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_error "You are not authenticated with Google Cloud. Please run 'gcloud auth login' first."
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    print_error "No project ID set. Please run 'gcloud config set project YOUR_PROJECT_ID' first."
    exit 1
fi

print_status "Using project: $PROJECT_ID"

# Set variables
SERVICE_NAME="moments-face-tagging"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Check if required APIs are enabled
print_status "Checking required APIs..."
REQUIRED_APIS=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "containerregistry.googleapis.com"
    "firestore.googleapis.com"
    "storage.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    if ! gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        print_status "Enabling $api..."
        gcloud services enable "$api"
    fi
done

# Create Firestore database if it doesn't exist
print_status "Checking Firestore database..."
if ! gcloud firestore databases list --format="value(name)" | grep -q "projects/$PROJECT_ID/databases"; then
    print_status "Creating Firestore database..."
    gcloud firestore databases create --region="$REGION"
fi

# Create storage bucket if it doesn't exist
BUCKET_NAME="moments-storage-$(echo $PROJECT_ID | tr '[:upper:]' '[:lower:]')"
print_status "Checking storage bucket: $BUCKET_NAME"
if ! gsutil ls gs://$BUCKET_NAME &> /dev/null; then
    print_status "Creating storage bucket: $BUCKET_NAME"
    gsutil mb gs://$BUCKET_NAME
fi

# Build and push the container image
print_status "Building and pushing container image..."
gcloud builds submit --tag "$IMAGE_NAME" .

# Deploy to Cloud Run
print_status "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_NAME" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 80 \
  --port 8080 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID,ENVIRONMENT=production,USE_GPU=false,FACE_RECOGNITION_ENABLED=true,BACKGROUND_PROCESSING_ENABLED=true,STORAGE_BUCKET=$BUCKET_NAME"

# Get service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format='value(status.url)')

print_status "Deployment completed successfully!"
print_status "Service URL: $SERVICE_URL"
print_status "Health check: $SERVICE_URL/health"
print_status "API docs: $SERVICE_URL/docs"

# Test the deployment
print_status "Testing deployment..."
if curl -s "$SERVICE_URL/health" | grep -q "healthy"; then
    print_status "✅ Health check passed!"
else
    print_warning "⚠️  Health check failed. Check the logs:"
    print_warning "gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --limit 50"
fi

print_status "Deployment script completed!"
print_status "You can now use your Moments Face Tagging Service at: $SERVICE_URL"
