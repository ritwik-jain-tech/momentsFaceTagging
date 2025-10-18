#!/bin/bash

# CI/CD Setup Script for Moments Face Tagging Service
# This script sets up automated deployment on Git push

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
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

print_header "Setting up CI/CD for Moments Face Tagging Service"
print_status "Using project: $PROJECT_ID"

# Get repository information
REPO_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [ -z "$REPO_URL" ]; then
    print_error "No Git repository found. Please make sure you're in a Git repository."
    exit 1
fi

print_status "Repository: $REPO_URL"

# Extract repository name and owner
if [[ $REPO_URL == *"github.com"* ]]; then
    REPO_NAME=$(echo $REPO_URL | sed 's/.*github.com[:/]\([^/]*\/[^/]*\)\.git/\1/')
    REPO_OWNER=$(echo $REPO_NAME | cut -d'/' -f1)
    REPO_NAME_ONLY=$(echo $REPO_NAME | cut -d'/' -f2)
    REPO_TYPE="github"
elif [[ $REPO_URL == *"gitlab.com"* ]]; then
    REPO_NAME=$(echo $REPO_URL | sed 's/.*gitlab.com[:/]\([^/]*\/[^/]*\)\.git/\1/')
    REPO_OWNER=$(echo $REPO_NAME | cut -d'/' -f1)
    REPO_NAME_ONLY=$(echo $REPO_NAME | cut -d'/' -f2)
    REPO_TYPE="gitlab"
else
    print_error "Unsupported repository type. Only GitHub and GitLab are supported."
    exit 1
fi

print_status "Repository: $REPO_NAME ($REPO_TYPE)"

# Enable required APIs
print_header "Enabling required APIs..."
REQUIRED_APIS=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "containerregistry.googleapis.com"
    "firestore.googleapis.com"
    "storage.googleapis.com"
    "sourcerepo.googleapis.com"
    "cloudresourcemanager.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    if ! gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        print_status "Enabling $api..."
        gcloud services enable "$api"
    else
        print_status "$api is already enabled"
    fi
done

# Create Firestore database if it doesn't exist
print_header "Setting up Firestore database..."
if ! gcloud firestore databases list --format="value(name)" | grep -q "projects/$PROJECT_ID/databases"; then
    print_status "Creating Firestore database..."
    gcloud firestore databases create --region="us-central1"
else
    print_status "Firestore database already exists"
fi

# Create storage bucket if it doesn't exist
BUCKET_NAME="moments-storage-$(echo $PROJECT_ID | tr '[:upper:]' '[:lower:]')"
print_header "Setting up Cloud Storage..."
if ! gsutil ls gs://$BUCKET_NAME &> /dev/null; then
    print_status "Creating storage bucket: $BUCKET_NAME"
    gsutil mb gs://$BUCKET_NAME
else
    print_status "Storage bucket already exists: $BUCKET_NAME"
fi

# Set up Cloud Build service account permissions
print_header "Configuring Cloud Build permissions..."
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Grant Cloud Build service account necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/run.admin" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/storage.admin" \
    --quiet

# Create Cloud Build trigger
print_header "Creating Cloud Build trigger..."

# Create trigger configuration
TRIGGER_CONFIG=$(cat <<EOF
{
  "name": "moments-face-tagging-trigger",
  "description": "Automated deployment for Moments Face Tagging Service",
  "github": {
    "owner": "$REPO_OWNER",
    "name": "$REPO_NAME_ONLY",
    "push": {
      "branch": "main"
    }
  },
  "filename": "cloudbuild-trigger.yaml",
  "substitutions": {
    "_SERVICE_NAME": "moments-face-tagging",
    "_REGION": "us-central1",
    "_MEMORY": "4Gi",
    "_CPU": "2",
    "_MAX_INSTANCES": "10",
    "_MIN_INSTANCES": "0",
    "_CONCURRENCY": "80",
    "_TIMEOUT": "900"
  },
  "build": {
    "steps": [
      {
        "name": "gcr.io/cloud-builders/docker",
        "args": [
          "build",
          "-t",
          "gcr.io/$PROJECT_ID/moments-face-tagging:\$COMMIT_SHA",
          "-t",
          "gcr.io/$PROJECT_ID/moments-face-tagging:latest",
          "."
        ]
      },
      {
        "name": "gcr.io/cloud-builders/docker",
        "args": [
          "push",
          "gcr.io/$PROJECT_ID/moments-face-tagging:\$COMMIT_SHA"
        ]
      },
      {
        "name": "gcr.io/cloud-builders/docker",
        "args": [
          "push",
          "gcr.io/$PROJECT_ID/moments-face-tagging:latest"
        ]
      },
      {
        "name": "gcr.io/google.com/cloudsdktool/cloud-sdk",
        "entrypoint": "gcloud",
        "args": [
          "run",
          "deploy",
          "moments-face-tagging",
          "--image",
          "gcr.io/$PROJECT_ID/moments-face-tagging:\$COMMIT_SHA",
          "--region",
          "us-central1",
          "--platform",
          "managed",
          "--allow-unauthenticated",
          "--memory",
          "4Gi",
          "--cpu",
          "2",
          "--timeout",
          "900",
          "--max-instances",
          "10",
          "--min-instances",
          "0",
          "--concurrency",
          "80",
          "--port",
          "8080",
          "--set-env-vars",
          "GOOGLE_CLOUD_PROJECT=$PROJECT_ID,ENVIRONMENT=production,USE_GPU=false,FACE_RECOGNITION_ENABLED=true,BACKGROUND_PROCESSING_ENABLED=true,STORAGE_BUCKET=$BUCKET_NAME,USE_SERVICE_ACCOUNT=false"
        ]
      }
    ],
    "images": [
      "gcr.io/$PROJECT_ID/moments-face-tagging:\$COMMIT_SHA",
      "gcr.io/$PROJECT_ID/moments-face-tagging:latest"
    ],
    "options": {
      "machineType": "E2_HIGHCPU_8",
      "diskSizeGb": 100,
      "logging": "CLOUD_LOGGING_ONLY"
    }
  }
}
EOF
)

# Create the trigger
print_status "Creating Cloud Build trigger..."
gcloud builds triggers create github \
    --repo-name="$REPO_NAME_ONLY" \
    --repo-owner="$REPO_OWNER" \
    --branch-pattern="main" \
    --build-config="cloudbuild-trigger.yaml" \
    --name="moments-face-tagging-trigger" \
    --description="Automated deployment for Moments Face Tagging Service" \
    --substitutions="_SERVICE_NAME=moments-face-tagging,_REGION=us-central1,_MEMORY=4Gi,_CPU=2,_MAX_INSTANCES=10,_MIN_INSTANCES=0,_CONCURRENCY=80,_TIMEOUT=900"

print_status "✅ Cloud Build trigger created successfully!"

# Test the trigger
print_header "Testing the setup..."
print_status "You can test the trigger by pushing to the main branch:"
print_status "git add . && git commit -m 'Test automated deployment' && git push origin main"

# Display useful information
print_header "Setup Complete! 🎉"
echo ""
print_status "Your CI/CD pipeline is now configured:"
print_status "• Repository: $REPO_URL"
print_status "• Trigger: moments-face-tagging-trigger"
print_status "• Service: moments-face-tagging"
print_status "• Region: us-central1"
print_status "• Storage Bucket: $BUCKET_NAME"
echo ""
print_status "Next steps:"
print_status "1. Push your code to trigger the first deployment:"
print_status "   git add . && git commit -m 'Initial deployment' && git push origin main"
print_status "2. Monitor the build:"
print_status "   gcloud builds list --limit=5"
print_status "3. Check your service:"
print_status "   gcloud run services list"
echo ""
print_status "Your service will be available at:"
print_status "https://moments-face-tagging-[hash]-uc.a.run.app"
print_status "(The exact URL will be shown after the first deployment)"
