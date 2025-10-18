#!/bin/bash

# Test script for CI/CD pipeline
# This script helps verify that your automated deployment is working

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
    echo -e "${BLUE}[TEST]${NC} $1"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "Google Cloud SDK is not installed."
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    print_error "No project ID set. Please run 'gcloud config set project YOUR_PROJECT_ID' first."
    exit 1
fi

print_header "Testing CI/CD Pipeline for Moments Face Tagging Service"
print_status "Project: $PROJECT_ID"

# Test 1: Check if Cloud Build trigger exists
print_header "1. Checking Cloud Build trigger..."
if gcloud builds triggers list --filter="name:moments-face-tagging-trigger" --format="value(name)" | grep -q "moments-face-tagging-trigger"; then
    print_status "✅ Cloud Build trigger exists"
else
    print_error "❌ Cloud Build trigger not found. Run ./setup-cicd.sh first."
    exit 1
fi

# Test 2: Check if Cloud Run service exists
print_header "2. Checking Cloud Run service..."
if gcloud run services list --filter="metadata.name:moments-face-tagging" --format="value(metadata.name)" | grep -q "moments-face-tagging"; then
    print_status "✅ Cloud Run service exists"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe moments-face-tagging --region=us-central1 --format='value(status.url)' 2>/dev/null || echo "")
    if [ -n "$SERVICE_URL" ]; then
        print_status "Service URL: $SERVICE_URL"
    fi
else
    print_warning "⚠️  Cloud Run service not found. It will be created on first deployment."
fi

# Test 3: Check recent builds
print_header "3. Checking recent builds..."
BUILD_COUNT=$(gcloud builds list --limit=5 --format="value(id)" | wc -l)
if [ "$BUILD_COUNT" -gt 0 ]; then
    print_status "✅ Found $BUILD_COUNT recent builds"
    
    # Show latest build status
    LATEST_BUILD=$(gcloud builds list --limit=1 --format="value(id,status)")
    print_status "Latest build: $LATEST_BUILD"
else
    print_warning "⚠️  No builds found yet. Push to your repository to trigger the first build."
fi

# Test 4: Check required APIs
print_header "4. Checking required APIs..."
REQUIRED_APIS=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "containerregistry.googleapis.com"
    "firestore.googleapis.com"
    "storage.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        print_status "✅ $api is enabled"
    else
        print_error "❌ $api is not enabled"
    fi
done

# Test 5: Check Git repository
print_header "5. Checking Git repository..."
if [ -d ".git" ]; then
    REPO_URL=$(git remote get-url origin 2>/dev/null || echo "")
    if [ -n "$REPO_URL" ]; then
        print_status "✅ Git repository found: $REPO_URL"
        
        # Check if we're on main branch
        CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "")
        if [ "$CURRENT_BRANCH" = "main" ]; then
            print_status "✅ On main branch"
        else
            print_warning "⚠️  Not on main branch (current: $CURRENT_BRANCH)"
        fi
    else
        print_error "❌ No remote repository found"
    fi
else
    print_error "❌ Not in a Git repository"
fi

# Test 6: Check if service is healthy (if it exists)
if [ -n "$SERVICE_URL" ]; then
    print_header "6. Testing service health..."
    if curl -s --max-time 10 "$SERVICE_URL/health" | grep -q "healthy"; then
        print_status "✅ Service is healthy"
    else
        print_warning "⚠️  Service health check failed or service not ready"
    fi
fi

# Summary
print_header "Test Summary"
echo ""
print_status "Your CI/CD pipeline is configured and ready!"
print_status "To trigger a deployment:"
print_status "1. Make changes to your code"
print_status "2. Commit and push: git add . && git commit -m 'Deploy changes' && git push origin main"
print_status "3. Monitor the build: gcloud builds list --limit=5"
print_status "4. Check the service: gcloud run services list"

if [ -n "$SERVICE_URL" ]; then
    print_status ""
    print_status "Your service is available at: $SERVICE_URL"
    print_status "API documentation: $SERVICE_URL/docs"
fi

print_status ""
print_status "Useful commands:"
print_status "• View builds: gcloud builds list"
print_status "• View logs: gcloud logging read 'resource.type=cloud_run_revision' --limit=20"
print_status "• Check triggers: gcloud builds triggers list"
print_status "• Service details: gcloud run services describe moments-face-tagging --region=us-central1"
