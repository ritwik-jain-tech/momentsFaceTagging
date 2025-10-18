#!/bin/bash

# Local development setup script for Moments Face Tagging Service
# This script helps set up the service account key without committing secrets

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

print_header "Setting up Moments Face Tagging Service for Local Development"

# Check if service account key exists
if [ -f "serviceAccountKey.json" ]; then
    print_status "✅ Service account key already exists"
else
    print_warning "⚠️  Service account key not found"
    echo ""
    print_status "To set up local development, you need to copy the service account key:"
    echo ""
    print_status "1. Copy from your MomentsBackend project:"
    print_status "   cp /path/to/momentsBackend/src/main/resources/serviceAccountKey.json ./serviceAccountKey.json"
    echo ""
    print_status "2. Or download from Google Cloud Console:"
    print_status "   - Go to IAM & Admin > Service Accounts"
    print_status "   - Find your service account"
    print_status "   - Create and download a new key"
    echo ""
    print_warning "⚠️  IMPORTANT: This file is in .gitignore and will NOT be committed to Git"
    echo ""
    read -p "Press Enter when you have copied the service account key file..."
    
    if [ ! -f "serviceAccountKey.json" ]; then
        print_error "Service account key file not found. Please copy it and run this script again."
        exit 1
    fi
fi

# Check if Python dependencies are installed
print_header "Checking Python dependencies..."
if [ -f "requirements.txt" ]; then
    if pip list | grep -q fastapi; then
        print_status "✅ Python dependencies already installed"
    else
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
    fi
else
    print_error "requirements.txt not found"
    exit 1
fi

# Set environment variables
print_header "Setting up environment variables..."
export ENVIRONMENT=development
export GOOGLE_CLOUD_PROJECT=moments-38b77
export GOOGLE_APPLICATION_CREDENTIALS=serviceAccountKey.json
export USE_SERVICE_ACCOUNT=true

print_status "Environment variables set:"
print_status "  ENVIRONMENT=development"
print_status "  GOOGLE_CLOUD_PROJECT=moments-38b77"
print_status "  GOOGLE_APPLICATION_CREDENTIALS=serviceAccountKey.json"
print_status "  USE_SERVICE_ACCOUNT=true"

# Test authentication
print_header "Testing authentication..."
if python test-auth.py; then
    print_status "✅ Authentication test passed"
else
    print_error "❌ Authentication test failed"
    exit 1
fi

print_header "Local development setup complete! 🎉"
echo ""
print_status "You can now run the service locally:"
print_status "  python main.py"
echo ""
print_status "The service will be available at:"
print_status "  http://localhost:8080"
print_status "  http://localhost:8080/docs (API documentation)"
echo ""
print_warning "Remember: serviceAccountKey.json is in .gitignore and will NOT be committed to Git"
