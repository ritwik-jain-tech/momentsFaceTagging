# Makefile for Moments Face Tagging Service

.PHONY: help install test build deploy clean

# Default target
help:
	@echo "Moments Face Tagging Service - Available Commands:"
	@echo ""
	@echo "  install    - Install Python dependencies"
	@echo "  test       - Run test suite"
	@echo "  run        - Run the service locally"
	@echo "  build      - Build Docker image"
	@echo "  deploy     - Deploy to Google Cloud Run"
	@echo "  clean      - Clean up temporary files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python test_service.py

# Run the service locally
run:
	python main.py

# Build Docker image
build:
	docker build -t moments-face-tagging .

# Deploy to Google Cloud Run
deploy:
	gcloud builds submit --config cloudbuild.yaml

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

# Development setup
dev-setup: install
	@echo "Development environment setup complete!"
	@echo "Run 'make run' to start the service locally"

# Production deployment
prod-deploy: build deploy
	@echo "Production deployment complete!"

# Quick test
quick-test:
	@echo "Running quick health check..."
	@curl -f http://localhost:8080/health || echo "Service not running"


