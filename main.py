"""
Moments Face Tagging Service
Production-grade facial recognition service for Cloud Run deployment
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from typing import List, Optional
import uvicorn

from app.core.config import settings
from app.api.v1.router import api_router
from app.core.face_recognition_insightface import InsightFaceRecognitionService
from app.core.firestore_client import FirestoreClient
from app.core.storage_client import StorageClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global services
face_recognition_service: Optional[InsightFaceRecognitionService] = None
firestore_client: Optional[FirestoreClient] = None
storage_client: Optional[StorageClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    global face_recognition_service, firestore_client, storage_client
    
    # Startup
    logger.info("Starting Moments Face Tagging Service...")
    
    try:
        # Initialize services
        face_recognition_service = InsightFaceRecognitionService()
        firestore_client = FirestoreClient()
        
        # Initialize Firestore collections
        await firestore_client.initialize_collections()
        
        storage_client = StorageClient()
        
        # Set global services in face embeddings endpoints
        from app.api.v1.endpoints import face_embeddings
        face_embeddings.face_recognition_service = face_recognition_service
        face_embeddings.firestore_client = firestore_client
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Moments Face Tagging Service...")
    if face_recognition_service:
        await face_recognition_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Moments Face Tagging Service",
    description="Production-grade facial recognition service for moments tagging",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "moments-face-tagging",
        "version": "1.0.0"
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint - checks if all services are ready"""
    try:
        # Check if face recognition service is available
        face_recognition_ready = face_recognition_service is not None
        firestore_ready = firestore_client is not None
        storage_ready = storage_client is not None
        
        if face_recognition_ready and firestore_ready and storage_ready:
            return {
                "status": "ready",
                "services": {
                    "face_recognition": "ready",
                    "firestore": "ready", 
                    "storage": "ready"
                }
            }
        else:
            return {
                "status": "not_ready",
                "services": {
                    "face_recognition": "ready" if face_recognition_ready else "initializing",
                    "firestore": "ready" if firestore_ready else "initializing",
                    "storage": "ready" if storage_ready else "initializing"
                }
            }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Moments Face Tagging Service",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    )
