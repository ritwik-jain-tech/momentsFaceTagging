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
from datetime import datetime

from app.core.config import settings
from app.api.v1.router import api_router
from app.core.face_recognition_insightface import InsightFaceRecognitionService
from app.core.firestore_client import FirestoreClient
from app.core.storage_client import StorageClient
from app.core.image_compression import ImageCompressionService

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
image_compression_service: Optional[ImageCompressionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    global face_recognition_service, firestore_client, storage_client, image_compression_service
    
    # Startup
    logger.info("Starting Moments Face Tagging Service...")
    
    try:
        # Initialize services with error handling
        logger.info("Initializing face recognition service...")
        try:
            face_recognition_service = InsightFaceRecognitionService()
            logger.info("Face recognition service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face recognition service: {e}")
            face_recognition_service = None
        
        logger.info("Initializing Firestore client...")
        try:
            firestore_client = FirestoreClient()
            await firestore_client.initialize_collections()
            logger.info("Firestore client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            firestore_client = None
        
        logger.info("Initializing storage client...")
        try:
            storage_client = StorageClient()
            logger.info("Storage client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize storage client: {e}")
            storage_client = None
        
        logger.info("Initializing image compression service...")
        try:
            image_compression_service = ImageCompressionService()
            logger.info("Image compression service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize image compression service: {e}")
            image_compression_service = None
        
        # Set global services in face embeddings endpoints
        from app.api.v1.endpoints import face_embeddings
        if face_recognition_service:
            face_embeddings.face_recognition_service = face_recognition_service
        if firestore_client:
            face_embeddings.firestore_client = firestore_client
        if storage_client:
            face_embeddings.storage_client = storage_client
        if image_compression_service:
            face_embeddings.image_compression_service = image_compression_service
        logger.info("Global services set in endpoints")
        
        logger.info("Service initialization completed (some services may have failed)")
        
    except Exception as e:
        logger.error(f"Critical error during service initialization: {e}")
        # Don't raise - allow service to start even if some services fail
        logger.warning("Service will start with limited functionality")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Moments Face Tagging Service...")
    if face_recognition_service:
        try:
            await face_recognition_service.cleanup()
        except Exception as e:
            logger.error(f"Error during face recognition service cleanup: {e}")


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
    """Health check endpoint - always returns healthy if service is running"""
    return {
        "status": "healthy",
        "service": "moments-face-tagging",
        "version": "1.0.0"
    }


@app.get("/startup")
async def startup_check():
    """Startup check endpoint - returns immediately"""
    return {
        "status": "started",
        "message": "Service is starting up",
        "timestamp": datetime.now().isoformat()
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
    port = int(os.getenv("PORT", 8080))
    environment = os.getenv("ENVIRONMENT", "development")
    
    logger.info(f"Starting FastAPI server on port {port}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Host: 0.0.0.0")
    logger.info(f"Reload mode: {environment == 'development'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=environment == "development"
    )
