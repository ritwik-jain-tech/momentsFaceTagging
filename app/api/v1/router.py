"""
API v1 router for Moments Face Tagging Service
"""

from fastapi import APIRouter
from app.api.v1.endpoints import face_embeddings

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    face_embeddings.router,
    prefix="/face-embeddings",
    tags=["face-embeddings"]
)

