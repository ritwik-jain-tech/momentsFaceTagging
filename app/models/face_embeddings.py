"""
Data models for face embedding collections
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class UserFaceEmbedding(BaseModel):
    """User face embedding model - one per user"""
    user_id: str = Field(..., description="Unique user identifier")
    event_id: str = Field(..., description="Event ID for the embedding")
    embedding: List[float] = Field(..., description="512-dimensional face embedding")
    quality_score: float = Field(..., description="Face quality score (0.0-1.0)")
    detection_score: float = Field(..., description="Face detection confidence (0.0-1.0)")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    age: Optional[int] = Field(None, description="Estimated age")
    gender: Optional[str] = Field(None, description="Estimated gender")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    selfie_url: str = Field(..., description="URL of the selfie image used for embedding")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MomentFaceEmbedding(BaseModel):
    """Moment face embedding model - one per moment"""
    moment_id: str = Field(..., description="Unique moment identifier")
    event_id: str = Field(..., description="Event ID for the moment")
    face_embeddings: List[Dict[str, Any]] = Field(default_factory=list, description="List of face embeddings in the moment")
    face_count: int = Field(..., description="Total number of faces detected")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    moment_url: str = Field(..., description="URL of the moment image")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FaceMatch(BaseModel):
    """Face match result model"""
    user_id: str = Field(..., description="Matched user ID")
    similarity: float = Field(..., description="Similarity score (0.0-1.0)")
    face_index: int = Field(..., description="Index of the matched face in moment")
    quality_score: float = Field(..., description="Quality score of the matched face")
    confidence: float = Field(..., description="Overall match confidence")
    moment_id: Optional[str] = Field(None, description="Moment ID where the match was found")
    moment_media_url: Optional[str] = Field(None, description="Media URL of the moment where the match was found")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FaceEmbeddingRequest(BaseModel):
    """Request model for face embedding operations"""
    image_url: str = Field(..., description="URL of the image to process")
    user_id: Optional[str] = Field(None, description="User ID (for selfie processing)")
    moment_id: Optional[str] = Field(None, description="Moment ID (for moment processing)")


class FaceEmbeddingResponse(BaseModel):
    """Response model for face embedding operations"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    face_count: int = Field(..., description="Number of faces detected")
    embeddings: Optional[List[Dict[str, Any]]] = Field(None, description="Face embeddings data")
    matches: Optional[List[FaceMatch]] = Field(None, description="Face matches found")
    quality_scores: Optional[List[float]] = Field(None, description="Quality scores for detected faces")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
