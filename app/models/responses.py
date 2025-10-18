"""
Response models for Moments Face Tagging Service
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")


class FaceEmbeddingResult(BaseModel):
    """Result of face embedding extraction"""
    success: bool = Field(..., description="Whether embedding extraction was successful")
    message: str = Field(..., description="Result message")
    embedding: Optional[List[float]] = Field(None, description="Face embedding vector")
    face_count: int = Field(0, description="Number of faces detected")
    detection_score: Optional[float] = Field(None, description="Face detection confidence score")


class ProcessSelfieResponse(BaseModel):
    """Response for selfie processing"""
    user_id: str = Field(..., description="User ID")
    event_id: str = Field(..., description="Event ID")
    face_id: str = Field(..., description="Unique face ID")
    embedding_result: FaceEmbeddingResult = Field(..., description="Embedding extraction result")
    background_processing_started: bool = Field(..., description="Whether background processing started")


class FaceMatch(BaseModel):
    """Face match result"""
    face_id: str = Field(..., description="Face ID that matched")
    similarity: float = Field(..., description="Similarity score")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates")
    detection_score: Optional[float] = Field(None, description="Face detection score")


class MomentProcessingResult(BaseModel):
    """Result of moment image processing"""
    success: bool = Field(..., description="Whether processing was successful")
    message: str = Field(..., description="Result message")
    faces: List[Dict[str, Any]] = Field(..., description="Detected faces with embeddings")
    face_count: int = Field(0, description="Number of faces detected")


class ProcessMomentResponse(BaseModel):
    """Response for moment processing"""
    moment_id: str = Field(..., description="Moment ID")
    event_id: str = Field(..., description="Event ID")
    user_id: str = Field(..., description="User ID")
    processing_result: MomentProcessingResult = Field(..., description="Processing result")
    background_processing_started: bool = Field(..., description="Whether background processing started")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: float = Field(..., description="Response timestamp")

