"""
Request models for Moments Face Tagging Service
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional


class ProcessSelfieRequest(BaseModel):
    """Request model for processing selfie images"""
    image_url: HttpUrl = Field(..., description="URL of the selfie image")
    event_id: str = Field(..., description="Event ID for the selfie")
    user_id: str = Field(..., description="User ID who uploaded the selfie")
    face_matching: bool = Field(False, description="Whether to perform face matching against existing moments")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://storage.googleapis.com/bucket/selfie.jpg",
                "event_id": "event_123",
                "user_id": "user_456",
                "face_matching": True
            }
        }


class ProcessMomentRequest(BaseModel):
    """Request model for processing moment images"""
    image_url: HttpUrl = Field(..., description="URL of the moment image")
    moment_id: str = Field(..., description="Moment ID for the image")
    user_id: str = Field(..., description="User ID who uploaded the moment")
    event_id: str = Field(..., description="Event ID for the moment")
    match_faces: bool = Field(False, description="Whether to match faces against existing user embeddings")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://storage.googleapis.com/bucket/moment.jpg",
                "moment_id": "moment_789",
                "user_id": "user_456",
                "event_id": "event_123",
                "match_faces": True
            }
        }


class ProcessEventMomentsRequest(BaseModel):
    """Request model for processing all moments in an event"""
    event_id: str = Field(..., description="Event ID to process all moments for")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "event_123"
            }
        }


class RemoveUserTaggingRequest(BaseModel):
    """Request model for removing user tagging from all moments in an event"""
    user_id: str = Field(..., description="User ID to remove tagging for")
    event_id: str = Field(..., description="Event ID to remove tagging from")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_456",
                "event_id": "event_123"
            }
        }
