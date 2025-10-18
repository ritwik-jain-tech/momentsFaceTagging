"""
Configuration settings for the Moments Face Tagging Service
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Google Cloud Configuration
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "moments-38b77")
    service_account_path: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Authentication Configuration (following MomentsBackend pattern)
    use_service_account: bool = os.getenv("USE_SERVICE_ACCOUNT", "true").lower() == "true"
    service_account_file: str = os.getenv("SERVICE_ACCOUNT_FILE", "serviceAccountKey.json")
    
    # Firestore Configuration
    firestore_collection_prefix: str = os.getenv("FIRESTORE_COLLECTION_PREFIX", "moments")
    
    # Cloud Storage Configuration
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "moments-storage")
    
    # Face Recognition Configuration
    face_recognition_enabled: bool = os.getenv("FACE_RECOGNITION_ENABLED", "true").lower() == "true"
    face_match_threshold: float = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))
    min_face_size_ratio: float = float(os.getenv("MIN_FACE_SIZE_RATIO", "0.05"))
    max_face_size_ratio: float = float(os.getenv("MAX_FACE_SIZE_RATIO", "0.8"))
    
    # Background Processing
    background_processing_enabled: bool = os.getenv("BACKGROUND_PROCESSING_ENABLED", "true").lower() == "true"
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # API Configuration
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes
    
    # Model Configuration
    model_path: str = os.getenv("MODEL_PATH", "/app/models")
    use_gpu: bool = os.getenv("USE_GPU", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
