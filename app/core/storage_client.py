"""
Google Cloud Storage client for Moments Face Tagging Service
Handles image storage and retrieval operations
"""

import logging
from typing import Optional, Dict, Any
from google.cloud import storage
from google.cloud.exceptions import NotFound
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import os
from urllib.parse import urlparse

from app.core.auth_config import get_storage_client

logger = logging.getLogger(__name__)


class StorageClient:
    """
    Google Cloud Storage client for image management
    """
    
    def __init__(self):
        self.client: Optional[storage.Client] = None
        self.bucket: Optional[storage.Bucket] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Cloud Storage client with proper authentication"""
        try:
            logger.info("Initializing Cloud Storage client with environment-based authentication...")
            self.client = get_storage_client()
            
            # Get bucket name from environment or use default
            bucket_name = os.getenv("STORAGE_BUCKET", "momentslive")
            self.bucket = self.client.bucket(bucket_name)
            
            logger.info(f"Cloud Storage client initialized with bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Storage client: {e}")
            raise
    
    async def upload_image(self, image_data: bytes, file_path: str, 
                          content_type: str = "image/jpeg") -> Optional[str]:
        """
        Upload image to Cloud Storage
        """
        try:
            if not self.bucket:
                logger.error("Storage bucket not initialized")
                return None
            
            if not image_data:
                logger.error(f"Cannot upload empty image data to {file_path}")
                return None
            
            logger.info(f"Uploading image to: {file_path} (size: {len(image_data)} bytes, content_type: {content_type})")
            
            # Create blob
            blob = self.bucket.blob(file_path)
            
            # Set metadata (custom metadata, not content_type)
            blob.metadata = {
                'uploaded_at': str(time.time()),
                'service': 'moments-face-tagging'
            }
            
            # Set cache-control headers to force CDN revalidation
            # "no-cache" means CDN must revalidate with origin before serving cached content
            blob.cache_control = "no-cache, must-revalidate"
            
            # Upload the data with content_type parameter
            blob.upload_from_string(image_data, content_type=content_type)
            
            # Generate CDN URL (matching Java service format)
            # Bucket has uniform bucket-level access enabled, so no need to call make_public()
            cdn_domain = os.getenv("CDN_DOMAIN", "images.moments.live")
            public_url = f"https://{cdn_domain}/{file_path}"
            
            logger.info(f"Successfully uploaded image: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload image to {file_path}: {e}", exc_info=True)
            import traceback
            logger.error(f"Upload traceback: {traceback.format_exc()}")
            return None
    
    async def download_image(self, image_url: str) -> Optional[bytes]:
        """
        Download image from Cloud Storage or external URL
        """
        try:
            logger.info(f"Downloading image from: {image_url}")
            
            # Check if it's a Cloud Storage URL
            if "storage.googleapis.com" in image_url:
                return await self._download_from_storage(image_url)
            else:
                # External URL - use requests
                import requests
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                return response.content
                
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            return None
    
    async def _download_from_storage(self, image_url: str) -> Optional[bytes]:
        """
        Download image from Cloud Storage
        """
        try:
            # Parse the URL to get blob path
            parsed_url = urlparse(image_url)
            blob_path = parsed_url.path.lstrip('/')
            
            # Remove bucket name from path if present
            if blob_path.startswith(self.bucket.name + '/'):
                blob_path = blob_path[len(self.bucket.name) + 1:]
            
            # Get blob
            blob = self.bucket.blob(blob_path)
            
            # Download content
            content = blob.download_as_bytes()
            
            logger.info(f"Successfully downloaded image from storage: {len(content)} bytes")
            return content
            
        except Exception as e:
            logger.error(f"Failed to download from storage: {e}")
            return None
    
    async def delete_image(self, image_url: str) -> bool:
        """
        Delete image from Cloud Storage
        """
        try:
            logger.info(f"Deleting image: {image_url}")
            
            # Parse the URL to get blob path
            parsed_url = urlparse(image_url)
            blob_path = parsed_url.path.lstrip('/')
            
            # Remove bucket name from path if present
            if blob_path.startswith(self.bucket.name + '/'):
                blob_path = blob_path[len(self.bucket.name) + 1:]
            
            # Get blob and delete
            blob = self.bucket.blob(blob_path)
            blob.delete()
            
            logger.info(f"Successfully deleted image: {image_url}")
            return True
            
        except NotFound:
            logger.warning(f"Image not found for deletion: {image_url}")
            return True  # Consider it successful if not found
        except Exception as e:
            logger.error(f"Failed to delete image: {e}")
            return False
    
    async def get_image_metadata(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Get image metadata from Cloud Storage
        """
        try:
            logger.info(f"Getting metadata for: {image_url}")
            
            # Parse the URL to get blob path
            parsed_url = urlparse(image_url)
            blob_path = parsed_url.path.lstrip('/')
            
            # Remove bucket name from path if present
            if blob_path.startswith(self.bucket.name + '/'):
                blob_path = blob_path[len(self.bucket.name) + 1:]
            
            # Get blob
            blob = self.bucket.blob(blob_path)
            
            # Reload to get metadata
            blob.reload()
            
            metadata = {
                'size': blob.size,
                'content_type': blob.content_type,
                'created': blob.time_created,
                'updated': blob.updated,
                'metadata': blob.metadata
            }
            
            logger.info(f"Retrieved metadata for image: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get image metadata: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            logger.info("Storage client cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

