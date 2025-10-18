"""
Authentication configuration for Moments Face Tagging Service
Handles Google Cloud authentication for both local development and production
Following the same pattern as MomentsBackend project
"""

import os
import logging
from typing import Optional
from google.auth import default
from google.oauth2 import service_account
from google.cloud import firestore, storage

logger = logging.getLogger(__name__)


class GoogleCloudAuth:
    """
    Google Cloud authentication handler
    Uses service account for local development and default credentials for production
    """
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "moments-38b77")
        self.service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self._credentials = None
        self._initialize_credentials()
    
    def _initialize_credentials(self):
        """Initialize Google Cloud credentials based on environment"""
        try:
            if self.environment.lower() == "production":
                # Production: Use default credentials (Cloud Run service account)
                logger.info("Using default credentials for production environment")
                self._credentials, _ = default()
                logger.info("Default credentials initialized successfully")
            else:
                # Development: Use service account key file
                logger.info("Using service account credentials for development environment")
                service_account_path = self._get_service_account_path()
                self._credentials = service_account.Credentials.from_service_account_file(
                    service_account_path
                )
                logger.info(f"Service account credentials loaded from: {service_account_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize credentials: {e}")
            raise
    
    def _get_service_account_path(self) -> str:
        """Get service account key file path"""
        # Check multiple possible locations
        possible_paths = [
            self.service_account_path,
            "serviceAccountKey.json",
            "app/serviceAccountKey.json",
            "src/main/resources/serviceAccountKey.json",
            os.path.join(os.getcwd(), "serviceAccountKey.json"),
            os.path.join(os.path.dirname(__file__), "..", "..", "serviceAccountKey.json")
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                logger.info(f"Found service account key at: {path}")
                return path
        
        raise FileNotFoundError(
            "Service account key file not found. Please ensure serviceAccountKey.json is available. "
            "Checked paths: " + ", ".join([p for p in possible_paths if p])
        )
    
    def get_credentials(self):
        """Get Google Cloud credentials"""
        return self._credentials
    
    def get_project_id(self) -> str:
        """Get Google Cloud project ID"""
        return self.project_id


class FirestoreAuth:
    """
    Firestore authentication handler
    """
    
    def __init__(self, auth: GoogleCloudAuth):
        self.auth = auth
        self._client: Optional[firestore.Client] = None
    
    def get_client(self) -> firestore.Client:
        """Get authenticated Firestore client"""
        if self._client is None:
            try:
                logger.info(f"Initializing Firestore client for project: {self.auth.get_project_id()}")
                self._client = firestore.Client(
                    project=self.auth.get_project_id(),
                    credentials=self.auth.get_credentials()
                )
                logger.info("Firestore client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Firestore client: {e}")
                raise
        return self._client


class StorageAuth:
    """
    Cloud Storage authentication handler
    """
    
    def __init__(self, auth: GoogleCloudAuth):
        self.auth = auth
        self._client: Optional[storage.Client] = None
    
    def get_client(self) -> storage.Client:
        """Get authenticated Cloud Storage client"""
        if self._client is None:
            try:
                logger.info(f"Initializing Cloud Storage client for project: {self.auth.get_project_id()}")
                self._client = storage.Client(
                    project=self.auth.get_project_id(),
                    credentials=self.auth.get_credentials()
                )
                logger.info("Cloud Storage client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Cloud Storage client: {e}")
                raise
        return self._client


# Global authentication instance
_google_auth: Optional[GoogleCloudAuth] = None
_firestore_auth: Optional[FirestoreAuth] = None
_storage_auth: Optional[StorageAuth] = None


def get_google_auth() -> GoogleCloudAuth:
    """Get global Google Cloud authentication instance"""
    global _google_auth
    if _google_auth is None:
        _google_auth = GoogleCloudAuth()
    return _google_auth


def get_firestore_auth() -> FirestoreAuth:
    """Get global Firestore authentication instance"""
    global _firestore_auth
    if _firestore_auth is None:
        _firestore_auth = FirestoreAuth(get_google_auth())
    return _firestore_auth


def get_storage_auth() -> StorageAuth:
    """Get global Storage authentication instance"""
    global _storage_auth
    if _storage_auth is None:
        _storage_auth = StorageAuth(get_google_auth())
    return _storage_auth


def get_firestore_client() -> firestore.Client:
    """Get authenticated Firestore client"""
    return get_firestore_auth().get_client()


def get_storage_client() -> storage.Client:
    """Get authenticated Cloud Storage client"""
    return get_storage_auth().get_client()
