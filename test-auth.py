#!/usr/bin/env python3
"""
Authentication test script for Moments Face Tagging Service
Tests both local development and production authentication patterns
"""

import os
import sys
import logging
from typing import Optional

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_authentication():
    """Test Google Cloud authentication setup"""
    print("🔐 Testing Google Cloud Authentication Setup")
    print("=" * 50)
    
    try:
        # Test 1: Import authentication modules
        print("\n1. Testing module imports...")
        from app.core.auth_config import get_google_auth, get_firestore_auth, get_storage_auth
        print("✅ Authentication modules imported successfully")
        
        # Test 2: Check environment configuration
        print("\n2. Checking environment configuration...")
        environment = os.getenv("ENVIRONMENT", "development")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "moments-38b77")
        use_service_account = os.getenv("USE_SERVICE_ACCOUNT", "true").lower() == "true"
        
        print(f"   Environment: {environment}")
        print(f"   Project ID: {project_id}")
        print(f"   Use Service Account: {use_service_account}")
        
        if environment == "development" and use_service_account:
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
            if os.path.exists(service_account_path):
                print(f"✅ Service account key found: {service_account_path}")
            else:
                print(f"❌ Service account key not found: {service_account_path}")
                return False
        else:
            print("✅ Using default credentials for production")
        
        # Test 3: Initialize authentication
        print("\n3. Testing authentication initialization...")
        auth = get_google_auth()
        credentials = auth.get_credentials()
        print("✅ Authentication credentials initialized")
        
        # Test 4: Test Firestore connection
        print("\n4. Testing Firestore connection...")
        try:
            firestore_auth = get_firestore_auth()
            firestore_client = firestore_auth.get_client()
            
            # Try to list collections (this will test the connection)
            collections = list(firestore_client.collections())
            print(f"✅ Firestore connection successful (found {len(collections)} collections)")
        except Exception as e:
            print(f"❌ Firestore connection failed: {e}")
            return False
        
        # Test 5: Test Storage connection
        print("\n5. Testing Cloud Storage connection...")
        try:
            storage_auth = get_storage_auth()
            storage_client = storage_auth.get_client()
            
            # Try to list buckets (this will test the connection)
            buckets = list(storage_client.list_buckets())
            print(f"✅ Storage connection successful (found {len(buckets)} buckets)")
        except Exception as e:
            print(f"❌ Storage connection failed: {e}")
            return False
        
        # Test 6: Test service clients
        print("\n6. Testing service client initialization...")
        try:
            from app.core.firestore_client import FirestoreClient
            from app.core.storage_client import StorageClient
            
            # Initialize clients (this will test the full authentication flow)
            firestore_client = FirestoreClient()
            storage_client = StorageClient()
            
            print("✅ Service clients initialized successfully")
        except Exception as e:
            print(f"❌ Service client initialization failed: {e}")
            return False
        
        print("\n🎉 All authentication tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Authentication test failed: {e}")
        return False

def test_environment_setup():
    """Test environment-specific configuration"""
    print("\n🌍 Testing Environment Setup")
    print("=" * 30)
    
    environment = os.getenv("ENVIRONMENT", "development")
    print(f"Current environment: {environment}")
    
    if environment == "development":
        print("\n📋 Local Development Configuration:")
        print("   - Using service account key file")
        print("   - Environment: development")
        print("   - Debug mode: enabled")
        
        # Check for service account key
        service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
        if os.path.exists(service_account_path):
            print(f"   ✅ Service account key: {service_account_path}")
        else:
            print(f"   ❌ Service account key not found: {service_account_path}")
            print("   💡 Copy serviceAccountKey.json to the project root")
            return False
            
    elif environment == "production":
        print("\n📋 Production Configuration:")
        print("   - Using default credentials (Cloud Run)")
        print("   - Environment: production")
        print("   - Debug mode: disabled")
        print("   ✅ No service account key needed")
    
    return True

def main():
    """Main test function"""
    print("🚀 Moments Face Tagging Service - Authentication Test")
    print("=" * 60)
    
    # Test environment setup
    if not test_environment_setup():
        print("\n❌ Environment setup test failed")
        sys.exit(1)
    
    # Test authentication
    if not test_authentication():
        print("\n❌ Authentication test failed")
        sys.exit(1)
    
    print("\n✅ All tests passed! Authentication is properly configured.")
    print("\n📚 Next steps:")
    print("   1. For local development: Ensure serviceAccountKey.json is in the project root")
    print("   2. For production: Deploy using the CI/CD pipeline")
    print("   3. Monitor logs for any authentication issues")

if __name__ == "__main__":
    main()
