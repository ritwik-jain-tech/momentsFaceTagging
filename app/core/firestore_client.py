"""
Firestore client for Moments Face Tagging Service
Handles all database operations for face embeddings and moment tagging
"""

import logging
from typing import List, Dict, Any, Optional
from google.cloud import firestore
from google.cloud.firestore import FieldFilter
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import numpy as np

from app.models.face_embeddings import UserFaceEmbedding, MomentFaceEmbedding, FaceMatch
from app.core.auth_config import get_firestore_client

logger = logging.getLogger(__name__)


class FirestoreClient:
    """
    Firestore client for face embedding and moment management
    """
    
    def __init__(self):
        self.db: Optional[firestore.Client] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Firestore client with proper authentication"""
        try:
            logger.info("Initializing Firestore client with environment-based authentication...")
            self.db = get_firestore_client()
            logger.info("Firestore client initialized successfully with proper authentication")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            raise
    
    def _convert_numpy_types(self, data):
        """Convert NumPy types to native Python types for Firestore compatibility"""
        if isinstance(data, dict):
            return {key: self._convert_numpy_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
    
    async def initialize_collections(self):
        """Initialize required collections in Firestore"""
        try:
            logger.info("Initializing Firestore collections...")
            
            # Create user_face_embeddings collection with a sample document
            user_collection = self.db.collection('user_face_embeddings')
            sample_user_doc = {
                'user_id': '_initialization_doc',
                'embedding': [0.0] * 512,
                'quality_score': 0.0,
                'detection_score': 0.0,
                'bbox': [0, 0, 0, 0],
                'age': None,
                'gender': None,
                'selfie_url': 'https://example.com/init.jpg',
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            
            # Create moment_face_embeddings collection with a sample document
            moment_collection = self.db.collection('moment_face_embeddings')
            sample_moment_doc = {
                'moment_id': '_initialization_doc',
                'face_embeddings': [],
                'face_count': 0,
                'moment_url': 'https://example.com/init.jpg',
                'processed_at': datetime.utcnow()
            }
            
            # Create face_matches collection with a sample document
            matches_collection = self.db.collection('face_matches')
            sample_match_doc = {
                'moment_id': '_initialization_doc',
                'matches': [],
                'created_at': datetime.utcnow()
            }
            
            # Write initialization documents (these will be deleted later)
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: user_collection.document('_init').set(sample_user_doc)
            )
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: moment_collection.document('_init').set(sample_moment_doc)
            )
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: matches_collection.document('_init').set(sample_match_doc)
            )
            
            # Delete initialization documents
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: user_collection.document('_init').delete()
            )
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: moment_collection.document('_init').delete()
            )
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: matches_collection.document('_init').delete()
            )
            
            logger.info("✅ Firestore collections initialized successfully")
            logger.info("   - user_face_embeddings collection created")
            logger.info("   - moment_face_embeddings collection created")
            logger.info("   - face_matches collection created")
            
        except Exception as e:
            logger.error(f"Failed to initialize collections: {e}")
            raise
    
    async def store_user_embedding(self, user_id: str, event_id: str, embedding: List[float], 
                                 image_url: str, face_id: str) -> bool:
        """
        Store user face embedding in Firestore
        """
        try:
            logger.info(f"Storing face embedding for user {user_id} in event {event_id}")
            
            # Create embedding document
            embedding_doc = {
                'user_id': user_id,
                'event_id': event_id,
                'embedding': embedding,
                'image_url': str(image_url),  # Convert HttpUrl to string
                'face_id': face_id,
                'created_at': time.time(),
                'updated_at': time.time()
            }
            
            # Store in Firestore
            doc_ref = self.db.collection('user_face_embeddings').document(user_id)
            doc_ref.set(embedding_doc)
            
            logger.info(f"Successfully stored face embedding for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store user embedding: {e}")
            return False
    
    async def get_user_embeddings(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get all user embeddings for an event
        """
        try:
            logger.info(f"Retrieving user embeddings for event {event_id}")
            
            # Query embeddings for the event
            query = self.db.collection('face_embeddings').where(
                filter=FieldFilter("event_id", "==", event_id)
            )
            
            docs = query.stream()
            embeddings = []
            
            for doc in docs:
                doc_data = doc.to_dict()
                embeddings.append({
                    'doc_id': doc.id,
                    'user_id': doc_data.get('user_id'),
                    'embedding': doc_data.get('embedding'),
                    'face_id': doc_data.get('face_id'),
                    'image_url': doc_data.get('image_url'),
                    'created_at': doc_data.get('created_at')
                })
            
            logger.info(f"Retrieved {len(embeddings)} user embeddings for event {event_id}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get user embeddings: {e}")
            return []
    
    async def get_eligible_moments(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get all moments eligible for feed in an event
        """
        try:
            logger.info(f"Retrieving eligible moments for event {event_id}")
            
            # Query moments for the event that are eligible for feed
            query = self.db.collection('moments').where(
                filter=FieldFilter("eventId", "==", event_id)
            ).where(
                filter=FieldFilter("status", "==", "APPROVED")
            ).order_by(
                "creationTime", direction="DESCENDING"
            )
            
            docs = query.stream()
            moments = []
            
            for doc in docs:
                doc_data = doc.to_dict()
                # Get media URL from the media object
                media = doc_data.get('media', {})
                image_url = None
                if isinstance(media, dict):
                    image_url = media.get('url') or media.get('imageUrl')
                
                moments.append({
                    'moment_id': doc.id,
                    'event_id': doc_data.get('eventId'),
                    'user_id': doc_data.get('creatorId'),
                    'image_url': image_url,
                    'created_at': doc_data.get('creationTime'),
                    'user_present': doc_data.get('taggedUserIds', [])  # Use taggedUserIds field
                })
            
            logger.info(f"Retrieved {len(moments)} eligible moments for event {event_id}")
            return moments
            
        except Exception as e:
            logger.error(f"Failed to get eligible moments: {e}")
            return []

    async def get_all_moments_for_event(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get all moments for an event (including non-approved ones)
        Used for user tag removal operations
        """
        try:
            logger.info(f"Retrieving all moments for event {event_id}")
            
            # Query all moments for the event (no status filter)
            query = self.db.collection('moments').where(
                filter=FieldFilter("eventId", "==", event_id)
            )
            
            docs = query.stream()
            moments = []
            
            for doc in docs:
                doc_data = doc.to_dict()
                # Get media URL from the media object
                media = doc_data.get('media', {})
                image_url = None
                if isinstance(media, dict):
                    image_url = media.get('url') or media.get('imageUrl')
                
                moments.append({
                    'moment_id': doc.id,
                    'event_id': doc_data.get('eventId'),
                    'user_id': doc_data.get('creatorId'),
                    'image_url': image_url,
                    'created_at': doc_data.get('creationTime'),
                    'status': doc_data.get('status'),
                    'user_present': doc_data.get('taggedUserIds', [])  # Use taggedUserIds field
                })
            
            logger.info(f"Retrieved {len(moments)} total moments for event {event_id}")
            return moments
            
        except Exception as e:
            logger.error(f"Failed to get all moments for event: {e}")
            return []
    
    async def update_moment_user_present(self, moment_id: str, user_ids: List[str]) -> bool:
        """
        Update the user_present list for a moment
        """
        try:
            logger.info(f"Updating user_present for moment {moment_id}: {user_ids}")
            
            # Update the moment document using the same field name as momentsBackend
            doc_ref = self.db.collection('moments').document(moment_id)
            doc_ref.update({
                'taggedUserIds': user_ids,
                'updatedAt': time.time()
            })
            
            logger.info(f"Successfully updated user_present for moment {moment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update moment user_present: {e}")
            return False
    
    async def store_moment_faces(self, moment_id: str, event_id: str, faces: List[Dict[str, Any]]) -> bool:
        """
        Store detected faces for a moment
        """
        try:
            logger.info(f"Storing {len(faces)} faces for moment {moment_id}")
            
            # Create moment faces document
            moment_faces_doc = {
                'moment_id': moment_id,
                'event_id': event_id,
                'faces': faces,
                'face_count': len(faces),
                'created_at': time.time(),
                'updated_at': time.time()
            }
            
            # Store in Firestore
            doc_ref = self.db.collection('moment_face_embeddings').document(moment_id)
            doc_ref.set(moment_faces_doc)
            
            logger.info(f"Successfully stored faces for moment {moment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store moment faces: {e}")
            return False
    
    async def get_moment_faces(self, moment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored faces for a moment
        """
        try:
            logger.info(f"Retrieving faces for moment {moment_id}")
            
            doc_ref = self.db.collection('moment_face_embeddings').document(moment_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                logger.warning(f"No faces found for moment {moment_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get moment faces: {e}")
            return None
    
    async def get_moment_by_id(self, moment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get moment details by ID
        """
        try:
            logger.info(f"Retrieving moment {moment_id}")
            
            doc_ref = self.db.collection('moments').document(moment_id)
            doc = doc_ref.get()
            
            if doc.exists:
                moment_data = doc.to_dict()
                moment_data['moment_id'] = doc.id
                return moment_data
            else:
                logger.warning(f"Moment {moment_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get moment: {e}")
            return None
    
    async def remove_user_tagging_from_event(self, user_id: str, event_id: str) -> Dict[str, Any]:
        """
        Remove user tagging from all moments in an event
        """
        try:
            logger.info(f"Removing user {user_id} tagging from all moments in event {event_id}")
            
            # Get all moments for the event
            moments_query = self.db.collection('moments').where('eventId', '==', event_id)
            moments = moments_query.stream()
            
            updated_moments = 0
            total_moments = 0
            errors = []
            
            for moment_doc in moments:
                total_moments += 1
                moment_id = moment_doc.id
                moment_data = moment_doc.to_dict()
                
                try:
                    # Get current taggedUserIds
                    tagged_user_ids = moment_data.get('taggedUserIds', [])
                    
                    if user_id in tagged_user_ids:
                        # Remove user from taggedUserIds
                        tagged_user_ids.remove(user_id)
                        
                        # Update the moment
                        moment_doc.reference.update({
                            'taggedUserIds': tagged_user_ids
                        })
                        
                        updated_moments += 1
                        logger.info(f"Removed user {user_id} from moment {moment_id}")
                    else:
                        logger.info(f"User {user_id} not found in moment {moment_id}")
                        
                except Exception as e:
                    error_msg = f"Failed to update moment {moment_id}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            result = {
                'success': True,
                'message': f'Removed user {user_id} tagging from {updated_moments} out of {total_moments} moments in event {event_id}',
                'user_id': user_id,
                'event_id': event_id,
                'total_moments': total_moments,
                'updated_moments': updated_moments,
                'errors': errors
            }
            
            logger.info(f"User tagging removal completed: {updated_moments}/{total_moments} moments updated")
            return result
            
        except Exception as e:
            logger.error(f"Failed to remove user tagging from event: {e}")
            return {
                'success': False,
                'message': f'Failed to remove user tagging: {str(e)}',
                'user_id': user_id,
                'event_id': event_id,
                'total_moments': 0,
                'updated_moments': 0,
                'errors': [str(e)]
            }
    
    # Face Embedding Collection Methods
    
    async def store_user_face_embedding(self, user_embedding: UserFaceEmbedding) -> bool:
        """Store user face embedding in Firestore"""
        try:
            logger.info(f"Storing user face embedding for user {user_embedding.user_id}")
            
            # Convert to dict for Firestore
            embedding_data = user_embedding.dict()
            embedding_data['created_at'] = user_embedding.created_at
            embedding_data['updated_at'] = user_embedding.updated_at
            
            # Convert NumPy types to native Python types
            embedding_data = self._convert_numpy_types(embedding_data)
            
            # Store in user_face_embeddings collection using async executor
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.db.collection('user_face_embeddings').document(user_embedding.user_id+'_'+user_embedding.event_id).set(embedding_data)
            )
            
            logger.info(f"Successfully stored user face embedding for user {user_embedding.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store user face embedding: {e}")
            return False
    
    async def get_user_face_embedding(self, user_id: str) -> Optional[UserFaceEmbedding]:
        """Get user face embedding by user_id"""
        try:
            doc_ref = self.db.collection('user_face_embeddings').document(user_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                return UserFaceEmbedding(**data)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user face embedding: {e}")
            return None
    
    async def get_all_user_face_embeddings(self) -> List[UserFaceEmbedding]:
        """Get all user face embeddings"""
        try:
            embeddings = []
            docs = self.db.collection('user_face_embeddings').stream()
            
            for doc in docs:
                data = doc.to_dict()
                embeddings.append(UserFaceEmbedding(**data))
            
            logger.info(f"Retrieved {len(embeddings)} user face embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get user face embeddings: {e}")
            return []
    
    async def delete_user_face_embedding(self, user_id: str) -> bool:
        """Delete user face embedding"""
        try:
            doc_ref = self.db.collection('user_face_embeddings').document(user_id)
            doc_ref.delete()
            
            logger.info(f"Deleted user face embedding for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user face embedding: {e}")
            return False
    
    async def store_moment_face_embedding(self, moment_embedding: MomentFaceEmbedding) -> bool:
        """Store moment face embedding in Firestore"""
        try:
            logger.info(f"Storing moment face embedding for moment {moment_embedding.moment_id}")
            
            # Convert to dict for Firestore
            embedding_data = moment_embedding.dict()
            embedding_data['processed_at'] = moment_embedding.processed_at
            
            # Convert NumPy types to native Python types
            embedding_data = self._convert_numpy_types(embedding_data)
            
            # Store in moment_face_embeddings collection using async executor
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.db.collection('moment_face_embeddings').document(moment_embedding.moment_id).set(embedding_data)
            )
            
            logger.info(f"Successfully stored moment face embedding for moment {moment_embedding.moment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store moment face embedding: {e}")
            return False
    
    async def get_moment_face_embedding(self, moment_id: str) -> Optional[MomentFaceEmbedding]:
        """Get moment face embedding by moment_id"""
        try:
            doc_ref = self.db.collection('moment_face_embeddings').document(moment_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                return MomentFaceEmbedding(**data)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get moment face embedding: {e}")
            return None
    
    async def delete_moment_face_embedding(self, moment_id: str) -> bool:
        """Delete moment face embedding"""
        try:
            doc_ref = self.db.collection('moment_face_embeddings').document(moment_id)
            doc_ref.delete()
            
            logger.info(f"Deleted moment face embedding for moment {moment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete moment face embedding: {e}")
            return False
    
    async def store_face_matches(self, moment_id: str, matches: List[FaceMatch]) -> bool:
        """Store face matches for a moment"""
        try:
            logger.info(f"Storing {len(matches)} face matches for moment {moment_id}")
            
            # Convert matches to dict for Firestore
            matches_data = [match.dict() for match in matches]
            
            # Convert NumPy types to native Python types
            matches_data = self._convert_numpy_types(matches_data)
            
            # Prepare document data
            doc_data = {
                'moment_id': moment_id,
                'matches': matches_data,
                'match_count': len(matches),
                'created_at': datetime.utcnow()
            }
            
            # Store in face_matches collection using async executor
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.db.collection('face_matches').document(moment_id).set(doc_data)
            )
            
            logger.info(f"Successfully stored face matches for moment {moment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store face matches: {e}")
            return False
    
    async def get_face_matches(self, moment_id: str) -> List[FaceMatch]:
        """Get face matches for a moment"""
        try:
            doc_ref = self.db.collection('face_matches').document(moment_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                matches_data = data.get('matches', [])
                return [FaceMatch(**match) for match in matches_data]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to get face matches: {e}")
            return []
    
    async def get_all_moment_face_embeddings(self) -> List[MomentFaceEmbedding]:
        """Get all moment face embeddings"""
        try:
            embeddings = []
            docs = self.db.collection('moment_face_embeddings').stream()
            
            for doc in docs:
                data = doc.to_dict()
                embeddings.append(MomentFaceEmbedding(**data))
            
            logger.info(f"Retrieved {len(embeddings)} moment face embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get moment face embeddings: {e}")
            return []
    
    async def get_moment_face_embeddings_by_event(self, event_id: str) -> List[MomentFaceEmbedding]:
        """Get moment face embeddings for a specific event"""
        try:
            embeddings = []
            docs = self.db.collection('moment_face_embeddings').where('event_id', '==', event_id).stream()
            
            for doc in docs:
                data = doc.to_dict()
                embeddings.append(MomentFaceEmbedding(**data))
            
            logger.info(f"Retrieved {len(embeddings)} moment face embeddings for event {event_id}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get moment face embeddings for event {event_id}: {e}")
            return []
    
    async def get_user_face_embeddings_by_event(self, event_id: str) -> List[UserFaceEmbedding]:
        """Get user face embeddings for a specific event"""
        try:
            embeddings = []
            docs = self.db.collection('user_face_embeddings').where('event_id', '==', event_id).stream()
            
            for doc in docs:
                data = doc.to_dict()
                embeddings.append(UserFaceEmbedding(**data))
            
            logger.info(f"Retrieved {len(embeddings)} user face embeddings for event {event_id}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get user face embeddings for event {event_id}: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            logger.info("Firestore client cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_moment_tagged_users(self, moment_id: str) -> List[str]:
        """Get current taggedUserIds for a specific moment"""
        try:
            logger.info(f"Getting tagged users for moment {moment_id}")
            
            # Get the moment document from Firestore
            doc = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.db.collection('moments').document(moment_id).get()
            )
            
            if doc.exists:
                data = doc.to_dict()
                tagged_users = data.get('taggedUserIds', [])
                logger.info(f"Found {len(tagged_users)} tagged users for moment {moment_id}")
                return tagged_users
            else:
                logger.warning(f"Moment {moment_id} not found in Firestore")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get tagged users for moment {moment_id}: {e}")
            return []

    async def update_moment_tagged_users(self, moment_id: str, tagged_users: List[str]) -> bool:
        """Update taggedUserIds for a specific moment"""
        try:
            logger.info(f"Updating tagged users for moment {moment_id}: {tagged_users}")
            
            # Update the moment document in Firestore
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.db.collection('moments').document(moment_id).update({
                    'taggedUserIds': tagged_users,
                    'updated_at': time.time()
                })
            )
            
            logger.info(f"Successfully updated tagged users for moment {moment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update tagged users for moment {moment_id}: {e}")
            return False
    
    async def update_moment_media_urls(self, moment_id: str, media_url: str, feed_url: str) -> bool:
        """
        Update moment media URLs (media.url and media.feedUrl)
        
        Args:
            moment_id: The moment ID
            media_url: High quality JPEG URL (for media.url)
            feed_url: Compressed WebP URL (for media.feedUrl)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Updating media URLs for moment {moment_id}")
            logger.info(f"  media.url: {media_url}")
            logger.info(f"  media.feedUrl: {feed_url}")
            
            # Update the moment document in Firestore
            # Using nested field update for media object
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.db.collection('moments').document(moment_id).update({
                    'media.url': media_url,
                    'media.feedUrl': feed_url,
                    'updated_at': time.time()
                })
            )
            
            logger.info(f"Successfully updated media URLs for moment {moment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update media URLs for moment {moment_id}: {e}")
            return False