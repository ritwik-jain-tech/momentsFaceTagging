"""
Production-grade face recognition service using InsightFace
Provides state-of-the-art face recognition accuracy with real deep learning models
Handles user_face_embedding and moment_face_embedding collections
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from typing import List, Dict, Any, Tuple, Optional
import logging
import os
from PIL import Image, ImageOps
import requests
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

from app.models.face_embeddings import (
    UserFaceEmbedding, 
    MomentFaceEmbedding, 
    FaceMatch, 
    FaceEmbeddingResponse
)

logger = logging.getLogger(__name__)

class InsightFaceRecognitionService:
    """
    Production-grade face recognition service using InsightFace
    Provides state-of-the-art accuracy with real deep learning models
    """
    
    def __init__(self):
        """Initialize the InsightFace recognition service."""
        self.face_app: Optional[FaceAnalysis] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize model asynchronously to avoid blocking startup
        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(f"Failed to initialize InsightFace model during startup: {e}")
            logger.info("Model will be initialized on first use")
        
        # FaceNet-style parameters for matching
        self.FACE_MATCH_THRESHOLD = 0.4  # Lowered threshold for more matches
        self.EMBEDDING_SIZE = 512         # InsightFace standard embedding size
        
        # Quality thresholds - adaptive for challenging conditions
        self.MIN_FACE_QUALITY = 0.2  # Lowered for challenging conditions
        self.MIN_DETECTION_SCORE = 0.3  # Lowered for low lighting/side angles
        
        # Adaptive quality thresholds based on conditions
        self.QUALITY_THRESHOLDS = {
            'excellent': 0.7,  # High quality, good lighting, frontal
            'good': 0.5,       # Decent quality, some challenges
            'challenging': 0.3, # Low lighting, side angles, poor quality
            'minimal': 0.2     # Very challenging conditions
        }
        
        # Anti-false-positive measures
        self.user_match_cooldown = {}
        self.max_matches_per_user_per_hour = 2
        
        logger.info("InsightFace Recognition Service initialized")
    
    def _cleanup_memory(self):
        
        import gc
        gc.collect()
    
    def _initialize_model(self):
        """Initialize the InsightFace model with optimal settings."""
        try:
            logger.info("Initializing InsightFace model...")
            
            # Configure InsightFace for production use
            # Using 'buffalo_s' model for better memory efficiency
            self.face_app = FaceAnalysis(
                name='buffalo_s',  # Smaller model for better memory usage
                providers=['CPUExecutionProvider']  # Use CPU for Cloud Run compatibility
            )
            
            # Prepare the model with smaller detection size for memory efficiency
            self.face_app.prepare(ctx_id=0, det_size=(320, 320))
            
            logger.info("InsightFace model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace model: {e}")
            # Fallback to even smaller model if buffalo_s fails
            try:
                logger.info("Trying fallback model...")
                self.face_app = FaceAnalysis(
                    name='buffalo_m',  # Even smaller model
                    providers=['CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=0, det_size=(320, 320))
                logger.info("Fallback InsightFace model initialized successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                # Don't raise exception - allow service to start without model
                # Model will be initialized on first use
                logger.warning("Service will start without face recognition model. Model will be initialized on first use.")
                self.face_app = None
    
    def _ensure_model_initialized(self):
        """Ensure the InsightFace model is initialized, initialize if needed."""
        if self.face_app is None:
            logger.info("Initializing InsightFace model on first use...")
            try:
                self.face_app = FaceAnalysis(
                    name='buffalo_s',
                    providers=['CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=0, det_size=(320, 320))
                logger.info("InsightFace model initialized successfully on first use")
            except Exception as e:
                logger.error(f"Failed to initialize InsightFace model on first use: {e}")
                # Try fallback model
                try:
                    self.face_app = FaceAnalysis(
                        name='buffalo_m',
                        providers=['CPUExecutionProvider']
                    )
                    self.face_app.prepare(ctx_id=0, det_size=(320, 320))
                    logger.info("Fallback InsightFace model initialized successfully on first use")
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed on first use: {fallback_error}")
                    raise Exception("Unable to initialize any InsightFace model")
    
    async def download_image(self, image_url: str) -> np.ndarray:
        """Download and preprocess image with enhanced quality."""
        try:
            logger.info(f"Downloading image from: {image_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
            }
            
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            image_data = response.content
            
            # Use PIL to handle EXIF orientation, then convert to numpy array
            # This ensures images are correctly oriented before face detection
            try:
                pil_image = Image.open(BytesIO(image_data))
                pil_image.load()
                # Apply EXIF orientation to fix rotation issues
                pil_image = ImageOps.exif_transpose(pil_image)
                logger.info(f"Applied EXIF orientation correction for face detection")
                
                # Convert PIL image to numpy array (RGB)
                image = np.array(pil_image)
                
                # If image is grayscale, convert to RGB
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3 and image.dtype == np.uint8:
                    # Already RGB, ensure it's the right format
                    pass
                else:
                    # Fallback to OpenCV decode
                    image_array = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Failed to decode image")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as pil_error:
                logger.warning(f"Failed to use PIL for EXIF handling, falling back to OpenCV: {pil_error}")
                # Fallback to OpenCV if PIL fails
                image_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise ValueError("Failed to decode image")
                
                # Convert BGR to RGB for InsightFace
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Enhanced preprocessing for better face detection
            image = self._enhance_image_quality(image)
            
            logger.info(f"Successfully downloaded and processed image: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to download image {image_url}: {e}")
            raise
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better face detection."""
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    def _detect_faces_insightface(self, image: np.ndarray, is_selfie: bool = False) -> List[Dict[str, Any]]:
        """Detect faces using InsightFace with high accuracy."""
        try:
            # Ensure model is initialized
            self._ensure_model_initialized()
            
            # Convert RGB to BGR for InsightFace
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect faces using InsightFace
            faces = self.face_app.get(bgr_image)
            
            if not faces:
                logger.info("No faces detected with InsightFace")
                return []
            
            processed_faces = []
            for i, face in enumerate(faces):
                # Extract face information
                bbox = face.bbox.astype(int).tolist()
                embedding = face.embedding
                kps = face.kps
                det_score = face.det_score
                age = getattr(face, 'age', None)
                gender = getattr(face, 'gender', None)
                
                # Quality assessment
                quality_score = self._assess_face_quality(face, image, bbox)
                
                # Filter by quality and detection score
                if (quality_score >= self.MIN_FACE_QUALITY and 
                    det_score >= self.MIN_DETECTION_SCORE):
                    
                    # Extract face region
                    x1, y1, x2, y2 = bbox
                    face_region = image[y1:y2, x1:x2]
                    
                    processed_faces.append({
                        'bbox': bbox,
                        'embedding': embedding,
                        'kps': kps,
                        'det_score': det_score,
                        'age': age,
                        'gender': gender,
                        'quality_score': quality_score,
                        'face_region': face_region
                    })
            
            # Sort by quality score
            processed_faces.sort(key=lambda x: x['quality_score'], reverse=True)
            
            logger.info(f"Detected {len(processed_faces)} high-quality faces using InsightFace")
            
            # Debug logging for small faces
            for i, face in enumerate(processed_faces):
                bbox = face['bbox']
                quality = face['quality_score']
                det_score = face['det_score']
                logger.info(f"Face {i}: bbox={bbox}, quality={quality:.3f}, det_score={det_score:.3f}")
            
            # Clean up memory after processing
            self._cleanup_memory()
            return processed_faces
            
        except Exception as e:
            logger.error(f"InsightFace face detection failed: {e}")
            return []
    
    def _assess_face_quality(self, face, full_image: np.ndarray, bbox: List[int]) -> float:
        """Assess face quality using multiple criteria with adaptive scoring for challenging conditions."""
        try:
            quality_score = 0.0
            
            # 1. Detection score from InsightFace (primary indicator)
            det_score = face.det_score
            quality_score += min(det_score, 1.0) * 0.4  # 40% weight to detection score
            
            # 2. Face size and position (adaptive for small faces)
            x1, y1, x2, y2 = bbox
            face_area = (x2 - x1) * (y2 - y1)
            img_area = full_image.shape[0] * full_image.shape[1]
            size_ratio = face_area / img_area
            
            if 0.01 <= size_ratio <= 0.5:
                quality_score += 0.3  # Good size
            elif 0.005 <= size_ratio <= 0.8:
                quality_score += 0.15  # Acceptable size
            elif 0.001 <= size_ratio < 0.005:
                quality_score += 0.1  # Small faces - partial credit
            
            # 3. Face position (less strict for side angles)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            img_center_x = full_image.shape[1] / 2
            img_center_y = full_image.shape[0] / 2
            
            distance_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_distance = np.sqrt(full_image.shape[1]**2 + full_image.shape[0]**2) / 2
            
            if distance_from_center < max_distance * 0.3:
                quality_score += 0.2  # Well centered
            elif distance_from_center < max_distance * 0.5:
                quality_score += 0.1  # Reasonably centered
            elif distance_from_center < max_distance * 0.8:
                quality_score += 0.05  # Side angles - minimal credit
            
            # 4. Face landmarks quality (if available)
            if hasattr(face, 'kps') and face.kps is not None:
                kps_quality = self._assess_landmarks_quality(face.kps, bbox)
                quality_score += kps_quality * 0.1
            
            # 5. Image quality assessment (lighting, blur, etc.)
            image_quality = self._assess_image_quality(full_image, bbox)
            quality_score += image_quality * 0.1
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Face quality assessment failed: {e}")
            return 0.0
    
    def _assess_landmarks_quality(self, kps: np.ndarray, bbox: List[int]) -> float:
        """Assess the quality of facial landmarks."""
        try:
            if kps is None or len(kps) == 0:
                return 0.0
            
            x1, y1, x2, y2 = bbox
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Check if landmarks are within reasonable bounds
            valid_landmarks = 0
            for kp in kps:
                if (x1 <= kp[0] <= x2 and y1 <= kp[1] <= y2):
                    valid_landmarks += 1
            
            return valid_landmarks / len(kps) if len(kps) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Landmarks quality assessment failed: {e}")
            return 0.0
    
    def _assess_image_quality(self, image: np.ndarray, bbox: List[int]) -> float:
        """Assess image quality factors like lighting, blur, and contrast."""
        try:
            x1, y1, x2, y2 = bbox
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale for analysis
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_region
            
            quality_score = 0.0
            
            # 1. Lighting assessment (variance in brightness)
            brightness_variance = np.var(gray_face)
            if brightness_variance > 1000:  # Good lighting variation
                quality_score += 0.3
            elif brightness_variance > 500:  # Acceptable lighting
                quality_score += 0.2
            elif brightness_variance > 200:  # Poor lighting but usable
                quality_score += 0.1
            
            # 2. Contrast assessment
            contrast = np.std(gray_face)
            if contrast > 50:  # Good contrast
                quality_score += 0.3
            elif contrast > 30:  # Acceptable contrast
                quality_score += 0.2
            elif contrast > 15:  # Low contrast but usable
                quality_score += 0.1
            
            # 3. Blur assessment (Laplacian variance)
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if blur_score > 100:  # Sharp image
                quality_score += 0.4
            elif blur_score > 50:  # Slightly blurred but usable
                quality_score += 0.3
            elif blur_score > 20:  # Blurry but might be usable
                quality_score += 0.2
            elif blur_score > 10:  # Very blurry but still attempt
                quality_score += 0.1
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Image quality assessment failed: {e}")
            return 0.0
    
    def _get_adaptive_similarity_threshold(self, face_quality: float) -> float:
        """Get adaptive similarity threshold based on face quality."""
        if face_quality >= 0.7:
            return 0.35  # High quality faces - lowered threshold for better matching
        elif face_quality >= 0.5:
            return 0.32  # Good quality faces - slightly lower
        elif face_quality >= 0.3:
            return 0.3  # Challenging conditions - lower threshold
        else:
            return 0.25  # Very challenging conditions - lowest threshold
    
    def _get_adaptive_quality_threshold(self, face_quality: float) -> float:
        """Get adaptive quality threshold based on face quality."""
        if face_quality >= 0.7:
            return 0.3  # High quality faces - standard threshold
        elif face_quality >= 0.5:
            return 0.25  # Good quality faces - slightly lower
        elif face_quality >= 0.3:
            return 0.2  # Challenging conditions - lower threshold
        else:
            return 0.15  # Very challenging conditions - lowest threshold
    
    def _calculate_confidence_score(self, similarity: float, quality: float, 
                                  confidence_gap: float, second_best: float) -> float:
        """Calculate comprehensive confidence score to reduce false positives."""
        try:
            # Base confidence from similarity and quality
            base_confidence = similarity * quality
            
            # Confidence gap bonus (larger gap = more confident)
            gap_bonus = min(confidence_gap * 2, 0.2)  # Max 0.2 bonus
            
            # Distance from second best (larger distance = more confident)
            distance_bonus = min((similarity - second_best) * 0.5, 0.1)  # Max 0.1 bonus
            
            # Quality bonus for high-quality faces
            quality_bonus = 0.0
            if quality >= 0.7:
                quality_bonus = 0.1
            elif quality >= 0.5:
                quality_bonus = 0.05
            
            # Similarity bonus for very high similarity
            similarity_bonus = 0.0
            if similarity >= 0.8:
                similarity_bonus = 0.1
            elif similarity >= 0.6:
                similarity_bonus = 0.05
            
            total_confidence = base_confidence + gap_bonus + distance_bonus + quality_bonus + similarity_bonus
            
            return min(total_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return similarity * quality  # Fallback to simple calculation
    
    async def extract_face_embedding(self, image_url: str) -> Dict[str, Any]:
        """Extract high-quality face embedding using InsightFace."""
        try:
            logger.info(f"Extracting InsightFace embedding from: {image_url}")
            
            # Download and preprocess image
            image = await self.download_image(image_url)
            
            # Detect faces using InsightFace
            faces = self._detect_faces_insightface(image, is_selfie=True)
            
            if not faces:
                return {
                    'success': False,
                    'message': 'No high-quality faces detected',
                    'faces': [],
                    'face_count': 0
                }
            
            # Select best face for selfie
            best_face = faces[0]  # Already sorted by quality
            
            return {
                'success': True,
                'message': f'Successfully extracted InsightFace embedding (quality: {best_face["quality_score"]:.3f})',
                'embedding': best_face['embedding'].tolist(),
                'face_count': len(faces),
                'quality_score': best_face['quality_score'],
                'detection_score': best_face['det_score'],
                'bbox': best_face['bbox'],
                'age': best_face.get('age'),
                'gender': best_face.get('gender')
            }
            
        except Exception as e:
            logger.error(f"InsightFace embedding extraction failed: {e}")
            return {
                'success': False,
                'message': f'InsightFace embedding extraction failed: {str(e)}',
                'faces': [],
                'face_count': 0
            }
    
    async def process_moment_image(self, image_url: str) -> Dict[str, Any]:
        """Process moment image with InsightFace detection."""
        try:
            logger.info(f"Processing moment image with InsightFace: {image_url}")
            
            # Download and preprocess image
            image = await self.download_image(image_url)
            
            # Detect faces using InsightFace
            faces = self._detect_faces_insightface(image, is_selfie=False)
            
            if not faces:
                return {
                    'success': True,
                    'message': 'No faces detected in moment',
                    'faces': [],
                    'face_count': 0
                }
            
            # Process each detected face
            processed_faces = []
            for i, face in enumerate(faces):
                try:
                    processed_face = {
                        'face_id': f"face_{i}",
                        'bbox': face['bbox'],
                        'embedding': face['embedding'].tolist(),
                        'quality_score': face['quality_score'],
                        'detection_score': face['det_score'],
                        'age': face.get('age'),
                        'gender': face.get('gender')
                    }
                    
                    processed_faces.append(processed_face)
                    
                except Exception as e:
                    logger.error(f"Failed to process face {i}: {e}")
                    continue
            
            logger.info(f"Processed {len(processed_faces)} faces with InsightFace")
            
            return {
                'success': True,
                'message': f'Successfully processed {len(processed_faces)} faces',
                'faces': processed_faces,
                'face_count': len(processed_faces)
            }
            
        except Exception as e:
            logger.error(f"InsightFace moment processing failed: {e}")
            return {
                'success': False,
                'message': f'InsightFace moment processing failed: {str(e)}',
                'faces': [],
                'face_count': 0
            }
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between InsightFace embeddings."""
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Ensure same length
            min_len = min(len(emb1), len(emb2))
            emb1 = emb1[:min_len]
            emb2 = emb2[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            logger.info(f"Similarity calculation: dot_product={dot_product:.6f}, norm1={norm1:.6f}, norm2={norm2:.6f}")
            
            if norm1 == 0 or norm2 == 0:
                logger.warning("Zero norm detected in similarity calculation")
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            logger.info(f"Raw similarity: {similarity:.6f}")
            
            # Apply quality checks
            similarity = self._apply_quality_checks(similarity, emb1, emb2)
            logger.info(f"Final similarity after quality checks: {similarity:.6f}")
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"InsightFace similarity calculation failed: {e}")
            return 0.0
    
    def _apply_quality_checks(self, similarity: float, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Apply quality checks to similarity score."""
        try:
            # Check for zero vectors
            if np.all(emb1 == 0) or np.all(emb2 == 0):
                return 0.0
            
            # Check for identical vectors (perfect match)
            if np.allclose(emb1, emb2, atol=1e-6):
                return 1.0
            
            # Apply confidence scaling based on vector norms
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            # Penalize very small norms (low confidence)
            if norm1 < 0.1 or norm2 < 0.1:
                similarity *= 0.5
            
            return similarity
            
        except Exception as e:
            logger.error(f"Quality checks failed: {e}")
            return similarity
    
    async def match_faces(self, face_embeddings: List[Dict[str, Any]], user_embeddings: List[Dict[str, Any]], threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Match face embeddings with user embeddings using InsightFace similarity."""
        try:
            logger.info(f"InsightFace matching: {len(face_embeddings)} faces vs {len(user_embeddings)} users (threshold: {threshold})")
            
            matches = []
            matched_users = set()  # Prevent multiple faces from matching the same user
            
            for face_idx, face_data in enumerate(face_embeddings):
                face_embedding = face_data.get('embedding')
                if not face_embedding:
                    continue
                
                # Check face quality
                face_quality = face_data.get('quality_score', 0.0)
                if face_quality < 0.2:  # Lowered quality threshold
                    logger.info(f"Skipping low-quality face {face_idx} (quality: {face_quality:.3f})")
                    continue
                
                best_match = None
                best_similarity = 0.0
                second_best_similarity = 0.0
                
                for user_data in user_embeddings:
                    user_id = user_data.get('user_id')
                    user_embedding = user_data.get('embedding')
                    
                    if not user_embedding or user_id in matched_users:
                        continue
                    
                    # Check cooldown to prevent over-matching (DISABLED FOR TESTING)
                    # current_time = time.time()
                    # if user_id in self.user_match_cooldown:
                    #     last_match_time = self.user_match_cooldown[user_id]
                    #     if current_time - last_match_time < 30:  # 30 second cooldown for testing
                    #         logger.info(f"Skipping user {user_id} due to cooldown")
                    #         continue
                    
                    # Calculate InsightFace similarity
                    similarity = self.calculate_similarity(face_embedding, user_embedding)
                    
                    # Track top 2 similarities for confidence check
                    if similarity > best_similarity:
                        second_best_similarity = best_similarity
                        best_similarity = similarity
                        best_match = {
                            'user_id': user_id,
                            'similarity': similarity,
                            'face_index': face_idx,
                            'face_quality': face_quality
                        }
                    elif similarity > second_best_similarity:
                        second_best_similarity = similarity
                
                # Apply matching criteria
                confidence_gap = best_similarity - second_best_similarity
                min_confidence_gap = 0.05  # Lowered confidence gap requirement
                
                if (best_match and 
                    best_similarity >= threshold and
                    confidence_gap >= min_confidence_gap and
                    face_quality >= 0.2):
                    
                    matches.append(best_match)
                    matched_users.add(best_match['user_id'])
                    
                    # Update cooldown for this user (DISABLED FOR TESTING)
                    # self.user_match_cooldown[best_match['user_id']] = time.time()
                    
                    logger.info(f"InsightFace match found: user {best_match['user_id']} with similarity {best_similarity:.3f} (quality: {best_match['face_quality']:.3f}, gap: {confidence_gap:.3f})")
                else:
                    logger.info(f"No confident match for face {face_idx} (best: {best_similarity:.3f}, second: {second_best_similarity:.3f}, gap: {confidence_gap:.3f}, quality: {face_quality:.3f})")
            
            # Limit total matches to prevent over-tagging
            max_matches = 5  # Allow more matches with better model
            if len(matches) > max_matches:
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                matches = matches[:max_matches]
                logger.info(f"Limited matches to {max_matches} to prevent over-tagging")
            
            logger.info(f"InsightFace matching completed: {len(matches)} high-confidence matches found")
            return matches
            
        except Exception as e:
            logger.error(f"InsightFace face matching failed: {e}")
            return []
    
    async def process_selfie_for_user(self, image_url: str, user_id: str) -> FaceEmbeddingResponse:
        """
        Process selfie image to create/update user_face_embedding and find matches in existing moments
        This should be called when a user uploads a selfie
        """
        try:
            logger.info(f"Processing selfie for user {user_id} from: {image_url}")
            
            # Download and preprocess image
            image = await self.download_image(image_url)
            
            # Detect faces using InsightFace
            faces = self._detect_faces_insightface(image, is_selfie=True)
            
            if not faces:
                return FaceEmbeddingResponse(
                    success=False,
                    message='No high-quality faces detected in selfie',
                    face_count=0
                )
            
            # Select best face for user (highest quality)
            best_face = faces[0]  # Already sorted by quality
            
            # Create user face embedding object
            # Convert gender to string if it's an integer
            gender = best_face.get('gender')
            if gender is not None:
                # Handle numpy types
                if hasattr(gender, 'item'):
                    gender = gender.item()
                
                if isinstance(gender, (int, np.integer)):
                    gender = "Male" if gender == 1 else "Female"
                elif isinstance(gender, (float, np.floating)):
                    gender = "Male" if gender > 0.5 else "Female"
                elif isinstance(gender, str):
                    pass  # Already a string
                else:
                    gender = str(gender)  # Convert to string as fallback
            
            # Note: UserFaceEmbedding object creation is now handled by the API endpoint
            # This method only processes the selfie and returns the embedding data
            
            logger.info(f"Successfully processed selfie for user {user_id} (quality: {best_face['quality_score']:.3f})")
            
            # Note: Face matching is now handled by the API endpoint, not here
            # This method only processes the selfie and extracts the embedding
            
            # Convert NumPy types to native Python types for serialization
            embedding_data = {
                'user_id': user_id,
                'embedding': best_face['embedding'].tolist(),
                'quality_score': float(best_face['quality_score']),
                'detection_score': float(best_face['det_score']),
                'bbox': [float(x) for x in best_face['bbox']],
                'age': int(best_face.get('age')) if best_face.get('age') is not None else None,
                'gender': gender
            }
            
            return FaceEmbeddingResponse(
                success=True,
                message=f'Successfully processed selfie for user {user_id}',
                face_count=len(faces),
                embeddings=[embedding_data],
                matches=[]  # No matches during selfie processing
            )
            
        except Exception as e:
            logger.error(f"Selfie processing failed for user {user_id}: {e}")
            return FaceEmbeddingResponse(
                success=False,
                message=f'Selfie processing failed: {str(e)}',
                face_count=0
            )
    
    async def process_moment_for_faces(self, image_url: str, moment_id: str, event_id: str = None) -> FaceEmbeddingResponse:
        """
        Process moment image to create moment_face_embedding
        This should be called when processing a moment image
        """
        try:
            logger.info(f"Processing moment {moment_id} from: {image_url}")
            
            # Download and preprocess image
            image = await self.download_image(image_url)
            
            # Detect faces using InsightFace
            faces = self._detect_faces_insightface(image, is_selfie=False)
            
            if not faces:
                return FaceEmbeddingResponse(
                    success=True,
                    message='No faces detected in moment',
                    face_count=0
                )
            
            # Process each detected face
            face_embeddings = []
            for i, face in enumerate(faces):
                # Convert gender to string if it's an integer
                gender = face.get('gender')
                if gender is not None:
                    # Handle numpy types
                    if hasattr(gender, 'item'):
                        gender = gender.item()
                    
                    if isinstance(gender, (int, np.integer)):
                        gender = "Male" if gender == 1 else "Female"
                    elif isinstance(gender, (float, np.floating)):
                        gender = "Male" if gender > 0.5 else "Female"
                    elif isinstance(gender, str):
                        pass  # Already a string
                    else:
                        gender = str(gender)  # Convert to string as fallback
                
                face_data = {
                    'face_id': f"face_{i}",
                    'embedding': face['embedding'].tolist(),
                    'quality_score': face['quality_score'],
                    'detection_score': face['det_score'],
                    'bbox': face['bbox'],
                    'age': face.get('age'),
                    'gender': gender
                }
                face_embeddings.append(face_data)
            
            # Create moment face embedding object
            moment_embedding = MomentFaceEmbedding(
                moment_id=moment_id,
                face_embeddings=face_embeddings,
                face_count=len(faces),
                moment_url=image_url,
                event_id=event_id
            )
            
            logger.info(f"Successfully processed moment {moment_id} with {len(faces)} faces")
            
            # Convert NumPy types to native Python types for serialization
            converted_embeddings = []
            for face_data in face_embeddings:
                converted_face = {
                    'face_id': face_data['face_id'],
                    'embedding': face_data['embedding'],
                    'quality_score': float(face_data['quality_score']),
                    'detection_score': float(face_data['detection_score']),
                    'bbox': [float(x) for x in face_data['bbox']],
                    'age': int(face_data['age']) if face_data.get('age') is not None else None,
                    'gender': face_data['gender']
                }
                converted_embeddings.append(converted_face)
            
            return FaceEmbeddingResponse(
                success=True,
                message=f'Successfully processed moment {moment_id}',
                face_count=len(faces),
                embeddings=converted_embeddings,
                quality_scores=[float(face['quality_score']) for face in faces]
            )
            
        except Exception as e:
            logger.error(f"Moment processing failed for {moment_id}: {e}")
            return FaceEmbeddingResponse(
                success=False,
                message=f'Moment processing failed: {str(e)}',
                face_count=0
            )
    
    async def match_moment_faces_with_users(self, moment_face_embeddings: List[Dict[str, Any]], 
                                          user_face_embeddings: List[UserFaceEmbedding],
                                          moment_media_url: Optional[str] = None) -> List[FaceMatch]:
        """
        Match faces in a moment with user face embeddings
        This should be called after processing a moment to find matching users
        """
        try:
            logger.info(f"Matching {len(moment_face_embeddings)} moment faces with {len(user_face_embeddings)} users")
            
            matches = []
            matched_users = set()  # Prevent multiple faces from matching the same user
            
            for face_idx, face_data in enumerate(moment_face_embeddings):
                face_embedding = face_data.get('embedding')
                if not face_embedding:
                    logger.info(f"❌ Face {face_idx}: No embedding found - SKIPPING")
                    continue
                
                # Check face quality
                face_quality = face_data.get('quality_score', 0.0)
                logger.info(f"🔍 Face {face_idx}: quality={face_quality:.3f}, bbox={face_data.get('bbox', 'N/A')}")
                if face_quality < 0.2:  # Lowered quality threshold
                    logger.info(f"❌ Face {face_idx}: SKIPPED - Low quality (quality: {face_quality:.3f} < 0.2)")
                    continue
                
                logger.info(f"✅ Face {face_idx}: PASSED quality check, proceeding to matching...")
                
                best_match = None
                best_similarity = 0.0
                second_best_similarity = 0.0
                
                for user_embedding in user_face_embeddings:
                    user_id = user_embedding.user_id
                    
                    if user_id in matched_users:
                        logger.info(f"Face {face_idx}: Skipping user {user_id} (already matched)")
                        continue
                    
                    # Check cooldown to prevent over-matching (DISABLED FOR TESTING)
                    # current_time = time.time()
                    # if user_id in self.user_match_cooldown:
                    #     last_match_time = self.user_match_cooldown[user_id]
                    #     if current_time - last_match_time < 30:  # 30 second cooldown for testing
                    #         logger.info(f"Skipping user {user_id} due to cooldown")
                    #         continue
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity(face_embedding, user_embedding.embedding)
                    logger.info(f"🔍 Face {face_idx} vs User {user_id}: similarity={similarity:.6f}")
                    
                    # Track top 2 similarities for confidence check
                    if similarity > best_similarity:
                        second_best_similarity = best_similarity
                        best_similarity = similarity
                        best_match = {
                            'user_id': user_id,
                            'similarity': similarity,
                            'face_index': face_idx,
                            'face_quality': face_quality
                        }
                    elif similarity > second_best_similarity:
                        second_best_similarity = similarity
                
                # Apply adaptive matching criteria based on face quality
                confidence_gap = best_similarity - second_best_similarity
                min_confidence_gap = 0.001  # Very low confidence gap for identical scores
                
                # Adaptive thresholds based on face quality
                adaptive_similarity_threshold = self._get_adaptive_similarity_threshold(face_quality)
                adaptive_quality_threshold = self._get_adaptive_quality_threshold(face_quality)
                
                if (best_match and 
                    best_similarity >= adaptive_similarity_threshold and
                    (confidence_gap >= min_confidence_gap or confidence_gap == 0.0) and
                    face_quality >= adaptive_quality_threshold):
                    
                    # Additional confidence validation to reduce false positives
                    confidence_score = self._calculate_confidence_score(
                        best_similarity, 
                        best_match['face_quality'], 
                        confidence_gap,
                        second_best_similarity
                    )
                    
                    # Create face match object
                    face_match = FaceMatch(
                        user_id=best_match['user_id'],
                        similarity=best_match['similarity'],
                        face_index=best_match['face_index'],
                        quality_score=best_match['face_quality'],
                        confidence=confidence_score,
                        moment_media_url=moment_media_url
                    )
                    
                    matches.append(face_match)
                    matched_users.add(best_match['user_id'])
                    
                    # Update cooldown for this user (DISABLED FOR TESTING)
                    # self.user_match_cooldown[best_match['user_id']] = time.time()
                    
                    logger.info(f"✅ Face {face_idx} MATCHED: user {best_match['user_id']} with similarity {best_similarity:.3f} (quality: {best_match['face_quality']:.3f}, gap: {confidence_gap:.3f})")
                else:
                    logger.info(f"❌ Face {face_idx} NO MATCH: best={best_similarity:.3f}, second={second_best_similarity:.3f}, gap={confidence_gap:.3f}, quality={face_quality:.3f}")
                    gap_criteria = confidence_gap >= 0.001 or confidence_gap == 0.0
                    similarity_criteria = best_similarity >= adaptive_similarity_threshold
                    quality_criteria = face_quality >= adaptive_quality_threshold
                    logger.info(f"   Adaptive Criteria: similarity >= {adaptive_similarity_threshold:.2f}: {similarity_criteria}, gap >= 0.001 or gap==0: {gap_criteria}, quality >= {adaptive_quality_threshold:.2f}: {quality_criteria}")
                
             
            
            # Limit total matches to prevent over-tagging
            max_matches = 5
            if len(matches) > max_matches:
                matches.sort(key=lambda x: x.confidence, reverse=True)
                matches = matches[:max_matches]
                logger.info(f"Limited matches to {max_matches} to prevent over-tagging")
            
            logger.info(f"🎯 FINAL RESULTS: Found {len(matches)} face matches out of {len(moment_face_embeddings)} faces")
            for i, match in enumerate(matches):
                logger.info(f"   Match {i+1}: Face {match.face_index} -> User {match.user_id} (similarity: {match.similarity:.3f})")
            return matches
            
        except Exception as e:
            logger.error(f"Face matching failed: {e}")
            return []
    
    async def _find_user_in_existing_moments(self, user_embedding: UserFaceEmbedding, 
                                             firestore_client) -> List[FaceMatch]:
        """
        Find user matches in existing moments
        This searches through all moment_face_embeddings to find where the user appears
        """
        try:
            logger.info(f"Searching for user {user_embedding.user_id} in existing moments")
            
            # Get all moment face embeddings from Firestore
            all_moments = await firestore_client.get_all_moment_face_embeddings()
            
            if not all_moments:
                logger.info("No existing moments found to search")
                return []
            
            matches = []
            
            for moment_embedding in all_moments:
                try:
                    # Match user embedding against all faces in this moment
                    moment_matches = await self.match_moment_faces_with_users(
                        moment_embedding.face_embeddings,
                        [user_embedding],
                        moment_embedding.moment_url
                    )
                    
                    # Filter matches for this specific user
                    user_matches = [match for match in moment_matches 
                                  if match.user_id == user_embedding.user_id]
                    
                    if user_matches:
                        # Add moment_id and media_url to matches for context
                        for match in user_matches:
                            match.moment_id = moment_embedding.moment_id
                            match.moment_media_url = moment_embedding.moment_url
                        
                        matches.extend(user_matches)
                        logger.info(f"Found {len(user_matches)} matches in moment {moment_embedding.moment_id}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process moment {moment_embedding.moment_id}: {e}")
                    continue
            
            # Sort matches by confidence (highest first)
            matches.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit results to prevent overwhelming response
            max_matches = 10
            if len(matches) > max_matches:
                matches = matches[:max_matches]
                logger.info(f"Limited results to {max_matches} matches")
            
            logger.info(f"Found {len(matches)} total matches for user {user_embedding.user_id}")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to search existing moments: {e}")
            return []
    
    async def get_all_moment_face_embeddings(self, firestore_client) -> List[MomentFaceEmbedding]:
        """Get all moment face embeddings from Firestore"""
        try:
            # This method should be implemented in FirestoreClient
            # For now, we'll use a placeholder
            return await firestore_client.get_all_moment_face_embeddings()
        except Exception as e:
            logger.error(f"Failed to get moment face embeddings: {e}")
            return []
    
    async def match_user_against_event_moments(self, user_embedding: UserFaceEmbedding, 
                                             event_id: str, firestore_client) -> List[FaceMatch]:
        """
        Match user face embedding against all moments in an event
        """
        try:
            logger.info(f"Matching user {user_embedding.user_id} against moments in event {event_id}")
            
            # Get all moment face embeddings for the event
            event_moments = await firestore_client.get_moment_face_embeddings_by_event(event_id)
            
            if not event_moments:
                logger.info(f"No moments found in event {event_id}")
                return []
            
            matches = []
            logger.info(f"Event moments count: {len(event_moments)}")
            for moment_embedding in event_moments:
                try:
                    # Match user embedding against all faces in this moment
                    moment_matches = await self.match_moment_faces_with_users(
                        moment_embedding.face_embeddings,
                        [user_embedding],
                        moment_embedding.moment_url
                    )
                    
                    # Filter matches for this specific user
                    user_matches = [match for match in moment_matches 
                                  if match.user_id == user_embedding.user_id]
                    
                    if user_matches:
                        # Add moment_id to matches for context
                        for match in user_matches:
                            match.moment_id = moment_embedding.moment_id
                        
                        matches.extend(user_matches)
                        logger.info(f"Found {len(user_matches)} matches in moment {moment_embedding.moment_id}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process moment {moment_embedding.moment_id}: {e}")
                    continue
            
            # Sort matches by confidence (highest first)
            matches.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Found {len(matches)} total matches for user {user_embedding.user_id} in event {event_id}")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to match user against event moments: {e}")
            return []
    
    async def match_moment_against_event_users(self, moment_embedding: MomentFaceEmbedding,
                                             event_id: str, firestore_client) -> List[FaceMatch]:
        """
        Match moment face embeddings against all user embeddings in an event
        """
        try:
            logger.info(f"Matching moment {moment_embedding.moment_id} against users in event {event_id}")
            
            # Get all user face embeddings for the event
            event_users = await firestore_client.get_user_face_embeddings_by_event(event_id)
            logger.info(f"Event users: {event_users}")
            
            if not event_users:
                logger.info(f"No users found in event {event_id}")
                return []
            
            # Match moment faces with users
            matches = await self.match_moment_faces_with_users(
                moment_embedding.face_embeddings,
                event_users,
                moment_embedding.moment_url
            )
            
            logger.info(f"Found {len(matches)} matches for moment {moment_embedding.moment_id} in event {event_id}")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to match moment against event users: {e}")
            return []
    
    def _convert_numpy_types(self, data):
        """Convert NumPy types to native Python types for serialization"""
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
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            logger.info("InsightFace recognition service cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
