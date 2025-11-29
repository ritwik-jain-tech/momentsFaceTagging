"""
API endpoints for face embedding operations
Handles user_face_embedding and moment_face_embedding collections
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging

from app.models.face_embeddings import (
    FaceEmbeddingResponse, 
    UserFaceEmbedding, 
    MomentFaceEmbedding,
    FaceMatch
)
from app.models.requests import ProcessSelfieRequest, ProcessMomentRequest, ProcessEventMomentsRequest, ProcessMomentsBatchRequest
from app.models.responses import ProcessMomentsBatchResponse, BatchMomentResult
from app.core.face_recognition_insightface import InsightFaceRecognitionService
from app.core.firestore_client import FirestoreClient
from app.core.storage_client import StorageClient
from app.core.image_compression import ImageCompressionService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global services (will be injected by main.py)
face_recognition_service: Optional[InsightFaceRecognitionService] = None
firestore_client: Optional[FirestoreClient] = None
storage_client: Optional[StorageClient] = None
image_compression_service: Optional[ImageCompressionService] = None


@router.post("/selfie/process", response_model=FaceEmbeddingResponse)
async def process_selfie_for_user(
    request: ProcessSelfieRequest,
    background_tasks: BackgroundTasks
):
    """
    Process selfie image to create/update user_face_embedding
    Optionally perform face matching against existing moments in the event
    """
    try:
        if not face_recognition_service:
            raise HTTPException(status_code=500, detail="Face recognition service not available")
        
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        logger.info(f"Processing selfie for user {request.user_id} in event {request.event_id}")
        
        # Step 1: Process selfie and extract face embedding
        result = await face_recognition_service.process_selfie_for_user(
            str(request.image_url), 
            request.user_id
        )
        
        if not result.success:
            return result
        
        # Step 2: Store user face embedding
        if not result.embeddings or len(result.embeddings) == 0:
            raise HTTPException(status_code=400, detail="No face embeddings found in selfie")
            
        user_embedding = UserFaceEmbedding(
            user_id=request.user_id,
            event_id=request.event_id,
            embedding=result.embeddings[0]['embedding'],
            quality_score=result.embeddings[0]['quality_score'],
            detection_score=result.embeddings[0]['detection_score'],
            bbox=result.embeddings[0]['bbox'],
            age=result.embeddings[0].get('age'),
            gender=result.embeddings[0].get('gender'),
            selfie_url=str(request.image_url)
        )
        
        # Store in Firestore
        await firestore_client.store_user_face_embedding(user_embedding)
        logger.info(f"Stored user face embedding for user {request.user_id}")
        
        # Step 3: Perform face matching if requested
        matches = []
        if request.face_matching:
            logger.info(f"Performing face matching for user {request.user_id} in event {request.event_id}")
            matches = await face_recognition_service.match_user_against_event_moments(
                user_embedding, 
                request.event_id,
                firestore_client
            )
            logger.info(f"Found {len(matches)} matches for user {request.user_id}")
            
            # Step 4: Update moments.taggedUserIds with found matches
            if matches:
                logger.info(f"Updating moments.taggedUserIds for {len(matches)} matches")
                updated_moments = []
                failed_moments = []
                
                for match in matches:
                    try:
                        moment_id = match.moment_id
                        if moment_id:
                            # Get current taggedUserIds for this moment
                            current_tagged_users = await firestore_client.get_moment_tagged_users(moment_id)
                            
                            # Add user_id if not already present
                            if request.user_id not in current_tagged_users:
                                updated_tagged_users = current_tagged_users + [request.user_id]
                                
                                # Update the moment in Firestore
                                success = await firestore_client.update_moment_tagged_users(
                                    moment_id, 
                                    updated_tagged_users
                                )
                                
                                if success:
                                    updated_moments.append({
                                        'moment_id': moment_id,
                                        'added_user': request.user_id,
                                        'total_users': len(updated_tagged_users)
                                    })
                                    logger.info(f"Successfully added user {request.user_id} to moment {moment_id}")
                                else:
                                    failed_moments.append({
                                        'moment_id': moment_id,
                                        'error': 'Failed to update moment in Firestore'
                                    })
                            else:
                                logger.info(f"User {request.user_id} already tagged in moment {moment_id}")
                                
                    except Exception as e:
                        logger.error(f"Failed to update moment {match.moment_id}: {e}")
                        failed_moments.append({
                            'moment_id': match.moment_id,
                            'error': str(e)
                        })
                
                logger.info(f"Updated {len(updated_moments)} moments, {len(failed_moments)} failed")
        
        # Update result with matches
        result.matches = matches
        
        return result
        
    except Exception as e:
        logger.error(f"Selfie processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Selfie processing failed: {str(e)}")


@router.post("/moment/process", response_model=FaceEmbeddingResponse)
async def process_moment_for_faces(
    request: ProcessMomentRequest,
    background_tasks: BackgroundTasks
):
    """
    Process moment image to create moment_face_embedding
    Optionally match faces against existing user embeddings in the event
    """
    try:
        if not face_recognition_service:
            raise HTTPException(status_code=500, detail="Face recognition service not available")
        
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        logger.info(f"Processing moment {request.moment_id} in event {request.event_id}")
        
        # Step 1: Process moment and extract face embeddings
        result = await face_recognition_service.process_moment_for_faces(
            str(request.image_url), 
            request.moment_id,
            request.event_id
        )
        
        if not result.success:
            return result
        
        # Step 1.5: Compress and upload images (if services are available)
        if storage_client and image_compression_service:
            try:
                logger.info(f"Starting image compression for moment {request.moment_id}")
                
                # Download original image
                original_image_data = await image_compression_service.download_image_bytes(str(request.image_url))
                
                if original_image_data:
                    # Check if image is JPEG - skip JPEG compression if not JPEG
                    is_jpeg = image_compression_service.is_jpeg_image(original_image_data)
                    
                    if not is_jpeg:
                        logger.info(f"Image is not JPEG format, skipping JPEG compression for moment {request.moment_id}")
                    
                    # Create compressed versions
                    # If JPEG: only compress JPEG (if size reduced), skip WebP
                    # If not JPEG: skip JPEG, create WebP
                    high_quality_jpeg, webp_feed = image_compression_service.compress_both_versions(original_image_data)
                    
                    if is_jpeg:
                        # For JPEG images: only compress JPEG, skip WebP
                        if high_quality_jpeg:
                            # Generate file path for compressed JPEG
                            jpeg_path, _ = image_compression_service.generate_compressed_file_paths(
                                str(request.image_url),
                                moment_id=request.moment_id
                            )
                            
                            logger.info(f"Uploading compressed JPEG for moment {request.moment_id}")
                            high_quality_url = await storage_client.upload_image(
                                high_quality_jpeg,
                                jpeg_path,
                                content_type="image/jpeg"
                            )
                            
                            if not high_quality_url:
                                logger.warning(f"Failed to upload JPEG for moment {request.moment_id}, will use original URL")
                                high_quality_url = str(request.image_url)
                        else:
                            # JPEG compression didn't reduce size, use original
                            high_quality_url = str(request.image_url)
                            logger.info(f"Using original image URL for moment {request.moment_id} (JPEG compression didn't reduce size)")
                        
                        # For JPEG images, don't create WebP - use None for feedUrl
                        webp_feed_url = None
                        
                        # Update moment media URLs in Firestore
                        logger.info(f"Updating moment {request.moment_id} media URLs")
                        logger.info(f"  media.url: {high_quality_url}")
                        logger.info(f"  media.feedUrl: {webp_feed_url} (skipped for JPEG)")
                        update_success = await firestore_client.update_moment_media_urls(
                            request.moment_id,
                            high_quality_url,
                            webp_feed_url
                        )
                        
                        if update_success:
                            logger.info(f"Successfully updated media URLs for moment {request.moment_id}")
                        else:
                            logger.warning(f"Failed to update media URLs in Firestore for moment {request.moment_id}")
                    else:
                        # For non-JPEG images: create WebP, skip JPEG compression
                        if webp_feed:
                            # Generate file path for WebP
                            _, webp_path = image_compression_service.generate_compressed_file_paths(
                                str(request.image_url),
                                moment_id=request.moment_id
                            )
                            
                            logger.info(f"Uploading WebP image for moment {request.moment_id}")
                            webp_feed_url = await storage_client.upload_image(
                                webp_feed,
                                webp_path,
                                content_type="image/webp"
                            )
                            
                            if webp_feed_url:
                                # Use original URL for media.url (no JPEG compression for non-JPEG)
                                high_quality_url = str(request.image_url)
                                
                                # Update moment media URLs in Firestore
                                logger.info(f"Updating moment {request.moment_id} media URLs")
                                logger.info(f"  media.url: {high_quality_url}")
                                logger.info(f"  media.feedUrl: {webp_feed_url}")
                                update_success = await firestore_client.update_moment_media_urls(
                                    request.moment_id,
                                    high_quality_url,
                                    webp_feed_url
                                )
                                
                                if update_success:
                                    logger.info(f"Successfully updated media URLs for moment {request.moment_id}")
                                else:
                                    logger.warning(f"Failed to update media URLs in Firestore for moment {request.moment_id}")
                            else:
                                logger.warning(f"Failed to upload WebP image for moment {request.moment_id}")
                        else:
                            logger.warning(f"Failed to compress WebP for moment {request.moment_id}")
                else:
                    logger.warning(f"Failed to download original image for compression: {request.image_url}")
                    
            except Exception as e:
                logger.error(f"Error during image compression for moment {request.moment_id}: {e}", exc_info=True)
                # Don't fail the entire request if compression fails
        else:
            logger.warning("Image compression services not available, skipping compression")
        
        # Step 2: Store moment face embedding
        moment_embedding = MomentFaceEmbedding(
            moment_id=request.moment_id,
            event_id=request.event_id,
            face_embeddings=result.embeddings or [],  # Use empty list if None
            face_count=result.face_count,
            moment_url=str(request.image_url)
        )
        
        # Store in Firestore
        await firestore_client.store_moment_face_embedding(moment_embedding)
        logger.info(f"Stored moment face embedding for moment {request.moment_id}")
        
        # Step 3: Perform face matching if requested
        matches = []
        if request.match_faces:
            logger.info(f"Performing face matching for moment {request.moment_id} in event {request.event_id}")
            matches = await face_recognition_service.match_moment_against_event_users(
                moment_embedding,
                request.event_id,
                firestore_client
            )
            logger.info(f"Found {len(matches)} matches for moment {request.moment_id}")
            
            # Store matches if any found
            if matches:
                await firestore_client.store_face_matches(request.moment_id, matches)
                logger.info(f"Stored {len(matches)} face matches for moment {request.moment_id}")
                
                # Update moment's taggedUserIds with found matches
                logger.info(f"Updating moment {request.moment_id} taggedUserIds with {len(matches)} matches")
                updated_moments = []
                failed_moments = []
                
                # Get current taggedUserIds for this moment
                current_tagged_users = await firestore_client.get_moment_tagged_users(request.moment_id)
                logger.info(f"Current tagged users for moment {request.moment_id}: {current_tagged_users}")
                
                # Extract unique user IDs from matches
                matched_user_ids = list(set([match.user_id for match in matches]))
                logger.info(f"Matched user IDs: {matched_user_ids}")
                
                # Add new users to taggedUserIds (avoid duplicates)
                updated_tagged_users = current_tagged_users.copy()
                for user_id in matched_user_ids:
                    if user_id not in updated_tagged_users:
                        updated_tagged_users.append(user_id)
                        logger.info(f"Adding user {user_id} to moment {request.moment_id} taggedUserIds")
                
                # Update the moment in Firestore
                if len(updated_tagged_users) != len(current_tagged_users):
                    success = await firestore_client.update_moment_tagged_users(
                        request.moment_id, 
                        updated_tagged_users
                    )
                    
                    if success:
                        updated_moments.append({
                            'moment_id': request.moment_id,
                            'added_users': matched_user_ids,
                            'total_users': len(updated_tagged_users)
                        })
                        logger.info(f"Successfully updated moment {request.moment_id} taggedUserIds")
                    else:
                        failed_moments.append({
                            'moment_id': request.moment_id,
                            'error': 'Failed to update moment in Firestore'
                        })
                        logger.error(f"Failed to update moment {request.moment_id} taggedUserIds")
                else:
                    logger.info(f"No new users to add to moment {request.moment_id} taggedUserIds")
                
                logger.info(f"Updated {len(updated_moments)} moments, {len(failed_moments)} failed")
        
        # Update result with matches
        result.matches = matches
        
        return result
        
    except Exception as e:
        logger.error(f"Moment processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Moment processing failed: {str(e)}")


@router.post("/moments/batch", response_model=ProcessMomentsBatchResponse)
async def process_moments_batch(
    request: ProcessMomentsBatchRequest,
    background_tasks: BackgroundTasks
):
    """
    Process multiple moments in batch for face tagging
    More efficient than processing moments individually
    """
    import time
    start_time = time.time()
    
    try:
        if not face_recognition_service:
            raise HTTPException(status_code=500, detail="Face recognition service not available")
        
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        if not request.moments:
            raise HTTPException(status_code=400, detail="No moments provided for processing")
        
        logger.info(f"Processing batch of {len(request.moments)} moments")
        
        results = []
        successful_count = 0
        failed_count = 0
        
        # Process each moment in the batch
        for moment_request in request.moments:
            try:
                logger.info(f"Processing moment {moment_request.moment_id} in batch")
                
                # Step 1: Process moment and extract face embeddings
                result = await face_recognition_service.process_moment_for_faces(
                    str(moment_request.image_url), 
                    moment_request.moment_id,
                    moment_request.event_id
                )
                
                if not result.success:
                    results.append(BatchMomentResult(
                        moment_id=moment_request.moment_id,
                        success=False,
                        message=result.message,
                        face_count=0,
                        matches_found=0,
                        error=result.message
                    ))
                    failed_count += 1
                    continue
                
                # Step 2: Store moment face embedding
                moment_embedding = MomentFaceEmbedding(
                    moment_id=moment_request.moment_id,
                    event_id=moment_request.event_id,
                    face_embeddings=result.embeddings or [],  # Use empty list if None
                    face_count=result.face_count,
                    moment_url=str(moment_request.image_url)
                )
                
                # Store in Firestore
                await firestore_client.store_moment_face_embedding(moment_embedding)
                logger.info(f"Stored moment face embedding for moment {moment_request.moment_id}")
                
                # Step 2.5: Compress and upload images (if services are available)
                if storage_client and image_compression_service:
                    try:
                        logger.info(f"Starting image compression for moment {moment_request.moment_id}")
                        
                        # Download original image
                        original_image_data = await image_compression_service.download_image_bytes(str(moment_request.image_url))
                        
                        if original_image_data:
                            # Check if image is JPEG - skip JPEG compression if not JPEG
                            is_jpeg = image_compression_service.is_jpeg_image(original_image_data)
                            
                            if not is_jpeg:
                                logger.info(f"Image is not JPEG format, skipping JPEG compression for moment {moment_request.moment_id}")
                            
                            # Create compressed versions
                            # If JPEG: only compress JPEG (if size reduced), skip WebP
                            # If not JPEG: skip JPEG, create WebP
                            high_quality_jpeg, webp_feed = image_compression_service.compress_both_versions(original_image_data)
                            
                            if is_jpeg:
                                # For JPEG images: only compress JPEG, skip WebP
                                if high_quality_jpeg:
                                    jpeg_path, _ = image_compression_service.generate_compressed_file_paths(
                                        str(moment_request.image_url),
                                        moment_id=moment_request.moment_id
                                    )
                                    logger.info(f"Uploading compressed JPEG for moment {moment_request.moment_id}")
                                    high_quality_url = await storage_client.upload_image(
                                        high_quality_jpeg,
                                        jpeg_path,
                                        content_type="image/jpeg"
                                    )
                                    if not high_quality_url:
                                        logger.warning(f"Failed to upload JPEG for moment {moment_request.moment_id}, will use original URL")
                                        high_quality_url = str(moment_request.image_url)
                                else:
                                    high_quality_url = str(moment_request.image_url)
                                    logger.info(f"Using original image URL for moment {moment_request.moment_id} (JPEG compression didn't reduce size)")
                                
                                webp_feed_url = None
                                
                                logger.info(f"Updating moment {moment_request.moment_id} media URLs")
                                logger.info(f"  media.url: {high_quality_url}")
                                logger.info(f"  media.feedUrl: {webp_feed_url} (skipped for JPEG)")
                                update_success = await firestore_client.update_moment_media_urls(
                                    moment_request.moment_id,
                                    high_quality_url,
                                    webp_feed_url
                                )
                                
                                if update_success:
                                    logger.info(f"Successfully updated media URLs for moment {moment_request.moment_id}")
                                else:
                                    logger.warning(f"Failed to update media URLs in Firestore for moment {moment_request.moment_id}")
                            else:
                                # For non-JPEG images: create WebP, skip JPEG compression
                                if webp_feed:
                                    _, webp_path = image_compression_service.generate_compressed_file_paths(
                                        str(moment_request.image_url),
                                        moment_id=moment_request.moment_id
                                    )
                                    logger.info(f"Uploading WebP image for moment {moment_request.moment_id}")
                                    webp_feed_url = await storage_client.upload_image(
                                        webp_feed,
                                        webp_path,
                                        content_type="image/webp"
                                    )
                                    
                                    if webp_feed_url:
                                        high_quality_url = str(moment_request.image_url)
                                        logger.info(f"Updating moment {moment_request.moment_id} media URLs")
                                        logger.info(f"  media.url: {high_quality_url}")
                                        logger.info(f"  media.feedUrl: {webp_feed_url}")
                                        update_success = await firestore_client.update_moment_media_urls(
                                            moment_request.moment_id,
                                            high_quality_url,
                                            webp_feed_url
                                        )
                                        
                                        if update_success:
                                            logger.info(f"Successfully updated media URLs for moment {moment_request.moment_id}")
                                        else:
                                            logger.warning(f"Failed to update media URLs in Firestore for moment {moment_request.moment_id}")
                                    else:
                                        logger.warning(f"Failed to upload WebP image for moment {moment_request.moment_id}")
                                else:
                                    logger.warning(f"Failed to compress WebP for moment {moment_request.moment_id}")
                        else:
                            logger.warning(f"Failed to download original image for compression: {moment_request.image_url}")
                            
                    except Exception as e:
                        logger.error(f"Error during image compression for moment {moment_request.moment_id}: {e}", exc_info=True)
                        # Don't fail the entire request if compression fails
                else:
                    logger.warning("Image compression services not available, skipping compression")
                
                # Step 3: Perform face matching if requested
                matches = []
                if moment_request.match_faces:
                    logger.info(f"Performing face matching for moment {moment_request.moment_id} in event {moment_request.event_id}")
                    matches = await face_recognition_service.match_moment_against_event_users(
                        moment_embedding,
                        moment_request.event_id,
                        firestore_client
                    )
                    logger.info(f"Found {len(matches)} matches for moment {moment_request.moment_id}")
                    
                    # Store matches if any found
                    if matches:
                        await firestore_client.store_face_matches(moment_request.moment_id, matches)
                        logger.info(f"Stored {len(matches)} face matches for moment {moment_request.moment_id}")
                        
                        # Update moment's taggedUserIds with found matches
                        logger.info(f"Updating moment {moment_request.moment_id} taggedUserIds with {len(matches)} matches")
                        
                        # Get current taggedUserIds for this moment
                        current_tagged_users = await firestore_client.get_moment_tagged_users(moment_request.moment_id)
                        logger.info(f"Current tagged users for moment {moment_request.moment_id}: {current_tagged_users}")
                        
                        # Extract unique user IDs from matches
                        matched_user_ids = list(set([match.user_id for match in matches]))
                        logger.info(f"Matched user IDs: {matched_user_ids}")
                        
                        # Add new users to taggedUserIds (avoid duplicates)
                        updated_tagged_users = current_tagged_users.copy()
                        for user_id in matched_user_ids:
                            if user_id not in updated_tagged_users:
                                updated_tagged_users.append(user_id)
                                logger.info(f"Adding user {user_id} to moment {moment_request.moment_id} taggedUserIds")
                        
                        # Update the moment in Firestore
                        if len(updated_tagged_users) != len(current_tagged_users):
                            success = await firestore_client.update_moment_tagged_users(
                                moment_request.moment_id, 
                                updated_tagged_users
                            )
                            
                            if success:
                                logger.info(f"Successfully updated moment {moment_request.moment_id} taggedUserIds")
                            else:
                                logger.warning(f"Failed to update moment {moment_request.moment_id} taggedUserIds")
                
                # Update result with matches
                result.matches = matches
                
                results.append(BatchMomentResult(
                    moment_id=moment_request.moment_id,
                    success=True,
                    message=f"Successfully processed moment with {result.face_count} faces",
                    face_count=result.face_count,
                    matches_found=len(matches),
                    error=None
                ))
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process moment {moment_request.moment_id}: {e}")
                results.append(BatchMomentResult(
                    moment_id=moment_request.moment_id,
                    success=False,
                    message=f"Processing failed: {str(e)}",
                    face_count=0,
                    matches_found=0,
                    error=str(e)
                ))
                failed_count += 1
        
        processing_time = time.time() - start_time
        
        logger.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed, {processing_time:.2f}s")
        
        return ProcessMomentsBatchResponse(
            total_moments=len(request.moments),
            successful_moments=successful_count,
            failed_moments=failed_count,
            results=results,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch moment processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch moment processing failed: {str(e)}")


@router.post("/moment/match", response_model=List[FaceMatch])
async def match_moment_faces_with_users(
    moment_id: str,
    background_tasks: BackgroundTasks
):
    """
    Match faces in a moment with user face embeddings
    This endpoint should be called after processing a moment to find matching users
    """
    try:
        if not face_recognition_service or not firestore_client:
            raise HTTPException(status_code=500, detail="Services not available")
        
        logger.info(f"Matching faces in moment {moment_id} with users")
        
        # Get moment face embeddings from Firestore
        moment_embedding = await firestore_client.get_moment_face_embedding(moment_id)
        if not moment_embedding:
            raise HTTPException(status_code=404, detail=f"Moment {moment_id} not found")
        
        # Get all user face embeddings from Firestore
        user_embeddings = await firestore_client.get_all_user_face_embeddings()
        
        if not user_embeddings:
            logger.info("No user face embeddings found")
            return []
        
        # Match moment faces with users
        matches = await face_recognition_service.match_moment_faces_with_users(
            moment_embedding.face_embeddings,
            user_embeddings,
            moment_embedding.moment_url
        )
        
        # Store matches in Firestore
        if matches:
            await firestore_client.store_face_matches(moment_id, matches)
            logger.info(f"Stored {len(matches)} face matches for moment {moment_id}")
        
        return matches
        
    except Exception as e:
        logger.error(f"Face matching failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face matching failed: {str(e)}")


@router.get("/user/{user_id}/embedding", response_model=UserFaceEmbedding)
async def get_user_face_embedding(user_id: str):
    """Get user face embedding by user_id"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        user_embedding = await firestore_client.get_user_face_embedding(user_id)
        if not user_embedding:
            raise HTTPException(status_code=404, detail=f"User face embedding not found for user {user_id}")
        
        return user_embedding
        
    except Exception as e:
        logger.error(f"Failed to get user face embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user face embedding: {str(e)}")


@router.get("/moment/{moment_id}/embedding", response_model=MomentFaceEmbedding)
async def get_moment_face_embedding(moment_id: str):
    """Get moment face embedding by moment_id"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        moment_embedding = await firestore_client.get_moment_face_embedding(moment_id)
        if not moment_embedding:
            raise HTTPException(status_code=404, detail=f"Moment face embedding not found for moment {moment_id}")
        
        return moment_embedding
        
    except Exception as e:
        logger.error(f"Failed to get moment face embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get moment face embedding: {str(e)}")


@router.get("/moment/{moment_id}/matches", response_model=List[FaceMatch])
async def get_moment_face_matches(moment_id: str):
    """Get face matches for a moment"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        matches = await firestore_client.get_face_matches(moment_id)
        return matches
        
    except Exception as e:
        logger.error(f"Failed to get face matches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get face matches: {str(e)}")


@router.delete("/user/{user_id}/embedding")
async def delete_user_face_embedding(user_id: str):
    """Delete user face embedding"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        await firestore_client.delete_user_face_embedding(user_id)
        return {"message": f"User face embedding deleted for user {user_id}"}
        
    except Exception as e:
        logger.error(f"Failed to delete user face embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user face embedding: {str(e)}")


@router.delete("/moment/{moment_id}/embedding")
async def delete_moment_face_embedding(moment_id: str):
    """Delete moment face embedding"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        await firestore_client.delete_moment_face_embedding(moment_id)
        return {"message": f"Moment face embedding deleted for moment {moment_id}"}
        
    except Exception as e:
        logger.error(f"Failed to delete moment face embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete moment face embedding: {str(e)}")


@router.post("/process-event")
async def process_event_moments(
    request: ProcessEventMomentsRequest,
    background_tasks: BackgroundTasks
):
    """
    Process all eligible moments in an event for face detection and tagging.
    This endpoint gets all moments for the event and processes each one.
    
    Args:
        request: ProcessEventMomentsRequest containing eventId
        background_tasks: FastAPI background tasks
    
    Returns:
        JSONResponse with processing results
    """
    try:
        if not face_recognition_service:
            raise HTTPException(status_code=500, detail="Face recognition service not available")
        
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        logger.info(f"Processing all moments in event {request.event_id}")
        
        # Get all moments for the event
        moments = await firestore_client.get_eligible_moments(request.event_id)
        
        if not moments:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"No moments found for event {request.event_id}",
                    "data": None
                }
            )
        
        processed_moments = []
        failed_moments = []
        logger.info(f"Processing {len(moments)} moments in event {request.event_id}")
        
        for moment in moments:
            try:
                moment_id = moment.get('moment_id')
                if not moment_id:
                    logger.warning(f"Moment missing ID: {moment}")
                    continue
                
                # Get image URL from moment
                image_url = moment.get('image_url')
                
                if not image_url:
                    logger.warning(f"No image URL found for moment {moment_id}")
                    failed_moments.append({
                        'moment_id': moment_id,
                        'error': 'No image URL found'
                    })
                    continue
                
                # Process the moment using the face_embeddings endpoint logic
                logger.info(f"Processing moment {moment_id} in event {request.event_id}")
                
                # Step 1: Process moment and extract face embeddings
                result = await face_recognition_service.process_moment_for_faces(
                    str(image_url), 
                    moment_id,
                    request.event_id
                )
                
                if not result.success:
                    failed_moments.append({
                        'moment_id': moment_id,
                        'error': result.message
                    })
                    continue
                
                # Step 2: Store moment face embedding
                moment_embedding = MomentFaceEmbedding(
                    moment_id=moment_id,
                    event_id=request.event_id,
                    face_embeddings=result.embeddings or [],  # Use empty list if None
                    face_count=result.face_count,
                    moment_url=str(image_url)
                )
                
                # Store in Firestore
                await firestore_client.store_moment_face_embedding(moment_embedding)
                logger.info(f"Stored moment face embedding for moment {moment_id}")
                
                # Step 2.5: Compress and upload images (if services are available)
                if storage_client and image_compression_service:
                    try:
                        logger.info(f"Starting image compression for moment {moment_id}")
                        
                        # Download original image
                        original_image_data = await image_compression_service.download_image_bytes(str(image_url))
                        
                        if original_image_data:
                            # Check if image is JPEG - skip JPEG compression if not JPEG
                            is_jpeg = image_compression_service.is_jpeg_image(original_image_data)
                            
                            if not is_jpeg:
                                logger.info(f"Image is not JPEG format, skipping JPEG compression for moment {moment_id}")
                            
                            # Create compressed versions
                            # If JPEG: only compress JPEG (if size reduced), skip WebP
                            # If not JPEG: skip JPEG, create WebP
                            high_quality_jpeg, webp_feed = image_compression_service.compress_both_versions(original_image_data)
                            
                            if is_jpeg:
                                # For JPEG images: only compress JPEG, skip WebP
                                if high_quality_jpeg:
                                    jpeg_path, _ = image_compression_service.generate_compressed_file_paths(
                                        str(image_url),
                                        moment_id=moment_id
                                    )
                                    logger.info(f"Uploading compressed JPEG for moment {moment_id}")
                                    high_quality_url = await storage_client.upload_image(
                                        high_quality_jpeg,
                                        jpeg_path,
                                        content_type="image/jpeg"
                                    )
                                    if not high_quality_url:
                                        logger.warning(f"Failed to upload JPEG for moment {moment_id}, will use original URL")
                                        high_quality_url = str(image_url)
                                else:
                                    high_quality_url = str(image_url)
                                    logger.info(f"Using original image URL for moment {moment_id} (JPEG compression didn't reduce size)")
                                
                                webp_feed_url = None
                                
                                logger.info(f"Updating moment {moment_id} media URLs")
                                logger.info(f"  media.url: {high_quality_url}")
                                logger.info(f"  media.feedUrl: {webp_feed_url} (skipped for JPEG)")
                                update_success = await firestore_client.update_moment_media_urls(
                                    moment_id,
                                    high_quality_url,
                                    webp_feed_url
                                )
                                
                                if update_success:
                                    logger.info(f"Successfully updated media URLs for moment {moment_id}")
                                else:
                                    logger.warning(f"Failed to update media URLs in Firestore for moment {moment_id}")
                            else:
                                # For non-JPEG images: create WebP, skip JPEG compression
                                if webp_feed:
                                    _, webp_path = image_compression_service.generate_compressed_file_paths(
                                        str(image_url),
                                        moment_id=moment_id
                                    )
                                    logger.info(f"Uploading WebP image for moment {moment_id}")
                                    webp_feed_url = await storage_client.upload_image(
                                        webp_feed,
                                        webp_path,
                                        content_type="image/webp"
                                    )
                                    
                                    if webp_feed_url:
                                        high_quality_url = str(image_url)
                                        logger.info(f"Updating moment {moment_id} media URLs")
                                        logger.info(f"  media.url: {high_quality_url}")
                                        logger.info(f"  media.feedUrl: {webp_feed_url}")
                                        update_success = await firestore_client.update_moment_media_urls(
                                            moment_id,
                                            high_quality_url,
                                            webp_feed_url
                                        )
                                        
                                        if update_success:
                                            logger.info(f"Successfully updated media URLs for moment {moment_id}")
                                        else:
                                            logger.warning(f"Failed to update media URLs in Firestore for moment {moment_id}")
                                    else:
                                        logger.warning(f"Failed to upload WebP image for moment {moment_id}")
                                else:
                                    logger.warning(f"Failed to compress WebP for moment {moment_id}")
                        else:
                            logger.warning(f"Failed to download original image for compression: {image_url}")
                            
                    except Exception as e:
                        logger.error(f"Error during image compression for moment {moment_id}: {e}", exc_info=True)
                        # Don't fail the entire request if compression fails
                else:
                    logger.warning("Image compression services not available, skipping compression")
                
                # Step 3: Perform face matching against existing users in the event
                logger.info(f"Performing face matching for moment {moment_id} in event {request.event_id}")
                matches = await face_recognition_service.match_moment_against_event_users(
                    moment_embedding,
                    request.event_id,
                    firestore_client
                )
                logger.info(f"Found {len(matches)} matches for moment {moment_id}")
                
                # Store matches if any found
                if matches:
                    await firestore_client.store_face_matches(moment_id, matches)
                    logger.info(f"Stored {len(matches)} face matches for moment {moment_id}")
                
                processed_moments.append({
                    'moment_id': moment_id,
                    'face_count': result.face_count,
                    'matches_count': len(matches),
                    'user_id': moment.get('user_id') or moment.get('creatorId') or 'unknown'
                })
                
                logger.info(f"Successfully processed moment {moment_id} with {result.face_count} faces and {len(matches)} matches")
                    
            except Exception as e:
                logger.error(f"Failed to process moment {moment_id}: {e}")
                failed_moments.append({
                    'moment_id': moment_id,
                    'error': str(e)
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Processed {len(processed_moments)} moments in event {request.event_id}",
                "data": {
                    "event_id": request.event_id,
                    "total_moments": len(moments),
                    "processed_moments": len(processed_moments),
                    "failed_moments": len(failed_moments),
                    "processed": processed_moments,
                    "failed": failed_moments
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Event processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Event processing failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        JSONResponse with service status
    """
    try:
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Face Embeddings Service is healthy",
                "data": {
                    "service": "face-embeddings",
                    "status": "healthy",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/event/{event_id}/compress-images")
async def compress_event_moment_images(
    event_id: str,
    background_tasks: BackgroundTasks
):
    """
    Compress all moment images in an event.
    For each moment with media.url:
    - Compresses to high-quality JPEG (80%) and updates media.url (same path, .jpg extension)
    - Creates WebP feed version and updates media.feedUrl (same base path, .webp extension)
    
    Args:
        event_id: The event ID to compress images for
        background_tasks: FastAPI background tasks
    
    Returns:
        JSON response with compression results
    """
    try:
        if not storage_client:
            raise HTTPException(status_code=500, detail="Storage client not available")
        
        if not image_compression_service:
            raise HTTPException(status_code=500, detail="Image compression service not available")
        
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        logger.info(f"Starting image compression for all moments in event {event_id}")
        
        # Get all moments for the event
        moments = await firestore_client.get_all_moments_for_event(event_id)
        
        if not moments:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"No moments found for event {event_id}",
                    "data": {
                        "event_id": event_id,
                        "total_moments": 0,
                        "compressed_moments": 0,
                        "failed_moments": 0,
                        "compressed": [],
                        "failed": []
                    }
                }
            )
        
        compressed_moments = []
        failed_moments = []
        
        # Storage statistics
        total_original_size = 0
        total_jpeg_size = 0
        total_webp_size = 0
        
        logger.info(f"Found {len(moments)} moments in event {event_id}")
        
        for moment in moments:
            moment_id = moment.get('moment_id')
            if not moment_id:
                logger.warning(f"Moment missing ID: {moment}")
                continue
            
            try:
                # Get media URL from moment
                # Check if moment has media.url directly or in nested structure
                image_url = moment.get('image_url')
                
                # If not in moment dict, try to get from Firestore directly
                if not image_url:
                    moment_doc = await firestore_client.get_moment_by_id(moment_id)
                    if moment_doc:
                        media = moment_doc.get('media', {})
                        if isinstance(media, dict):
                            image_url = media.get('url') or media.get('imageUrl')
                
                if not image_url:
                    logger.warning(f"No image URL found for moment {moment_id}")
                    failed_moments.append({
                        'moment_id': moment_id,
                        'error': 'No image URL found in media.url'
                    })
                    continue
                
                logger.info(f"Compressing image for moment {moment_id}: {image_url}")
                
                # Download original image
                original_image_data = await image_compression_service.download_image_bytes(str(image_url))
                
                if not original_image_data:
                    logger.warning(f"Failed to download image for moment {moment_id}")
                    failed_moments.append({
                        'moment_id': moment_id,
                        'error': 'Failed to download original image'
                    })
                    continue
                
                # Check if image is JPEG - skip JPEG compression if not JPEG
                is_jpeg = image_compression_service.is_jpeg_image(original_image_data)
                
                if not is_jpeg:
                    logger.info(f"Image is not JPEG format, skipping JPEG compression for moment {moment_id}")
                
                # Create compressed versions
                # If JPEG: only compress JPEG (if size reduced), skip WebP
                # If not JPEG: skip JPEG, create WebP
                high_quality_jpeg, webp_feed = image_compression_service.compress_both_versions(original_image_data)
                
                if is_jpeg:
                    # For JPEG images: only compress JPEG, skip WebP
                    if high_quality_jpeg:
                        jpeg_path, _ = image_compression_service.generate_compressed_file_paths(
                            str(image_url),
                            moment_id=moment_id
                        )
                        logger.info(f"Uploading compressed JPEG for moment {moment_id}")
                        high_quality_url = await storage_client.upload_image(
                            high_quality_jpeg,
                            jpeg_path,
                            content_type="image/jpeg"
                        )
                        if not high_quality_url:
                            logger.warning(f"Failed to upload JPEG for moment {moment_id}, will use original URL")
                            high_quality_url = str(image_url)
                    else:
                        high_quality_url = str(image_url)
                        logger.info(f"Using original image URL for moment {moment_id} (JPEG compression didn't reduce size)")
                    
                    webp_feed_url = None
                else:
                    # For non-JPEG images: create WebP, skip JPEG compression
                    if not webp_feed:
                        logger.warning(f"Failed to compress WebP for moment {moment_id}")
                        failed_moments.append({
                            'moment_id': moment_id,
                            'error': 'Failed to compress WebP image',
                            'original_size': len(original_image_data) if original_image_data else 0
                        })
                        continue
                    
                    _, webp_path = image_compression_service.generate_compressed_file_paths(
                        str(image_url),
                        moment_id=moment_id
                    )
                    
                    logger.info(f"Uploading WebP image for moment {moment_id}")
                    webp_feed_url = await storage_client.upload_image(
                        webp_feed,
                        webp_path,
                        content_type="image/webp"
                    )
                    
                    if not webp_feed_url:
                        logger.warning(f"Failed to upload WebP image for moment {moment_id}")
                        failed_moments.append({
                            'moment_id': moment_id,
                            'error': 'Failed to upload WebP image'
                        })
                        continue
                    
                    high_quality_url = str(image_url)
                
                # Update moment media URLs in Firestore
                logger.info(f"Updating media URLs for moment {moment_id}")
                logger.info(f"  media.url: {high_quality_url}")
                logger.info(f"  media.feedUrl: {webp_feed_url}")
                update_success = await firestore_client.update_moment_media_urls(
                    moment_id,
                    high_quality_url,
                    webp_feed_url
                )
                
                if update_success:
                    original_size = len(original_image_data)
                    jpeg_size = len(high_quality_jpeg) if high_quality_jpeg else original_size
                    webp_size = len(webp_feed)
                    
                    # Update storage statistics
                    total_original_size += original_size
                    if high_quality_jpeg:
                        total_jpeg_size += jpeg_size
                    total_webp_size += webp_size
                    
                    compressed_moments.append({
                        'moment_id': moment_id,
                        'jpeg_url': high_quality_url,
                        'webp_url': webp_feed_url,
                        'original_size': original_size,
                        'jpeg_size': jpeg_size if high_quality_jpeg else None,
                        'webp_size': webp_size,
                        'jpeg_compressed': high_quality_jpeg is not None
                    })
                    logger.info(f"Successfully compressed and updated moment {moment_id} - "
                               f"Original: {original_size} bytes, "
                               f"JPEG: {jpeg_size} bytes ({'compressed' if high_quality_jpeg else 'original'}), "
                               f"WebP: {webp_size} bytes")
                else:
                    logger.warning(f"Failed to update Firestore for moment {moment_id}")
                    failed_moments.append({
                        'moment_id': moment_id,
                        'error': 'Failed to update Firestore'
                    })
                    
            except Exception as e:
                logger.error(f"Error compressing image for moment {moment_id}: {e}")
                failed_moments.append({
                    'moment_id': moment_id,
                    'error': str(e)
                })
        
        # Calculate storage savings (only count JPEG savings if compression was actually done)
        # Note: total_jpeg_size includes original sizes for moments where JPEG wasn't compressed
        jpeg_savings = total_original_size - total_jpeg_size
        webp_savings = total_original_size - total_webp_size
        total_savings = total_original_size - total_webp_size  # Using WebP as final storage
        
        logger.info(f"Image compression completed for event {event_id}: "
                   f"{len(compressed_moments)} successful, {len(failed_moments)} failed")
        logger.info(f"Storage statistics - Original: {total_original_size:,} bytes, "
                   f"JPEG: {total_jpeg_size:,} bytes, WebP: {total_webp_size:,} bytes")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Compressed {len(compressed_moments)} moments in event {event_id}",
                "data": {
                    "event_id": event_id,
                    "total_moments": len(moments),
                    "compressed_moments": len(compressed_moments),
                    "failed_moments": len(failed_moments),
                    "storage_statistics": {
                        "total_original_storage_bytes": total_original_size,
                        "total_original_storage_mb": round(total_original_size / (1024 * 1024), 2),
                        "total_jpeg_storage_bytes": total_jpeg_size,
                        "total_jpeg_storage_mb": round(total_jpeg_size / (1024 * 1024), 2),
                        "total_webp_storage_bytes": total_webp_size,
                        "total_webp_storage_mb": round(total_webp_size / (1024 * 1024), 2),
                        "jpeg_storage_savings_bytes": jpeg_savings,
                        "jpeg_storage_savings_mb": round(jpeg_savings / (1024 * 1024), 2),
                        "webp_storage_savings_bytes": webp_savings,
                        "webp_storage_savings_mb": round(webp_savings / (1024 * 1024), 2),
                        "total_storage_savings_bytes": total_savings,
                        "total_storage_savings_mb": round(total_savings / (1024 * 1024), 2)
                    },
                    "compressed": compressed_moments,
                    "failed": failed_moments
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Event image compression failed: {e}")
        raise HTTPException(status_code=500, detail=f"Event image compression failed: {str(e)}")


@router.delete("/event/{event_id}/user/{user_id}/tags")
async def remove_user_tags_from_event(
    event_id: str,
    user_id: str
):
    """
    Remove a specific userId from taggedUserIds in all moments of a given event.
    
    Args:
        event_id: The event ID to remove tags from
        user_id: The user ID to remove from taggedUserIds
        
    Returns:
        JSON response with removal results
    """
    try:
        if not firestore_client:
            raise HTTPException(status_code=500, detail="Firestore client not available")
        
        logger.info(f"Removing user {user_id} tags from all moments in event {event_id}")
        
        # Get all moments for the event (not just approved ones for user removal)
        moments = await firestore_client.get_all_moments_for_event(event_id)
        
        if not moments:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"No moments found for event {event_id}",
                    "data": None
                }
            )
        
        updated_moments = []
        failed_moments = []
        
        for moment in moments:
            try:
                moment_id = moment.get('moment_id')
                if not moment_id:
                    logger.warning(f"Moment missing ID: {moment}")
                    continue
                
                # Get current taggedUserIds directly from Firestore
                current_tagged_users = await firestore_client.get_moment_tagged_users(moment_id)
                
                if user_id not in current_tagged_users:
                    logger.info(f"User {user_id} not found in moment {moment_id} taggedUserIds")
                    continue
                
                # Remove user_id from taggedUserIds
                updated_tagged_users = [uid for uid in current_tagged_users if uid != user_id]
                
                # Update the moment in Firestore
                success = await firestore_client.update_moment_tagged_users(
                    moment_id, 
                    updated_tagged_users
                )
                
                if success:
                    updated_moments.append({
                        'moment_id': moment_id,
                        'removed_user': user_id,
                        'remaining_users': updated_tagged_users,
                        'users_count': len(updated_tagged_users)
                    })
                    logger.info(f"Successfully removed user {user_id} from moment {moment_id}")
                else:
                    failed_moments.append({
                        'moment_id': moment_id,
                        'error': 'Failed to update moment in Firestore'
                    })
                    
            except Exception as e:
                logger.error(f"Failed to remove user {user_id} from moment {moment_id}: {e}")
                failed_moments.append({
                    'moment_id': moment_id,
                    'error': str(e)
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Removed user {user_id} from {len(updated_moments)} moments in event {event_id}",
                "data": {
                    "event_id": event_id,
                    "user_id": user_id,
                    "total_moments": len(moments),
                    "updated_moments": len(updated_moments),
                    "failed_moments": len(failed_moments),
                    "updated": updated_moments,
                    "failed": failed_moments
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Remove user tags failed: {e}")
        raise HTTPException(status_code=500, detail=f"Remove user tags failed: {str(e)}")
