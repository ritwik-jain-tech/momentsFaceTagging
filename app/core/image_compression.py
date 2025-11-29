"""
Image compression utilities for Moments Face Tagging Service
Handles image compression and format conversion for optimized storage and delivery
"""

import logging
from typing import Tuple, Optional
from PIL import Image
from io import BytesIO
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ImageCompressionService:
    """
    Service for compressing and converting images
    """
    
    def __init__(self):
        # High quality JPEG settings (80% quality)
        self.HIGH_QUALITY_JPEG_QUALITY = 80
        
        # WebP settings for feed version
        self.WEBP_QUALITY = 30  # Fixed 30% quality
        
        # Maximum dimensions for feed image (reduced for better compression)
        self.FEED_MAX_WIDTH = 1600  # Reduced from 1920
        self.FEED_MAX_HEIGHT = 1600  # Reduced from 1920
        
        logger.info("Image Compression Service initialized")
    
    async def download_image_bytes(self, image_url: str) -> Optional[bytes]:
        """
        Download image from URL and return as bytes
        """
        try:
            logger.info(f"Downloading image from: {image_url}")
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
            }
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            image_data = response.content
            logger.info(f"Downloaded image: {len(image_data)} bytes")
            return image_data
        except Exception as e:
            logger.error(f"Failed to download image {image_url}: {e}")
            return None
    
    def compress_to_high_quality_jpeg(self, image_data: bytes, 
                                      quality: int = None) -> Optional[bytes]:
        """
        Compress image to high-quality JPEG (80% quality by default)
        Returns compressed image as bytes
        Ensures the compressed size is smaller than original
        
        Args:
            image_data: Original image bytes
            quality: JPEG quality (0-100), defaults to 80
        
        Returns:
            Compressed JPEG image bytes or None if failed
        """
        try:
            if quality is None:
                quality = self.HIGH_QUALITY_JPEG_QUALITY
            
            original_size = len(image_data)
            logger.info(f"Compressing image to high-quality JPEG (quality: {quality}, original: {original_size} bytes)")
            
            # Open image from bytes
            try:
                image = Image.open(BytesIO(image_data))
                # Load the image to ensure it's fully decoded
                image.load()
            except Exception as load_error:
                logger.error(f"Failed to open/load image: {load_error}", exc_info=True)
                return None
            
            # Convert to RGB if necessary (removes alpha channel)
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Compress to JPEG - only return if size is reduced
            compressed_data = None
            current_quality = quality
            
            # Try compressing, reducing quality if size doesn't reduce
            for attempt in range(3):
                output = BytesIO()
                image.save(output, format='JPEG', quality=current_quality, optimize=True)
                candidate_data = output.getvalue()
                
                # Only use if size is reduced
                if len(candidate_data) < original_size:
                    compressed_data = candidate_data
                    break
                else:
                    # Reduce quality if compression didn't reduce size
                    current_quality = max(50, current_quality - 10)
                    logger.warning(f"JPEG compression didn't reduce size, trying quality {current_quality}")
            
            if not compressed_data:
                logger.warning(f"JPEG compression didn't reduce size. Original: {original_size}, skipping JPEG compression.")
                return None
            
            compressed_size = len(compressed_data)
            compression_ratio = (1 - compressed_size / original_size) * 100
            size_reduction = original_size / compressed_size if compressed_size > 0 else 1
            
            logger.info(f"High-quality JPEG compression: {original_size} -> {compressed_size} bytes "
                       f"({compression_ratio:.1f}% reduction, {size_reduction:.2f}x smaller)")
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Failed to compress image to high-quality JPEG: {e}", exc_info=True)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def compress_to_webp_feed(self, image_data: bytes, 
                              quality: int = None,
                              max_width: int = None,
                              max_height: int = None) -> Optional[bytes]:
        """
        Compress image to WebP format at 30% quality
        Returns compressed WebP image as bytes
        
        Args:
            image_data: Original image bytes
            quality: WebP quality (0-100), defaults to 30
            max_width: Maximum width for feed image
            max_height: Maximum height for feed image
        
        Returns:
            Compressed WebP image bytes or None if failed
        """
        try:
            if quality is None:
                quality = self.WEBP_QUALITY
            if max_width is None:
                max_width = self.FEED_MAX_WIDTH
            if max_height is None:
                max_height = self.FEED_MAX_HEIGHT
            
            original_size = len(image_data)
            logger.info(f"Compressing image to WebP feed format at {quality}% quality (original: {original_size} bytes)")
            
            # Open image from bytes
            try:
                image = Image.open(BytesIO(image_data))
                # Load the image to ensure it's fully decoded
                image.load()
            except Exception as load_error:
                logger.error(f"Failed to open/load image: {load_error}", exc_info=True)
                return None
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if image is too large (maintain aspect ratio)
            original_width, original_height = image.size
            if original_width > max_width or original_height > max_height:
                ratio = min(max_width / original_width, max_height / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
            
            # Compress to WebP at fixed quality
            output = BytesIO()
            image.save(output, format='WebP', quality=quality, method=6)  # method=6 for best compression
            compressed_data = output.getvalue()
            
            if not compressed_data:
                logger.error("Failed to compress image to WebP - no data generated")
                return None
            
            compressed_size = len(compressed_data)
            logger.info(f"WebP feed compression: {original_size} -> {compressed_size} bytes at {quality}% quality")
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Failed to compress image to WebP: {e}", exc_info=True)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def is_jpeg_image(self, image_data: bytes) -> bool:
        """
        Check if image data is JPEG format
        
        Args:
            image_data: Image bytes
        
        Returns:
            True if JPEG, False otherwise
        """
        try:
            test_image = Image.open(BytesIO(image_data))
            image_format = test_image.format
            test_image.close()
            # JPEG formats: JPEG, JPG
            is_jpeg = image_format in ('JPEG', 'JPG')
            logger.info(f"Image format detected: {image_format}, is JPEG: {is_jpeg}")
            return is_jpeg
        except Exception as e:
            logger.error(f"Failed to detect image format: {e}")
            return False
    
    def compress_both_versions(self, image_data: bytes) -> Tuple[Optional[bytes], Optional[bytes]]:
        """
        Create compressed versions based on original format:
        - If JPEG: only compress JPEG (if size is reduced), skip WebP
        - If not JPEG: skip JPEG compression, create WebP
        
        Args:
            image_data: Original image bytes
        
        Returns:
            Tuple of (high_quality_jpeg_bytes, webp_feed_bytes)
        """
        try:
            if not image_data:
                logger.error("Image data is empty or None")
                return None, None
            
            if len(image_data) == 0:
                logger.error("Image data has zero length")
                return None, None
            
            logger.info(f"Creating compressed versions of image (size: {len(image_data)} bytes)")
            
            # Validate that we can open the image and check format
            try:
                test_image = Image.open(BytesIO(image_data))
                image_format = test_image.format
                image_size = test_image.size
                image_mode = test_image.mode
                test_image.close()
                logger.info(f"Image validated: format={image_format}, size={image_size}, mode={image_mode}")
            except Exception as img_error:
                logger.error(f"Invalid image data: {img_error}", exc_info=True)
                return None, None
            
            # Check if original is JPEG
            is_jpeg = image_format in ('JPEG', 'JPG')
            
            high_quality_jpeg = None
            webp_feed = None
            
            if is_jpeg:
                # If original is JPEG: only compress JPEG (if size is reduced), skip WebP
                logger.info("Original image is JPEG, attempting JPEG compression (skipping WebP conversion)")
                high_quality_jpeg = self.compress_to_high_quality_jpeg(image_data)
                if not high_quality_jpeg:
                    logger.warning("JPEG compression didn't reduce size or failed")
            else:
                # If original is not JPEG: skip JPEG compression, create WebP
                logger.info(f"Original image is {image_format}, skipping JPEG compression, creating WebP version")
                webp_feed = self.compress_to_webp_feed(image_data)
                if not webp_feed:
                    logger.error("Failed to compress to WebP feed")
            
            return high_quality_jpeg, webp_feed
            
        except Exception as e:
            logger.error(f"Failed to create compressed versions: {e}", exc_info=True)
            return None, None
    
    def get_file_extension_from_url(self, url: str) -> str:
        """
        Extract file extension from URL, defaulting to 'jpg'
        """
        try:
            # Remove query parameters
            url_path = url.split('?')[0]
            # Get extension
            if '.' in url_path:
                ext = url_path.rsplit('.', 1)[1].lower()
                # Validate extension
                if ext in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                    return ext
            return 'jpg'
        except Exception:
            return 'jpg'
    
    def generate_compressed_file_paths(self, original_url: str, moment_id: str = None) -> Tuple[str, str]:
        """
        Generate file paths for compressed versions
        Returns (high_quality_jpeg_path, webp_feed_path)
        
        Both versions keep the same base folder path as original, only extension changes:
        - JPEG: .jpg extension
        - WebP: .webp extension
        
        Args:
            original_url: Original image URL
            moment_id: Optional moment ID to use in path if URL parsing fails
        
        Returns:
            Tuple of (jpeg_path, webp_path) where both use same base folder path
        """
        try:
            # Parse the original URL to get the filename
            parsed = urlparse(original_url)
            path = parsed.path.lstrip('/')
            
            # Handle CDN URLs (images.moments.live) - extract just the filename
            if 'images.moments.live' in original_url or 'moments.live' in original_url:
                # Path is just the filename, e.g., "1762084552317_6.jpeg"
                filename = path.split('/')[-1] if '/' in path else path
            # Handle GCS URLs - remove bucket name if present
            elif 'storage.googleapis.com' in original_url:
                # Path format: bucket-name/path/to/file.jpg
                # We want just: path/to/file.jpg
                parts = path.split('/', 1)
                if len(parts) > 1:
                    path = parts[1]
                filename = path.split('/')[-1] if '/' in path else path
            else:
                # Extract filename from path
                filename = path.split('/')[-1] if '/' in path else path
            
            # Extract base filename without extension
            if '.' in filename:
                base_filename = filename.rsplit('.', 1)[0]
            else:
                base_filename = filename
            
            # If we couldn't extract a filename, use moment_id
            if not base_filename or base_filename == '/':
                if moment_id:
                    base_filename = moment_id
                else:
                    base_filename = "image"
            
            # Generate paths with same filename, different extensions
            jpeg_path = f"{base_filename}.jpg"
            webp_path = f"{base_filename}.webp"
            
            logger.info(f"Generated paths - JPEG: {jpeg_path}, WebP feed: {webp_path}")
            
            return jpeg_path, webp_path
            
        except Exception as e:
            logger.error(f"Failed to generate file paths: {e}")
            # Fallback to moment_id-based filename or default
            if moment_id:
                base_filename = moment_id
            else:
                base_filename = "image"
            jpeg_path = f"{base_filename}.jpg"
            webp_path = f"{base_filename}.webp"
            return jpeg_path, webp_path

