"""
Face Recognition Service for Driver Verification

This module provides face detection, facial recognition, and liveness detection
capabilities using a combination of computer vision techniques and ML models.
It supports both cloud-based (Hugging Face, AWS Rekognition) and local models.

@version 1.0.0
@author EasyGo Team
"""

import os
import json
import base64
import logging
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import hashlib
import requests
from PIL import Image
import io

# Configure logging
logger = logging.getLogger(__name__)

# Constants
FACE_CONFIDENCE_THRESHOLD = 0.85
LIVENESS_THRESHOLD = 0.70
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_FACE_SIZE = 200  # pixels
SUPPORTED_FORMATS = ['jpeg', 'jpg', 'png', 'webp']

@dataclass
class FaceDetectionResult:
    """Face detection result container"""
    has_face: bool
    face_count: int
    confidence: float
    bounding_boxes: List[Dict[str, float]]
    landmarks: List[Dict[str, float]]
    quality_score: float
    processing_time_ms: float

@dataclass
class FaceComparisonResult:
    """Face comparison result container"""
    is_match: bool
    confidence_score: float
    similarity_score: float
    source_face_quality: float
    target_face_quality: float
    processing_time_ms: float
    correlation_id: str

@dataclass
class LivenessDetectionResult:
    """Liveness detection result container"""
    is_live: bool
    confidence_score: float
    liveness_score: float
    spoof_score: float
    method: str
    processing_time_ms: float
    correlation_id: str


class FaceRecognitionService:
    """
    Main service for face recognition operations including detection,
    comparison, and liveness detection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the face recognition service.
        
        Args:
            config: Configuration dictionary with service settings
        """
        self.config = config or {}
        self._initialize_models()
        self._initialize_services()
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        logger.info("FaceRecognitionService initialized successfully")

    def _initialize_services(self):
        """Initialize optional service state without requiring cloud credentials."""
        self.hf_api_token = getattr(self, 'hf_api_token', os.getenv('HUGGINGFACE_TOKEN'))
        self.hf_api_url = getattr(self, 'hf_api_url', 'https://api-inference.huggingface.co/models')
        self.rekognition = getattr(self, 'rekognition', None)
        self.s3_client = getattr(self, 's3_client', None)
        self.vision_client = getattr(self, 'vision_client', None)
    
    def _initialize_models(self):
        """Initialize AI models for face recognition"""
        self.models = {}
        self.model_type = self.config.get('model_type', 'hybrid')  # hybrid, cloud, local
        
        try:
            # Try to load local models first
            if self.model_type in ['hybrid', 'local']:
                self._load_local_models()
            
            # Initialize cloud service clients
            if self.model_type in ['hybrid', 'cloud']:
                self._initialize_cloud_services()
                
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            self.models = {}
    
    def _load_local_models(self):
        """Load local face recognition models"""
        try:
            # Import local model libraries if available
            import face_recognition
            import dlib
            
            self.models['face_recognition'] = face_recognition
            self.models['dlib'] = dlib
            self.models['local_available'] = True
            
            logger.info("Local face recognition models loaded")
        except ImportError as e:
            logger.warning(f"Local models not available: {e}")
            self.models['local_available'] = False
    
    def _initialize_cloud_services(self):
        """Initialize cloud AI services"""
        cloud_config = self.config.get('cloud_services', {})
        
        # Hugging Face API
        if cloud_config.get('huggingface_enabled', True):
            self.hf_api_token = os.getenv('HUGGINGFACE_TOKEN')
            self.hf_api_url = 'https://api-inference.huggingface.co/models'
            
            # Default models
            self.hf_models = {
                'face_detection': 'huggingface/face-detection',
                'face_embedding': 'sentence-transformers/all-MiniLM-L6-v2',
                'ocr': 'microsoft/trocr-large-printed',
            }
            self.hf_models.update(cloud_config.get('hf_models', {}))
        
        # AWS Rekognition
        if cloud_config.get('aws_enabled', False):
            import boto3
            self.rekognition = boto3.client(
                'rekognition',
                region_name=cloud_config.get('aws_region', 'us-east-1'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            )
            self.s3_client = boto3.client('s3')
        
        # Google Cloud Vision
        if cloud_config.get('google_enabled', False):
            from google.cloud import vision
            self.vision_client = vision.ImageAnnotatorClient()
    
    def detect_faces(self, image_data: bytes, correlation_id: str = None) -> FaceDetectionResult:
        """
        Detect faces in an image.
        
        Args:
            image_data: Image data as bytes
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            FaceDetectionResult: Detection results
        """
        start_time = time.time()
        correlation_id = correlation_id or hashlib.md5(image_data).hexdigest()[:8]
        
        logger.info(f"Starting face detection", {'correlation_id': correlation_id})
        
        try:
            # Validate image
            self._validate_image(image_data)
            
            # Try local detection first if available
            if self.models.get('local_available', False):
                result = self._detect_faces_local(image_data, correlation_id)
                if result.has_face:
                    result.processing_time_ms = (time.time() - start_time) * 1000
                    return result
            
            # Fallback to cloud API
            result = self._detect_faces_cloud(image_data, correlation_id)
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Cache result
            self._cache_result(correlation_id, 'detection', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}", {'correlation_id': correlation_id})
            return FaceDetectionResult(
                has_face=False,
                face_count=0,
                confidence=0.0,
                bounding_boxes=[],
                landmarks=[],
                quality_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _detect_faces_local(self, image_data: bytes, correlation_id: str) -> FaceDetectionResult:
        """Detect faces using local libraries"""
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            
            # Convert to RGB if needed
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
            # Use face_recognition library
            face_locations = face_recognition.face_locations(image_np)
            face_encodings = face_recognition.face_encodings(image_np, face_locations)
            
            if not face_locations:
                return FaceDetectionResult(
                    has_face=False,
                    face_count=0,
                    confidence=0.0,
                    bounding_boxes=[],
                    landmarks=[],
                    quality_score=0.0,
                    processing_time_ms=0
                )
            
            # Get primary face
            primary_face = face_locations[0]
            primary_encoding = face_encodings[0] if face_encodings else None
            
            # Calculate bounding boxes
            bounding_boxes = []
            landmarks = []
            
            for top, right, bottom, left in face_locations:
                bounding_boxes.append({
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'left': left,
                    'width': right - left,
                    'height': bottom - top
                })
            
            # Calculate quality score
            quality_score = self._calculate_face_quality(image_np, primary_face)
            
            return FaceDetectionResult(
                has_face=True,
                face_count=len(face_locations),
                confidence=0.95,  # Local detection is usually high confidence
                bounding_boxes=bounding_boxes,
                landmarks=landmarks,
                quality_score=quality_score,
                processing_time_ms=0
            )
            
        except Exception as e:
            logger.error(f"Local face detection failed: {e}")
            return FaceDetectionResult(
                has_face=False,
                face_count=0,
                confidence=0.0,
                bounding_boxes=[],
                landmarks=[],
                quality_score=0.0,
                processing_time_ms=0
            )
    
    def _detect_faces_cloud(self, image_data: bytes, correlation_id: str) -> FaceDetectionResult:
        """Detect faces using cloud API (Hugging Face)"""
        try:
            # Use Hugging Face face detection
            if self.hf_api_token:
                response = self._call_huggingface_api(
                    self.hf_models['face_detection'],
                    image_data,
                    correlation_id
                )
                
                if response and len(response) > 0:
                    # Parse response
                    faces = response[0] if isinstance(response, list) else response
                    
                    bounding_boxes = []
                    if 'bounding_box' in faces:
                        bbox = faces['bounding_box']
                        bounding_boxes.append({
                            'x': bbox.get('xmin', 0),
                            'y': bbox.get('ymin', 0),
                            'width': bbox.get('xmax', 0) - bbox.get('xmin', 0),
                            'height': bbox.get('ymax', 0) - bbox.get('ymin', 0)
                        })
                    
                    return FaceDetectionResult(
                        has_face=True,
                        face_count=1,
                        confidence=faces.get('score', 0.85),
                        bounding_boxes=bounding_boxes,
                        landmarks=[],
                        quality_score=0.8,
                        processing_time_ms=0
                    )
            
            return FaceDetectionResult(
                has_face=False,
                face_count=0,
                confidence=0.0,
                bounding_boxes=[],
                landmarks=[],
                quality_score=0.0,
                processing_time_ms=0
            )
            
        except Exception as e:
            logger.error(f"Cloud face detection failed: {e}")
            return FaceDetectionResult(
                has_face=False,
                face_count=0,
                confidence=0.0,
                bounding_boxes=[],
                landmarks=[],
                quality_score=0.0,
                processing_time_ms=0
            )
    
    def compare_faces(self, source_image: bytes, target_image: bytes, 
                      correlation_id: str = None) -> FaceComparisonResult:
        """
        Compare two faces and determine if they match.
        
        Args:
            source_image: Source image bytes
            target_image: Target image bytes
            correlation_id: Optional correlation ID
            
        Returns:
            FaceComparisonResult: Comparison results
        """
        start_time = time.time()
        correlation_id = correlation_id or hashlib.md5(source_image + target_image).hexdigest()[:8]
        
        logger.info(f"Starting face comparison", {'correlation_id': correlation_id})
        
        try:
            # Validate images
            self._validate_image(source_image)
            self._validate_image(target_image)
            
            # Check cache
            cache_key = hashlib.md5(source_image + target_image).hexdigest()
            cached = self._get_cached_result(cache_key)
            if cached:
                cached.processing_time_ms = (time.time() - start_time) * 1000
                return cached
            
            # Try local comparison first
            if self.models.get('local_available', False):
                result = self._compare_faces_local(source_image, target_image, correlation_id)
                if result.is_match is not None:
                    result.processing_time_ms = (time.time() - start_time) * 1000
                    self._cache_result(cache_key, 'comparison', result)
                    return result
            
            # Fallback to cloud API
            result = self._compare_faces_cloud(source_image, target_image, correlation_id)
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Cache result
            self._cache_result(cache_key, 'comparison', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}", {'correlation_id': correlation_id})
            return FaceComparisonResult(
                is_match=False,
                confidence_score=0.0,
                similarity_score=0.0,
                source_face_quality=0.0,
                target_face_quality=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                correlation_id=correlation_id
            )
    
    def _compare_faces_local(self, source_image: bytes, target_image: bytes, 
                            correlation_id: str) -> FaceComparisonResult:
        """Compare faces using local libraries"""
        try:
            # Convert images
            source = face_recognition.load_image_file(io.BytesIO(source_image))
            target = face_recognition.load_image_file(io.BytesIO(target_image))
            
            # Get face encodings
            source_encodings = face_recognition.face_encodings(source)
            target_encodings = face_recognition.face_encodings(target)
            
            if not source_encodings or not target_encodings:
                return FaceComparisonResult(
                    is_match=False,
                    confidence_score=0.0,
                    similarity_score=0.0,
                    source_face_quality=0.0,
                    target_face_quality=0.0,
                    processing_time_ms=0,
                    correlation_id=correlation_id
                )
            
            # Compare
            source_encoding = source_encodings[0]
            target_encoding = target_encodings[0]
            
            # Calculate distance
            distance = np.linalg.norm(source_encoding - target_encoding)
            similarity = 1 - (distance / 2)  # Normalize to 0-1
            confidence = similarity * 100
            
            is_match = similarity >= FACE_CONFIDENCE_THRESHOLD
            
            # Calculate face quality
            source_quality = self._calculate_face_quality(
                cv2.cvtColor(source, cv2.COLOR_RGB2BGR), 
                face_recognition.face_locations(source)[0]
            )
            target_quality = self._calculate_face_quality(
                cv2.cvtColor(target, cv2.COLOR_RGB2BGR),
                face_recognition.face_locations(target)[0]
            )
            
            return FaceComparisonResult(
                is_match=is_match,
                confidence_score=confidence,
                similarity_score=similarity,
                source_face_quality=source_quality,
                target_face_quality=target_quality,
                processing_time_ms=0,
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(f"Local face comparison failed: {e}")
            return FaceComparisonResult(
                is_match=None,
                confidence_score=0.0,
                similarity_score=0.0,
                source_face_quality=0.0,
                target_face_quality=0.0,
                processing_time_ms=0,
                correlation_id=correlation_id
            )
    
    def _compare_faces_cloud(self, source_image: bytes, target_image: bytes,
                            correlation_id: str) -> FaceComparisonResult:
        """Compare faces using cloud API"""
        try:
            # Get face embeddings
            source_embedding = self._get_face_embedding(source_image, correlation_id)
            target_embedding = self._get_face_embedding(target_image, correlation_id)
            
            if source_embedding is None or target_embedding is None:
                return FaceComparisonResult(
                    is_match=False,
                    confidence_score=0.0,
                    similarity_score=0.0,
                    source_face_quality=0.0,
                    target_face_quality=0.0,
                    processing_time_ms=0,
                    correlation_id=correlation_id
                )
            
            # Calculate cosine similarity
            similarity = np.dot(source_embedding, target_embedding) / (
                np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding)
            )
            confidence = similarity * 100
            
            is_match = similarity >= FACE_CONFIDENCE_THRESHOLD
            
            return FaceComparisonResult(
                is_match=is_match,
                confidence_score=confidence,
                similarity_score=similarity,
                source_face_quality=0.8,
                target_face_quality=0.8,
                processing_time_ms=0,
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(f"Cloud face comparison failed: {e}")
            return FaceComparisonResult(
                is_match=False,
                confidence_score=0.0,
                similarity_score=0.0,
                source_face_quality=0.0,
                target_face_quality=0.0,
                processing_time_ms=0,
                correlation_id=correlation_id
            )
    
    def _get_face_embedding(self, image_data: bytes, correlation_id: str) -> Optional[np.ndarray]:
        """Get face embedding using Hugging Face API"""
        try:
            if not self.hf_api_token:
                return None
            
            # Call Hugging Face for feature extraction
            response = self._call_huggingface_api(
                self.hf_models['face_embedding'],
                image_data,
                correlation_id
            )
            
            if response and 'embeddings' in response:
                return np.array(response['embeddings'])
            
            return None
            
        except Exception as e:
            logger.error(f"Face embedding failed: {e}")
            return None
    
    def detect_liveness(self, image_data: bytes, correlation_id: str = None) -> LivenessDetectionResult:
        """
        Detect if the face in the image is from a live person.
        
        Args:
            image_data: Image data as bytes
            correlation_id: Optional correlation ID
            
        Returns:
            LivenessDetectionResult: Liveness detection results
        """
        start_time = time.time()
        correlation_id = correlation_id or hashlib.md5(image_data).hexdigest()[:8]
        
        logger.info(f"Starting liveness detection", {'correlation_id': correlation_id})
        
        try:
            # Validate image
            self._validate_image(image_data)
            
            # Check cache
            cache_key = hashlib.md5(image_data).hexdigest()
            cached = self._get_cached_result(f"liveness_{cache_key}")
            if cached:
                cached.processing_time_ms = (time.time() - start_time) * 1000
                return cached
            
            # Try local liveness detection first
            if self.models.get('local_available', False):
                result = self._detect_liveness_local(image_data, correlation_id)
                if result.is_live is not None:
                    result.processing_time_ms = (time.time() - start_time) * 1000
                    self._cache_result(f"liveness_{cache_key}", 'liveness', result)
                    return result
            
            # Fallback to cloud
            result = self._detect_liveness_cloud(image_data, correlation_id)
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            self._cache_result(f"liveness_{cache_key}", 'liveness', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Liveness detection failed: {e}", {'correlation_id': correlation_id})
            return LivenessDetectionResult(
                is_live=False,
                confidence_score=0.0,
                liveness_score=0.0,
                spoof_score=0.0,
                method='error',
                processing_time_ms=(time.time() - start_time) * 1000,
                correlation_id=correlation_id
            )
    
    def _detect_liveness_local(self, image_data: bytes, correlation_id: str) -> LivenessDetectionResult:
        """Detect liveness using local methods"""
        try:
            # Use face landmarks for liveness detection
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            
            # Get face landmarks
            face_landmarks = face_recognition.face_landmarks(image_np)
            
            if not face_landmarks:
                return LivenessDetectionResult(
                    is_live=False,
                    confidence_score=0.0,
                    liveness_score=0.0,
                    spoof_score=0.0,
                    method='local',
                    processing_time_ms=0,
                    correlation_id=correlation_id
                )
            
            landmarks = face_landmarks[0]
            
            # Check for key facial features
            has_eyes = 'left_eye' in landmarks and 'right_eye' in landmarks
            has_nose = 'nose_bridge' in landmarks
            has_mouth = 'top_lip' in landmarks or 'bottom_lip' in landmarks
            
            liveness_score = 0
            if has_eyes: liveness_score += 0.35
            if has_nose: liveness_score += 0.35
            if has_mouth: liveness_score += 0.30
            
            is_live = liveness_score >= LIVENESS_THRESHOLD
            
            return LivenessDetectionResult(
                is_live=is_live,
                confidence_score=liveness_score * 100,
                liveness_score=liveness_score,
                spoof_score=1 - liveness_score,
                method='landmark_analysis',
                processing_time_ms=0,
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(f"Local liveness detection failed: {e}")
            return LivenessDetectionResult(
                is_live=None,
                confidence_score=0.0,
                liveness_score=0.0,
                spoof_score=0.0,
                method='error',
                processing_time_ms=0,
                correlation_id=correlation_id
            )
    
    def _detect_liveness_cloud(self, image_data: bytes, correlation_id: str) -> LivenessDetectionResult:
        """Fail closed when no configured liveness provider is available.

        A single still image cannot prove liveness. The API must be backed by
        a real liveness provider or the verification must remain pending.
        """
        return LivenessDetectionResult(
            is_live=False,
            confidence_score=0.0,
            liveness_score=0.0,
            spoof_score=1.0,
            method='provider_unavailable',
            processing_time_ms=0,
            correlation_id=correlation_id
        )
    
    def _call_huggingface_api(self, model_id: str, image_data: bytes, 
                              correlation_id: str) -> Dict[str, Any]:
        """Call Hugging Face Inference API"""
        if not self.hf_api_token:
            logger.warning("Hugging Face API token not set")
            return None
        
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare request
            url = f"{self.hf_api_url}/{model_id}"
            headers = {
                'Authorization': f'Bearer {self.hf_api_token}',
                'Content-Type': 'application/json',
            }
            
            # Make request
            response = requests.post(
                url,
                headers=headers,
                json={'inputs': image_b64},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Hugging Face API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Hugging Face API call failed: {e}")
            return None
    
    def _calculate_face_quality(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> float:
        """Calculate quality score for a detected face"""
        try:
            top, right, bottom, left = face_location
            face_roi = image[top:bottom, left:right]
            
            if face_roi.size == 0:
                return 0.0
            
            # Calculate sharpness
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray) / 255.0
            
            # Calculate contrast
            contrast = np.std(gray) / 255.0
            
            # Combine metrics
            quality = (min(sharpness / 100, 1.0) * 0.4 +
                      brightness * 0.3 +
                      min(contrast * 2, 1.0) * 0.3)
            
            return min(max(quality, 0), 1.0)
            
        except Exception:
            return 0.5
    
    def _validate_image(self, image_data: bytes) -> bool:
        """Validate image data"""
        if not image_data or len(image_data) == 0:
            raise ValueError("Image data is empty")
        
        if len(image_data) > MAX_IMAGE_SIZE:
            raise ValueError(f"Image size {len(image_data)} exceeds maximum {MAX_IMAGE_SIZE}")
        
        try:
            Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")
        
        return True
    
    def _cache_result(self, key: str, result_type: str, result: Any) -> None:
        """Cache result for faster retrieval"""
        self.cache[f"{key}_{result_type}"] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if still valid"""
        if key in self.cache:
            cached = self.cache[key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['result']
        return None


# Singleton instance
face_recognition_service = FaceRecognitionService()