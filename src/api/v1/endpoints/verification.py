"""
Verification API Endpoints

Provides REST API endpoints for face recognition, document verification,
and liveness detection.

@version 1.0.0
@author EasyGo Team
"""

import logging
import base64
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse
from typing import Optional
import uuid
from datetime import datetime

from ....services.face_recognition import face_recognition_service
from ....services.document_verification import document_verification_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/verification", tags=["verification"])


@router.post("/detect-face")
async def detect_face(
    image: UploadFile = File(...),
    correlation_id: Optional[str] = None
):
    """
    Detect face in uploaded image.
    
    Returns face detection results including face count, bounding boxes,
    and confidence scores.
    """
    try:
        # Validate file type
        if image.content_type not in ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # Read image data
        image_data = await image.read()
        
        if len(image_data) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="Image size exceeds 10MB")
        
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Detect faces
        result = face_recognition_service.detect_faces(
            image_data=image_data,
            correlation_id=correlation_id
        )
        
        return JSONResponse(content={
            'success': True,
            'data': {
                'has_face': result.has_face,
                'face_count': result.face_count,
                'confidence': result.confidence,
                'bounding_boxes': result.bounding_boxes,
                'quality_score': result.quality_score,
                'processing_time_ms': result.processing_time_ms,
                'correlation_id': result.correlation_id if hasattr(result, 'correlation_id') else correlation_id
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-faces")
async def compare_faces(
    source_image: UploadFile = File(..., description="Source image (reference)"),
    target_image: UploadFile = File(..., description="Target image to compare"),
    correlation_id: Optional[str] = None
):
    """
    Compare two faces and determine if they match.
    
    Returns similarity score, match result, and confidence level.
    """
    try:
        # Validate file types
        for img in [source_image, target_image]:
            if img.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
                raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # Read images
        source_data = await source_image.read()
        target_data = await target_image.read()
        
        # Generate correlation ID
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Compare faces
        result = face_recognition_service.compare_faces(
            source_image=source_data,
            target_image=target_data,
            correlation_id=correlation_id
        )
        
        return JSONResponse(content={
            'success': True,
            'data': {
                'is_match': result.is_match,
                'confidence_score': result.confidence_score,
                'similarity_score': result.similarity_score,
                'source_face_quality': result.source_face_quality,
                'target_face_quality': result.target_face_quality,
                'processing_time_ms': result.processing_time_ms,
                'correlation_id': result.correlation_id
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-liveness")
async def detect_liveness(
    image: UploadFile = File(...),
    correlation_id: Optional[str] = None
):
    """
    Detect liveness in uploaded image.
    
    Returns liveness detection results including whether the face is live,
    liveness score, and spoof score.
    """
    try:
        # Validate file type
        if image.content_type not in ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # Read image
        image_data = await image.read()
        
        # Generate correlation ID
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Detect liveness
        result = face_recognition_service.detect_liveness(
            image_data=image_data,
            correlation_id=correlation_id
        )
        
        return JSONResponse(content={
            'success': True,
            'data': {
                'is_live': result.is_live,
                'confidence_score': result.confidence_score,
                'liveness_score': result.liveness_score,
                'spoof_score': result.spoof_score,
                'method': result.method,
                'processing_time_ms': result.processing_time_ms,
                'correlation_id': result.correlation_id
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Liveness detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-document")
async def extract_document(
    document_type: str = Form(..., description="driver_license, national_id, or vehicle_registration"),
    image: UploadFile = File(...),
    correlation_id: Optional[str] = None
):
    """
    Extract data from document image using OCR.
    
    Returns extracted document data including fields, confidence scores,
    and validation results.
    """
    try:
        # Validate document type
        valid_types = ['driver_license', 'national_id', 'vehicle_registration']
        if document_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid document type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Validate file type
        if image.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # Read image
        image_data = await image.read()
        
        # Generate correlation ID
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Extract document data
        result = document_verification_service.extract_document_text(
            image_data=image_data,
            document_type=document_type,
            correlation_id=correlation_id
        )
        
        return JSONResponse(content={
            'success': True,
            'data': {
                'document_type': result.document_type,
                'extracted_data': result.extracted_data,
                'confidence_score': result.confidence_score,
                'validation_results': result.validation_results,
                'processing_time_ms': result.processing_time_ms,
                'correlation_id': result.correlation_id
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-driver")
async def verify_driver(
    selfie: UploadFile = File(...),
    id_document: UploadFile = File(...),
    license_document: UploadFile = File(...),
    vehicle_document: Optional[UploadFile] = File(None),
    correlation_id: Optional[str] = None
):
    """
    Complete driver verification flow combining face matching,
    document OCR, and liveness detection.
    """
    try:
        # Generate correlation ID
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Read files
        selfie_data = await selfie.read()
        id_data = await id_document.read()
        license_data = await license_document.read()
        
        # Step 1: Detect face in selfie
        selfie_face = face_recognition_service.detect_faces(
            image_data=selfie_data,
            correlation_id=f"{correlation_id}-selfie"
        )
        
        if not selfie_face.has_face:
            raise HTTPException(status_code=400, detail="No face detected in selfie")
        
        if selfie_face.face_count > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected in selfie")
        
        # Step 2: Detect liveness
        liveness = face_recognition_service.detect_liveness(
            image_data=selfie_data,
            correlation_id=f"{correlation_id}-liveness"
        )
        
        if not liveness.is_live:
            raise HTTPException(status_code=400, detail="Liveness detection failed")
        
        # Step 3: Extract ID document
        id_extraction = document_verification_service.extract_document_text(
            image_data=id_data,
            document_type='national_id',
            correlation_id=f"{correlation_id}-id"
        )
        
        # Step 4: Extract license document
        license_extraction = document_verification_service.extract_document_text(
            image_data=license_data,
            document_type='driver_license',
            correlation_id=f"{correlation_id}-license"
        )
        
        # Step 5: Compare selfie with ID photo (if available)
        # In production, you would extract the photo from the ID document
        # and compare with the selfie
        face_match = None
        if id_extraction.extracted_data:
            # This is simplified - you'd extract face from ID document
            face_match = face_recognition_service.compare_faces(
                source_image=id_data,  # Assuming ID contains face photo
                target_image=selfie_data,
                correlation_id=f"{correlation_id}-match"
            )
        
        # Calculate overall score
        verification_score = 0
        flags = []
        
        # Selfie score
        verification_score += selfie_face.confidence * 0.2
        
        # Liveness score
        verification_score += liveness.confidence_score * 0.3
        
        # Document scores
        verification_score += id_extraction.confidence_score * 0.15
        verification_score += license_extraction.confidence_score * 0.15
        
        # Face match score
        if face_match:
            verification_score += face_match.confidence_score * 0.2
            if not face_match.is_match:
                flags.append('FACE_MISMATCH')
        
        # Validation flags
        if not id_extraction.validation_results.get('is_valid', False):
            flags.append('ID_VALIDATION_FAILED')
        
        if not license_extraction.validation_results.get('is_valid', False):
            flags.append('LICENSE_VALIDATION_FAILED')
        
        is_verified = verification_score >= 70 and len(flags) == 0
        
        return JSONResponse(content={
            'success': True,
            'data': {
                'is_verified': is_verified,
                'verification_score': verification_score,
                'flags': flags,
                'selfie': {
                    'has_face': selfie_face.has_face,
                    'confidence': selfie_face.confidence,
                    'quality': selfie_face.quality_score,
                },
                'liveness': {
                    'is_live': liveness.is_live,
                    'score': liveness.confidence_score,
                },
                'documents': {
                    'national_id': {
                        'extracted': id_extraction.extracted_data,
                        'confidence': id_extraction.confidence_score,
                        'valid': id_extraction.validation_results.get('is_valid', False),
                    },
                    'driver_license': {
                        'extracted': license_extraction.extracted_data,
                        'confidence': license_extraction.confidence_score,
                        'valid': license_extraction.validation_results.get('is_valid', False),
                    }
                },
                'face_match': face_match.is_match if face_match else None,
                'processing_time_ms': sum([
                    selfie_face.processing_time_ms,
                    liveness.processing_time_ms,
                    id_extraction.processing_time_ms,
                    license_extraction.processing_time_ms,
                    face_match.processing_time_ms if face_match else 0
                ]),
                'correlation_id': correlation_id
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Driver verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'face_recognition': face_recognition_service.models.get('local_available', False) or bool(face_recognition_service.hf_api_token),
            'document_verification': document_verification_service.vision_available or document_verification_service.tesseract_available,
        }
    })
