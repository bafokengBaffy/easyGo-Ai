"""
Document Verification Service

Handles OCR and document verification using Google Vision API and
Tesseract OCR. Supports driver's license, national ID, and vehicle
registration documents.

@version 1.0.0
@author EasyGo Team
"""

import os
import re
import json
import base64
import logging
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import io
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DocumentExtractionResult:
    """Document extraction result container"""
    document_type: str
    extracted_data: Dict[str, Any]
    confidence_score: float
    validation_results: Dict[str, Any]
    raw_text: str
    processing_time_ms: float
    correlation_id: str

@dataclass
class DocumentVerificationResult:
    """Document verification result container"""
    document_id: str
    is_verified: bool
    confidence_score: float
    flags: List[str]
    extracted_data: Dict[str, Any]
    validation_status: str
    processing_time_ms: float
    correlation_id: str


class DocumentVerificationService:
    """
    Service for document OCR and verification using Google Cloud Vision
    and Tesseract OCR.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize document verification service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._initialize_services()
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Document field extraction patterns
        self.patterns = {
            'driver_license': {
                'license_number': [
                    r'(?:license|licence|licence\s*number|licence\s*#)\s*[:\-]?\s*([A-Z0-9\-]{5,20})',
                    r'([A-Z]{1,3}[0-9]{4,8})',
                ],
                'full_name': [
                    r'(?:name|full\s*name|driver\'s\s*name)\s*[:\-]?\s*([A-Za-z\s]+)',
                ],
                'date_of_birth': [
                    r'(?:birth|date\s*of\s*birth|dob)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                ],
                'expiry_date': [
                    r'(?:expiry|expiration|expires|valid\s*until)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                ],
                'license_class': [
                    r'(?:class|licence\s*class)\s*[:\-]?\s*([A-D])',
                ],
            },
            'national_id': {
                'id_number': [
                    r'(?:id|national\s*id|identity)\s*[:\-]?\s*([A-Z0-9\-]{5,15})',
                ],
                'full_name': [
                    r'(?:name|full\s*name)\s*[:\-]?\s*([A-Za-z\s]+)',
                ],
                'date_of_birth': [
                    r'(?:birth|date\s*of\s*birth|dob)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                ],
                'nationality': [
                    r'(?:nationality|nation)\s*[:\-]?\s*([A-Za-z\s]+)',
                ],
                'gender': [
                    r'(?:gender|sex)\s*[:\-]?\s*([MFmf])',
                ],
            },
            'vehicle_registration': {
                'registration_number': [
                    r'(?:registration|reg\s*no|plate|license\s*plate)\s*[:\-]?\s*([A-Z0-9\-]{5,15})',
                ],
                'vehicle_make': [
                    r'(?:make|brand)\s*[:\-]?\s*([A-Za-z\s\-]+)',
                ],
                'vehicle_model': [
                    r'(?:model)\s*[:\-]?\s*([A-Za-z0-9\s\-]+)',
                ],
                'vehicle_year': [
                    r'(?:year|model\s*year)\s*[:\-]?\s*(\d{4})',
                ],
                'vin': [
                    r'(?:vin|chassis|vehicle\s*id)\s*[:\-]?\s*([A-Z0-9]{17})',
                ],
                'owner_name': [
                    r'(?:owner|registered\s*to)\s*[:\-]?\s*([A-Za-z\s]+)',
                ],
            }
        }
        
        logger.info("DocumentVerificationService initialized")
    
    def _initialize_services(self):
        """Initialize cloud services for document processing"""
        # Google Cloud Vision
        try:
            from google.cloud import vision
            self.vision_client = vision.ImageAnnotatorClient()
            self.vision_available = True
        except ImportError:
            self.vision_available = False
            logger.warning("Google Cloud Vision not available")
        
        # Tesseract OCR
        try:
            import pytesseract
            self.tesseract = pytesseract
            self.tesseract_available = True
        except ImportError:
            self.tesseract_available = False
            logger.warning("Tesseract OCR not available")
    
    def extract_document_text(self, image_data: bytes, document_type: str,
                              correlation_id: str = None) -> DocumentExtractionResult:
        """
        Extract text from document image.
        
        Args:
            image_data: Document image as bytes
            document_type: Type of document (driver_license, national_id, vehicle_registration)
            correlation_id: Optional correlation ID
            
        Returns:
            DocumentExtractionResult: Extraction results
        """
        start_time = time.time()
        correlation_id = correlation_id or hashlib.md5(image_data).hexdigest()[:8]
        
        logger.info(f"Starting document text extraction", {
            'correlation_id': correlation_id,
            'document_type': document_type
        })
        
        try:
            # Check cache
            cache_key = hashlib.md5(image_data + document_type.encode()).hexdigest()
            cached = self._get_cached_result(cache_key)
            if cached:
                cached.processing_time_ms = (time.time() - start_time) * 1000
                return cached
            
            # Extract raw text
            raw_text = self._perform_ocr(image_data, correlation_id)
            
            # Extract structured data
            extracted_data = self._extract_fields(raw_text, document_type, correlation_id)
            
            # Validate extracted data
            validation_results = self._validate_extracted_data(extracted_data, document_type)
            
            # Calculate confidence
            confidence_score = self._calculate_extraction_confidence(
                raw_text, extracted_data, document_type
            )
            
            result = DocumentExtractionResult(
                document_type=document_type,
                extracted_data=extracted_data,
                confidence_score=confidence_score,
                validation_results=validation_results,
                raw_text=raw_text,
                processing_time_ms=(time.time() - start_time) * 1000,
                correlation_id=correlation_id
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            logger.info(f"Document extraction completed", {
                'correlation_id': correlation_id,
                'confidence': confidence_score,
                'processing_time': result.processing_time_ms
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Document extraction failed: {e}", {
                'correlation_id': correlation_id
            })
            return DocumentExtractionResult(
                document_type=document_type,
                extracted_data={},
                confidence_score=0.0,
                validation_results={'error': str(e)},
                raw_text='',
                processing_time_ms=(time.time() - start_time) * 1000,
                correlation_id=correlation_id
            )
    
    def _perform_ocr(self, image_data: bytes, correlation_id: str) -> str:
        """Perform OCR on image using available services"""
        try:
            # Try Google Cloud Vision first if available
            if self.vision_available:
                text = self._ocr_google_vision(image_data)
                if text and len(text) > 10:
                    return text
            
            # Fallback to Tesseract
            if self.tesseract_available:
                text = self._ocr_tesseract(image_data)
                if text and len(text) > 10:
                    return text
            
            # Return empty if no OCR available
            return ''
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ''
    
    def _ocr_google_vision(self, image_data: bytes) -> str:
        """Perform OCR using Google Cloud Vision"""
        try:
            image = vision.Image(content=image_data)
            response = self.vision_client.text_detection(image=image)
            
            if response.error.message:
                logger.error(f"Google Vision error: {response.error.message}")
                return ''
            
            texts = response.text_annotations
            if texts:
                return texts[0].description
            return ''
            
        except Exception as e:
            logger.error(f"Google Vision OCR failed: {e}")
            return ''
    
    def _ocr_tesseract(self, image_data: bytes) -> str:
        """Perform OCR using Tesseract"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image for better OCR
            image_np = np.array(image)
            
            # Convert to grayscale if needed
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Perform OCR
            text = self.tesseract.image_to_string(thresh, config='--psm 6')
            return text
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ''
    
    def _extract_fields(self, text: str, document_type: str, 
                       correlation_id: str) -> Dict[str, Any]:
        """Extract structured fields from raw text"""
        extracted = {}
        
        if document_type not in self.patterns:
            return extracted
        
        patterns = self.patterns[document_type]
        
        for field_name, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    extracted[field_name] = matches.group(1).strip()
                    break
            
            if field_name not in extracted:
                extracted[field_name] = ''
        
        return extracted
    
    def _validate_extracted_data(self, data: Dict[str, Any], 
                                document_type: str) -> Dict[str, Any]:
        """Validate extracted document data"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'field_validations': {}
        }
        
        if document_type == 'driver_license':
            validation = self._validate_driver_license(data)
        elif document_type == 'national_id':
            validation = self._validate_national_id(data)
        elif document_type == 'vehicle_registration':
            validation = self._validate_vehicle_registration(data)
        
        return validation
    
    def _validate_driver_license(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate driver license data"""
        errors = []
        warnings = []
        field_validations = {}
        
        # License number
        license_number = data.get('license_number', '')
        if not license_number:
            errors.append('License number is required')
            field_validations['license_number'] = {'valid': False, 'error': 'Missing'}
        elif len(license_number) < 5:
            warnings.append('License number seems too short')
            field_validations['license_number'] = {'valid': True, 'warning': 'Short length'}
        else:
            field_validations['license_number'] = {'valid': True}
        
        # Full name
        full_name = data.get('full_name', '')
        if not full_name:
            errors.append('Full name is required')
            field_validations['full_name'] = {'valid': False, 'error': 'Missing'}
        elif len(full_name.split()) < 2:
            warnings.append('Name may be incomplete')
            field_validations['full_name'] = {'valid': True, 'warning': 'Incomplete name'}
        else:
            field_validations['full_name'] = {'valid': True}
        
        # Date of birth
        dob = data.get('date_of_birth', '')
        if dob:
            try:
                dob_date = self._parse_date(dob)
                age = (datetime.now() - dob_date).days / 365.25
                if age < 18:
                    errors.append('Driver must be at least 18 years old')
                    field_validations['date_of_birth'] = {'valid': False, 'error': 'Underage'}
                elif age > 70:
                    warnings.append('Driver is over 70 years old, may require medical check')
                    field_validations['date_of_birth'] = {'valid': True, 'warning': 'Age > 70'}
                else:
                    field_validations['date_of_birth'] = {'valid': True}
            except:
                warnings.append('Could not parse date of birth')
                field_validations['date_of_birth'] = {'valid': True, 'warning': 'Unparseable'}
        else:
            errors.append('Date of birth is required')
            field_validations['date_of_birth'] = {'valid': False, 'error': 'Missing'}
        
        # Expiry date
        expiry = data.get('expiry_date', '')
        if expiry:
            try:
                expiry_date = self._parse_date(expiry)
                if expiry_date < datetime.now():
                    errors.append('License has expired')
                    field_validations['expiry_date'] = {'valid': False, 'error': 'Expired'}
                else:
                    days_until_expiry = (expiry_date - datetime.now()).days
                    if days_until_expiry < 30:
                        warnings.append(f'License expires in {days_until_expiry} days')
                        field_validations['expiry_date'] = {'valid': True, 'warning': 'Expiring soon'}
                    else:
                        field_validations['expiry_date'] = {'valid': True}
            except:
                warnings.append('Could not parse expiry date')
                field_validations['expiry_date'] = {'valid': True, 'warning': 'Unparseable'}
        else:
            errors.append('Expiry date is required')
            field_validations['expiry_date'] = {'valid': False, 'error': 'Missing'}
        
        # License class
        license_class = data.get('license_class', '')
        if license_class and license_class.upper() not in ['A', 'B', 'C', 'D', 'E']:
            warnings.append('License class may be invalid')
            field_validations['license_class'] = {'valid': True, 'warning': 'Unknown class'}
        else:
            field_validations['license_class'] = {'valid': True if license_class else False}
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'field_validations': field_validations
        }
    
    def _validate_national_id(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate national ID data"""
        errors = []
        warnings = []
        field_validations = {}
        
        # ID number
        id_number = data.get('id_number', '')
        if not id_number:
            errors.append('ID number is required')
            field_validations['id_number'] = {'valid': False, 'error': 'Missing'}
        elif len(id_number) < 5:
            warnings.append('ID number seems too short')
            field_validations['id_number'] = {'valid': True, 'warning': 'Short length'}
        else:
            field_validations['id_number'] = {'valid': True}
        
        # Full name
        full_name = data.get('full_name', '')
        if not full_name:
            errors.append('Full name is required')
            field_validations['full_name'] = {'valid': False, 'error': 'Missing'}
        else:
            field_validations['full_name'] = {'valid': True}
        
        # Date of birth
        dob = data.get('date_of_birth', '')
        if dob:
            try:
                dob_date = self._parse_date(dob)
                field_validations['date_of_birth'] = {'valid': True}
            except:
                warnings.append('Could not parse date of birth')
                field_validations['date_of_birth'] = {'valid': True, 'warning': 'Unparseable'}
        else:
            warnings.append('Date of birth is recommended')
            field_validations['date_of_birth'] = {'valid': True, 'warning': 'Missing'}
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'field_validations': field_validations
        }
    
    def _validate_vehicle_registration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate vehicle registration data"""
        errors = []
        warnings = []
        field_validations = {}
        
        # Registration number
        reg_no = data.get('registration_number', '')
        if not reg_no:
            errors.append('Registration number is required')
            field_validations['registration_number'] = {'valid': False, 'error': 'Missing'}
        else:
            field_validations['registration_number'] = {'valid': True}
        
        # Vehicle make
        make = data.get('vehicle_make', '')
        if not make:
            errors.append('Vehicle make is required')
            field_validations['vehicle_make'] = {'valid': False, 'error': 'Missing'}
        else:
            field_validations['vehicle_make'] = {'valid': True}
        
        # Vehicle model
        model = data.get('vehicle_model', '')
        if not model:
            errors.append('Vehicle model is required')
            field_validations['vehicle_model'] = {'valid': False, 'error': 'Missing'}
        else:
            field_validations['vehicle_model'] = {'valid': True}
        
        # Vehicle year
        year = data.get('vehicle_year', '')
        if year:
            try:
                year_int = int(year)
                current_year = datetime.now().year
                if year_int < 2000 or year_int > current_year:
                    warnings.append(f'Vehicle year {year_int} seems invalid')
                    field_validations['vehicle_year'] = {'valid': True, 'warning': 'Invalid year'}
                else:
                    field_validations['vehicle_year'] = {'valid': True}
            except:
                warnings.append('Could not parse vehicle year')
                field_validations['vehicle_year'] = {'valid': True, 'warning': 'Unparseable'}
        else:
            warnings.append('Vehicle year is recommended')
            field_validations['vehicle_year'] = {'valid': True, 'warning': 'Missing'}
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'field_validations': field_validations
        }
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string from various formats"""
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        # Try with year inference
        for fmt in ['%d/%m/%y', '%m/%d/%y', '%d-%m-%y', '%m-%d-%y']:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse date: {date_str}")
    
    def _calculate_extraction_confidence(self, raw_text: str, 
                                        extracted_data: Dict[str, Any],
                                        document_type: str) -> float:
        """Calculate confidence score for extraction"""
        if not raw_text:
            return 0.0
        
        # Check how many fields were extracted
        total_fields = len(self.patterns.get(document_type, {}))
        extracted_fields = sum(1 for v in extracted_data.values() if v)
        
        if total_fields == 0:
            return 0.0
        
        field_ratio = extracted_fields / total_fields
        
        # Length-based confidence
        text_length = len(raw_text)
        length_score = min(text_length / 500, 1.0)  # 500 chars = full score
        
        # Overall confidence
        confidence = (field_ratio * 0.6 + length_score * 0.4) * 100
        
        return min(max(confidence, 0), 100)
    
    def _cache_result(self, key: str, result: Any) -> None:
        """Cache result"""
        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result"""
        if key in self.cache:
            cached = self.cache[key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['result']
        return None


# Singleton instance
document_verification_service = DocumentVerificationService()