"""
Water Level Detection Module - Gemini Only
Uses Google Gemini AI exclusively for water level gauge reading.

No OpenCV preprocessing - sends images directly to Gemini for analysis.
"""

import os
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GeminiWaterLevelDetector:
    """
    Water level detection using Google Gemini AI exclusively.
    No OpenCV dependencies - pure Gemini-based analysis.
    """
    
    CONFIDENCE_THRESHOLD = 0.60  # 60% minimum for reliable reading
    VOICE_FALLBACK_THRESHOLD = 0.40  # Below this, strongly suggest voice
    
    def __init__(self):
        """Initialize the Gemini detector."""
        self.last_detection_metadata = {}
    
    def detect(self, image_path: str, site_config: Optional[Dict] = None) -> Dict:
        """
        Main detection method using Gemini AI.
        
        Args:
            image_path: Path to image file
            site_config: Optional site-specific configuration (not used, for compatibility)
            
        Returns:
            Detection result with water_level, confidence, method, etc.
        """
        start_time = time.time()
        
        # Validate input
        if not os.path.exists(image_path):
            return self._error_result(f"Image not found: {image_path}")
        
        logger.info(f"Processing image with Gemini: {image_path}")
        
        # Call Gemini API
        result = self._analyze_with_gemini(image_path)
        
        # Add metadata
        result['total_time_ms'] = round((time.time() - start_time) * 1000, 2)
        result['image_path'] = image_path
        result['method'] = 'gemini'
        
        # Check if voice fallback is needed
        confidence = result.get('confidence', 0)
        result['requires_voice_fallback'] = confidence < self.VOICE_FALLBACK_THRESHOLD
        result['suggest_retry'] = confidence < self.CONFIDENCE_THRESHOLD
        
        # Log result
        if result.get('water_level') is not None:
            logger.info(f"Gemini detected water level: {result['water_level']}m, confidence: {result['confidence']}")
        else:
            logger.warning(f"Gemini could not detect water level: {result.get('reason', 'Unknown error')}")
        
        self.last_detection_metadata = result
        return result
    
    def _analyze_with_gemini(self, image_path: str) -> Dict:
        """
        Send image to Gemini AI for water level analysis with scene validation and tamper detection.
        """
        try:
            from utils.image_processing import analyze_water_gauge
            
            result = analyze_water_gauge(image_path)
            
            if result:
                # Check scene validation first
                scene_valid = result.get('scene_valid', True)
                no_gauge = result.get('no_gauge_detected', False)
                is_phone_image = result.get('is_phone_image', False)
                scene_type = result.get('scene_type', 'unknown')
                
                # Check for tamper detection
                tamper_detected = result.get('tamper_detected', False)
                
                # Build response with all available fields
                response = {
                    'water_level': result.get('water_level'),
                    'confidence': result.get('confidence', 0.0),
                    'is_valid': result.get('is_valid', False),
                    'reason': result.get('reason', 'Analysis complete'),
                    # Scene detection fields
                    'scene_type': scene_type,
                    'scene_valid': scene_valid,
                    'is_phone_image': is_phone_image,
                    'scene_issue': result.get('scene_issue'),
                    'no_gauge_detected': no_gauge,
                    # Enhanced fields
                    'gauge_location': result.get('gauge_location', 'Not detected'),
                    'water_line_position': result.get('water_line_position', 'Not detected'),
                    'visible_markers': result.get('visible_markers', []),
                    'scale_direction': result.get('scale_direction'),
                    'reasoning': result.get('reasoning'),
                    'tamper_detected': tamper_detected,
                    'tamper_reason': result.get('tamper_reason'),
                    'image_quality': result.get('image_quality', 'unknown'),
                    'suggestions': result.get('suggestions', [])
                }
                
                # Add demo notice for phone images
                if is_phone_image and response.get('water_level') is not None:
                    response['demo_notice'] = "📱 Image from phone screen - Allowed for demo purposes"
                    response['reason'] = f"{response.get('reason', '')} [Demo: Phone image detected]"
                
                # Handle invalid scene (gauge not in water, etc.)
                if not scene_valid:
                    response['is_valid'] = False
                    response['water_level'] = None
                    scene_issue = result.get('scene_issue', 'Invalid scene detected')
                    response['reason'] = f"Scene validation failed: {scene_issue}"
                    if not response['suggestions']:
                        response['suggestions'] = [
                            "Ensure the gauge is physically installed in water",
                            "Take a photo of the actual gauge at the river site",
                            "The gauge must be partially submerged in water"
                        ]
                
                # Handle no gauge detected
                if no_gauge:
                    response['is_valid'] = False
                    response['water_level'] = None
                    response['reason'] = "No water level gauge detected in this image"
                    response['suggestions'] = [
                        "Point the camera at the water level gauge/staff",
                        "Ensure the gauge numbers are visible",
                        "Move closer if the gauge is too far away"
                    ]
                
                # If tampered, mark as invalid and adjust reason
                if tamper_detected:
                    response['is_valid'] = False
                    response['reason'] = f"Image appears tampered: {result.get('tamper_reason', 'Suspicious alterations detected')}"
                    if not response['suggestions']:
                        response['suggestions'] = [
                            "Take a fresh photo without any editing",
                            "Ensure the gauge is clearly visible",
                            "Capture the image in good lighting"
                        ]
                
                return response
            else:
                return {
                    'water_level': None,
                    'confidence': 0.0,
                    'is_valid': False,
                    'reason': 'OpenCV is taking too long to process. Please try again.',
                    'suggestions': ['Try taking the photo again with better lighting', 'Ensure the gauge is clearly visible']
                }
                
        except ImportError as e:
            logger.error(f"OpenCV import error: {e}")
            return self._error_result("OpenCV processing unavailable")
        except Exception as e:
            logger.error(f"OpenCV analysis error: {e}")
            return self._error_result(f"OpenCV processing error: Please try again")
    
    def _error_result(self, message: str) -> Dict:
        """Create a standardized error result."""
        return {
            'water_level': None,
            'confidence': 0.0,
            'is_valid': False,
            'reason': message,
            'error': message
        }


# Create singleton for easy access
detector = GeminiWaterLevelDetector()


def detect_water_level(image_path: str, site_config: Optional[Dict] = None) -> Dict:
    """
    Convenience function to detect water level using Gemini.
    
    Args:
        image_path: Path to the image file
        site_config: Optional site configuration
        
    Returns:
        Detection result dictionary
    """
    return detector.detect(image_path, site_config)


# For backward compatibility with existing imports
class HybridWaterLevelDetector(GeminiWaterLevelDetector):
    """Alias for backward compatibility - now uses Gemini only."""
    
    def __init__(self, pixels_per_cm: float = 10.0, use_gemini: bool = True):
        super().__init__()
        # pixels_per_cm and use_gemini ignored - always uses Gemini
