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
        Send image to Gemini AI for water level analysis.
        """
        try:
            from utils.image_processing import analyze_water_gauge
            
            result = analyze_water_gauge(image_path)
            
            if result and result.get('water_level') is not None:
                return {
                    'water_level': result.get('water_level'),
                    'confidence': result.get('confidence', 0.8),
                    'is_valid': result.get('is_valid', True),
                    'reason': 'Gemini AI analysis'
                }
            else:
                return {
                    'water_level': None,
                    'confidence': 0.0,
                    'is_valid': False,
                    'reason': result.get('reason', 'Could not read gauge') if result else 'Gemini returned no result'
                }
                
        except ImportError as e:
            logger.error(f"Gemini import error: {e}")
            return self._error_result("Gemini API not available")
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return self._error_result(str(e))
    
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
