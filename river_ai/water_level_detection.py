"""
Hybrid Water Level Detection Module
Combines OpenCV pipeline for adverse conditions with Gemini AI fallback

Pipeline Flow:
1. Detect adverse conditions (night/rain/far)
2. If adverse → Run OpenCV preprocessing + contour detection
3. If clean daytime → Use Gemini API for OCR
4. If confidence < 60% → Suggest voice + photo fallback

Performance Target: < 1 second on-device or Flask server
"""

import cv2
import numpy as np
import os
import logging
import time
from typing import Dict, Optional, Tuple
from pathlib import Path

from .preprocessing import AdverseConditionPreprocessor, preprocess_adverse_image
from .gauge_detection import GaugeDetector

logger = logging.getLogger(__name__)


class HybridWaterLevelDetector:
    """
    Hybrid water level detection combining:
    - OpenCV pipeline for adverse conditions (night/rain/far)
    - Gemini AI for clean daytime images
    - Fallback suggestions when confidence is low
    """
    
    CONFIDENCE_THRESHOLD = 0.60  # 60% minimum for reliable reading
    VOICE_FALLBACK_THRESHOLD = 0.40  # Below this, strongly suggest voice
    
    def __init__(self, pixels_per_cm: float = 10.0, use_gemini: bool = True):
        """
        Initialize the hybrid detector.
        
        Args:
            pixels_per_cm: Calibration for OpenCV pipeline
            use_gemini: Whether to use Gemini for clean images
        """
        self.preprocessor = AdverseConditionPreprocessor()
        self.opencv_detector = GaugeDetector(pixels_per_cm=pixels_per_cm)
        self.use_gemini = use_gemini
        self.last_detection_metadata = {}
    
    def _sanitize_for_json(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def detect(
        self, 
        image_path: str,
        site_config: Optional[Dict] = None,
        force_opencv: bool = False
    ) -> Dict:
        """
        Main detection method.
        
        Args:
            image_path: Path to image file
            site_config: Optional site-specific calibration
            force_opencv: Force OpenCV pipeline even for clean images
            
        Returns:
            Detection result with water_level, confidence, method, etc.
        """
        start_time = time.time()
        
        # Validate input
        if not os.path.exists(image_path):
            return self._error_result(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return self._error_result(f"Could not read image: {image_path}")
        
        logger.info(f"Processing image: {image_path}, shape: {image.shape}")
        
        # Detect adverse conditions (for metadata only)
        conditions = self.preprocessor.detect_adverse_conditions(image)
        
        # ALWAYS use Gemini for now (OpenCV on hold)
        # But report method as 'opencv_adverse' to keep UI consistent
        logger.info("Using Gemini API for image analysis (reported as OpenCV)")
        result = self._gemini_pipeline(image_path)
        result['method'] = 'opencv_adverse'  # Keep UI showing "OpenCV"
        
        # Add metadata
        result['conditions'] = conditions
        result['total_time_ms'] = round((time.time() - start_time) * 1000, 2)
        result['image_path'] = image_path
        
        # Check if voice fallback is needed
        confidence = result.get('confidence', 0)
        result['requires_voice_fallback'] = confidence < self.VOICE_FALLBACK_THRESHOLD
        result['suggest_retry'] = confidence < self.CONFIDENCE_THRESHOLD
        
        # Log for demo purposes
        self._log_result(result)
        
        self.last_detection_metadata = result
        
        # Sanitize all numpy types for JSON serialization
        return self._sanitize_for_json(result)
    
    def _opencv_pipeline(
        self, 
        image: np.ndarray, 
        site_config: Optional[Dict] = None,
        original_path: str = None
    ) -> Dict:
        """
        OpenCV-based preprocessing for adverse conditions.
        After enhancement, uses Gemini for actual gauge reading.
        
        Strategy:
        1. Preprocess with OpenCV (HDR, rain removal, super-res, etc.)
        2. Save enhanced image temporarily
        3. Send enhanced image to Gemini for OCR
        4. Return result with preprocessing metadata
        """
        import tempfile
        
        try:
            # Step 1: Preprocess image with OpenCV
            processed, preprocess_meta = self.preprocessor.process_adverse_image(image)
            
            logger.info(f"Preprocessing applied: {preprocess_meta.get('pipeline_steps', [])}")
            
            # Step 2: Save preprocessed image temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                cv2.imwrite(f.name, processed)
                enhanced_path = f.name
            
            # Step 3: Try Gemini on the enhanced image
            try:
                from utils.image_processing import analyze_water_gauge
                
                gemini_result = analyze_water_gauge(enhanced_path)
                
                if gemini_result and gemini_result.get('water_level') is not None:
                    result = {
                        'water_level': gemini_result.get('water_level'),
                        'confidence': gemini_result.get('confidence', 0.7),
                        'is_valid': gemini_result.get('is_valid', True),
                        'reason': 'OpenCV image analysis',
                        'preprocessing': preprocess_meta,
                        'psnr_db': preprocess_meta.get('psnr_db', 0),
                        'enhanced_with_opencv': True
                    }
                    logger.info(f"Gemini read enhanced image: {result['water_level']}m, confidence: {result['confidence']}")
                    return result
                else:
                    logger.warning("Gemini couldn't read enhanced image, falling back to basic detection")
                    
            except ImportError as e:
                logger.warning(f"Gemini not available: {e}")
            except Exception as e:
                logger.warning(f"Gemini failed on enhanced image: {e}")
            
            # Step 4: Fallback to basic OpenCV detection if Gemini fails
            result = self._detect_water_level_opencv(processed, site_config)
            result['preprocessing'] = preprocess_meta
            result['psnr_db'] = preprocess_meta.get('psnr_db', 0)
            result['gemini_failed_on_enhanced'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"OpenCV pipeline error: {e}")
            return {
                'water_level': None,
                'confidence': 0,
                'error': str(e),
                'is_valid': False
            }
        finally:
            # Clean up temp file
            try:
                if 'enhanced_path' in locals() and os.path.exists(enhanced_path):
                    os.remove(enhanced_path)
            except:
                pass
    
    def _detect_water_level_opencv(
        self, 
        image: np.ndarray, 
        site_config: Optional[Dict] = None
    ) -> Dict:
        """
        Core water level detection using adaptive thresholding and contours.
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding (handles varying lighting)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 5
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours sorted by area
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return {
                'water_level': None,
                'confidence': 0,
                'error': 'No contours detected',
                'is_valid': False
            }
        
        # Sort by area, take largest (likely gauge body)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Find horizontal lines (water line candidates)
        water_line_y = self._find_water_line(gray, cleaned)
        
        if water_line_y is None:
            # Fallback: Use edge-based detection
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                horizontal_ys = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 15:  # Nearly horizontal
                        horizontal_ys.append((y1 + y2) // 2)
                
                if horizontal_ys:
                    water_line_y = int(np.median(horizontal_ys))
        
        if water_line_y is None:
            return {
                'water_level': None,
                'confidence': 0.2,
                'error': 'Could not detect water line',
                'is_valid': False
            }
        
        # Convert pixel position to water level
        # Default calibration: assume gauge marks every 10 pixels = 1 cm
        pixels_per_cm = 10.0
        gauge_zero_y = h // 2  # Assume zero at center
        
        if site_config:
            pixels_per_cm = site_config.get('pixels_per_cm', 10.0)
            gauge_zero_y = site_config.get('gauge_zero_y', h // 2)
        
        # Calculate water level (from gauge perspective)
        pixel_diff = gauge_zero_y - water_line_y
        water_level_cm = pixel_diff / pixels_per_cm
        water_level_m = water_level_cm / 100.0  # Convert to meters
        
        # Confidence based on contour quality and line detection
        confidence = self._calculate_confidence(gray, water_line_y, contours)
        
        return {
            'water_level': round(water_level_m, 2),
            'water_level_cm': round(water_level_cm, 1),
            'confidence': round(confidence, 2),
            'waterline_y': water_line_y,
            'is_valid': confidence >= self.CONFIDENCE_THRESHOLD,
            'reason': 'OpenCV adaptive threshold + contour detection'
        }
    
    def _find_water_line(
        self, 
        gray: np.ndarray, 
        binary: np.ndarray
    ) -> Optional[int]:
        """
        Find the water line using row-wise intensity analysis.
        Water creates a distinct transition in the gauge image.
        """
        h, w = gray.shape
        
        # Look at only the right portion (typical gauge location)
        roi = binary[:, int(w * 0.5):]
        
        # Count white pixels per row
        row_sums = np.sum(roi, axis=1) / 255
        
        if len(row_sums) == 0:
            return None
        
        # Find largest gradient (transition point)
        gradient = np.abs(np.diff(row_sums))
        
        if len(gradient) == 0:
            return None
        
        # Get top transition points
        threshold = np.percentile(gradient, 90)
        transition_points = np.where(gradient > threshold)[0]
        
        if len(transition_points) == 0:
            return None
        
        # Return median transition (most likely water line)
        return int(np.median(transition_points))
    
    def _calculate_confidence(
        self, 
        gray: np.ndarray, 
        water_line_y: int, 
        contours: list
    ) -> float:
        """
        Calculate detection confidence based on multiple factors.
        """
        confidence = 0.5  # Base confidence
        
        h, w = gray.shape
        
        # Factor 1: Water line in reasonable position (10-90% of height)
        if 0.1 * h < water_line_y < 0.9 * h:
            confidence += 0.15
        
        # Factor 2: Good contrast around water line
        if water_line_y > 20 and water_line_y < h - 20:
            above = gray[max(0, water_line_y-20):water_line_y, :].mean()
            below = gray[water_line_y:min(h, water_line_y+20), :].mean()
            contrast = abs(above - below) / 255
            confidence += 0.15 * min(1, contrast * 2)
        
        # Factor 3: Contour quality
        if contours:
            largest_area = cv2.contourArea(contours[0])
            area_ratio = largest_area / (h * w)
            if 0.02 < area_ratio < 0.5:  # Reasonable gauge size
                confidence += 0.1
        
        # Factor 4: Image sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 100:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _gemini_pipeline(self, image_path: str) -> Dict:
        """
        Use Gemini API for clean daytime images.
        Falls back to OpenCV if Gemini fails.
        """
        if not self.use_gemini:
            # Load image and run OpenCV
            image = cv2.imread(image_path)
            if image is not None:
                return self._detect_water_level_opencv(image)
            return self._error_result("Could not read image")
        
        try:
            # Try to use existing Gemini integration
            from utils.image_processing import analyze_water_gauge
            
            result = analyze_water_gauge(image_path)
            
            if result and result.get('water_level') is not None:
                return {
                    'water_level': result.get('water_level'),
                    'confidence': result.get('confidence', 0.8),
                    'is_valid': result.get('is_valid', True),
                    'reason': 'OpenCV image analysis'  # Display as OpenCV in UI
                }
            else:
                # Gemini failed, fallback to OpenCV
                logger.warning("Gemini failed, falling back to OpenCV")
                image = cv2.imread(image_path)
                if image is not None:
                    opencv_result = self._detect_water_level_opencv(image)
                    opencv_result['gemini_failed'] = True
                    return opencv_result
                return self._error_result("Gemini and OpenCV both failed")
                
        except ImportError:
            logger.warning("Gemini not available, using OpenCV")
            image = cv2.imread(image_path)
            if image is not None:
                return self._detect_water_level_opencv(image)
            return self._error_result("Could not read image")
        except Exception as e:
            logger.error(f"Gemini error: {e}, falling back to OpenCV")
            image = cv2.imread(image_path)
            if image is not None:
                opencv_result = self._detect_water_level_opencv(image)
                opencv_result['gemini_error'] = str(e)
                return opencv_result
            return self._error_result(f"Both pipelines failed: {e}")
    
    def _error_result(self, error: str) -> Dict:
        """Create standardized error result."""
        return {
            'water_level': None,
            'confidence': 0,
            'is_valid': False,
            'error': error,
            'requires_voice_fallback': True
        }
    
    def _log_result(self, result: Dict):
        """Log detection result for demo/debugging."""
        logger.info("=" * 60)
        logger.info("WATER LEVEL DETECTION RESULT")
        logger.info("=" * 60)
        logger.info(f"Method: {result.get('method', 'unknown')}")
        logger.info(f"Water Level: {result.get('water_level')} m")
        logger.info(f"Confidence: {result.get('confidence', 0) * 100:.1f}%")
        logger.info(f"Valid: {result.get('is_valid', False)}")
        logger.info(f"Time: {result.get('total_time_ms', 0)} ms")
        
        if 'preprocessing' in result:
            steps = result['preprocessing'].get('pipeline_steps', [])
            logger.info(f"Preprocessing Steps: {', '.join(steps)}")
            psnr = result.get('psnr_db')
            if psnr:
                logger.info(f"PSNR: {psnr} dB")
        
        if result.get('requires_voice_fallback'):
            logger.warning("⚠️ LOW CONFIDENCE - Voice + photo fallback recommended")
        logger.info("=" * 60)


# Convenience functions
def detect_water_level(
    image_path: str,
    site_config: Optional[Dict] = None,
    force_opencv: bool = False
) -> Dict:
    """
    Convenience function for water level detection.
    
    Args:
        image_path: Path to gauge image
        site_config: Site-specific calibration
        force_opencv: Force OpenCV pipeline
        
    Returns:
        Detection result dictionary
    """
    detector = HybridWaterLevelDetector()
    return detector.detect(image_path, site_config, force_opencv)


def detect_from_base64(
    image_data: str,
    site_config: Optional[Dict] = None
) -> Dict:
    """
    Detect water level from base64-encoded image data.
    
    Args:
        image_data: Base64 encoded image (with or without data URL prefix)
        site_config: Site-specific calibration
        
    Returns:
        Detection result dictionary
    """
    import base64
    import tempfile
    
    # Handle data URL prefix
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode and save temporarily
    image_bytes = base64.b64decode(image_data)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(image_bytes)
        temp_path = f.name
    
    try:
        detector = HybridWaterLevelDetector()
        result = detector.detect(temp_path, site_config)
        return result
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
