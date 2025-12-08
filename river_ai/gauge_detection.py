"""
Gauge Detection Module
Detects water level from river gauge images using computer vision

Methods:
1. ROI-based detection using site calibration
2. Edge detection for waterline finding
3. Pixel-to-cm conversion using calibration data
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class GaugeDetector:
    """
    Detects water level from gauge images.
    Uses calibration data from river_sites table.
    """
    
    def __init__(self, pixels_per_cm: float = 10.0, gauge_zero_y: int = 500):
        """
        Initialize gauge detector with calibration.
        
        Args:
            pixels_per_cm: Calibration factor (pixels per centimeter)
            gauge_zero_y: Y-pixel position of the gauge's zero mark
        """
        self.pixels_per_cm = pixels_per_cm
        self.gauge_zero_y = gauge_zero_y
    
    def detect_water_level(
        self, 
        image: np.ndarray,
        gauge_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Detect water level from image.
        
        Args:
            image: BGR image (OpenCV format)
            gauge_roi: Optional (x, y, width, height) for gauge region
            
        Returns:
            Dict with water_level_cm, confidence, waterline_y, debug_info
        """
        if image is None:
            return {"water_level_cm": None, "confidence": 0, "error": "No image provided"}
        
        h, w = image.shape[:2]
        
        # Extract gauge region (or use center-right portion as default)
        if gauge_roi:
            x, y, gw, gh = gauge_roi
            gauge_region = image[y:y+gh, x:x+gw]
        else:
            # Default: right third of image, full height
            x = int(w * 0.6)
            gauge_region = image[:, x:]
        
        # Convert to grayscale
        gray = cv2.cvtColor(gauge_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Method 1: Edge-based waterline detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find horizontal edges (waterline tends to be horizontal)
        kernel = np.ones((1, 15), np.uint8)
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find the strongest horizontal line using Hough transform
        lines = cv2.HoughLinesP(horizontal_edges, 1, np.pi/180, 50, 
                                minLineLength=30, maxLineGap=10)
        
        waterline_y = None
        confidence = 0.0
        
        if lines is not None:
            # Find horizontal lines (angle close to 0)
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 15:  # Nearly horizontal
                    horizontal_lines.append((y1 + y2) // 2)
            
            if horizontal_lines:
                # Use the most prominent horizontal line
                waterline_y = int(np.median(horizontal_lines))
                confidence = min(1.0, len(horizontal_lines) / 10)
        
        # Method 2: Color-based water detection (fallback)
        if waterline_y is None or confidence < 0.3:
            hsv = cv2.cvtColor(gauge_region, cv2.COLOR_BGR2HSV)
            
            # Water tends to be blue/green/brown
            water_mask = cv2.inRange(hsv, (0, 20, 20), (180, 255, 200))
            
            # Find the transition point
            row_sums = np.sum(water_mask, axis=1)
            if len(row_sums) > 0:
                # Find where water starts (from bottom up)
                for i in range(len(row_sums) - 1, 0, -1):
                    if row_sums[i] < row_sums[i-1] * 0.5:
                        waterline_y = i
                        confidence = 0.4
                        break
        
        # Convert pixel position to water level in cm
        water_level_cm = None
        if waterline_y is not None:
            # Adjust for ROI offset
            if gauge_roi:
                actual_y = gauge_roi[1] + waterline_y
            else:
                actual_y = waterline_y
            
            # Calculate water level (higher pixel = lower water level)
            pixel_diff = self.gauge_zero_y - actual_y
            water_level_cm = pixel_diff / self.pixels_per_cm
            
            # Clamp to reasonable range
            water_level_cm = max(0, min(1000, water_level_cm))
        
        return {
            "water_level_cm": round(water_level_cm, 1) if water_level_cm else None,
            "confidence": round(confidence, 2),
            "waterline_y": waterline_y,
            "method": "edge_detection" if confidence >= 0.3 else "color_based",
            "debug_info": {
                "gauge_zero_y": self.gauge_zero_y,
                "pixels_per_cm": self.pixels_per_cm,
                "image_height": h,
                "image_width": w
            }
        }


def detect_water_level(
    image_path: str,
    site_config: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to detect water level from image file.
    
    Args:
        image_path: Path to image file
        site_config: Optional site configuration with calibration data
        
    Returns:
        Detection result dictionary
    """
    import os
    
    if not os.path.exists(image_path):
        return {"water_level_cm": None, "confidence": 0, "error": f"File not found: {image_path}"}
    
    image = cv2.imread(image_path)
    if image is None:
        return {"water_level_cm": None, "confidence": 0, "error": "Could not read image"}
    
    # Extract calibration from site config
    pixels_per_cm = 10.0
    gauge_zero_y = image.shape[0] // 2
    gauge_roi = None
    
    if site_config:
        pixels_per_cm = site_config.get("gauge_calibration_pixels_per_cm", 10.0)
        gauge_zero_y = site_config.get("gauge_zero_pixel_y", image.shape[0] // 2)
        if "gauge_roi" in site_config:
            gauge_roi = tuple(site_config["gauge_roi"])
    
    detector = GaugeDetector(pixels_per_cm, gauge_zero_y)
    return detector.detect_water_level(image, gauge_roi)
