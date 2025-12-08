"""
Gauge Health Analysis Module
Detects gauge condition issues: algae, fading, damage, tilt

Output:
- gauge_algae_present: boolean
- gauge_faded: boolean  
- gauge_broken: boolean
- gauge_tilt_angle: float (degrees)
- gauge_visibility_score: 0-100
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class GaugeHealthAnalyzer:
    """
    Analyzes gauge condition from images.
    Detects algae, fading, damage, and tilt.
    """
    
    def __init__(self):
        pass
    
    def analyze(
        self, 
        image: np.ndarray,
        gauge_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Analyze gauge health from image.
        
        Args:
            image: BGR image
            gauge_roi: Optional gauge region (x, y, width, height)
            
        Returns:
            Gauge health analysis results
        """
        if image is None:
            return {"error": "No image provided"}
        
        h, w = image.shape[:2]
        
        # Extract gauge region
        if gauge_roi:
            x, y, gw, gh = gauge_roi
            gauge_region = image[y:y+gh, x:x+gw]
        else:
            # Default: right 30% of image
            x = int(w * 0.7)
            gauge_region = image[:, x:]
        
        if gauge_region.size == 0:
            return {"error": "Empty gauge region"}
        
        # Analyze each condition
        algae_result = self._detect_algae(gauge_region)
        fading_result = self._detect_fading(gauge_region)
        damage_result = self._detect_damage(gauge_region)
        tilt_result = self._detect_tilt(gauge_region)
        
        # Compute overall visibility score (0-100)
        visibility_penalties = 0
        if algae_result["algae_present"]:
            visibility_penalties += 25
        if fading_result["faded"]:
            visibility_penalties += 30
        if damage_result["broken"]:
            visibility_penalties += 35
        if abs(tilt_result["tilt_angle"]) > 10:
            visibility_penalties += 10
        
        visibility_score = max(0, 100 - visibility_penalties)
        
        return {
            "gauge_algae_present": algae_result["algae_present"],
            "algae_coverage": algae_result["coverage"],
            "gauge_faded": fading_result["faded"],
            "contrast_score": fading_result["contrast_score"],
            "gauge_broken": damage_result["broken"],
            "damage_score": damage_result["damage_score"],
            "gauge_tilt_angle": tilt_result["tilt_angle"],
            "gauge_visibility_score": visibility_score,
            "condition_notes": self._generate_notes(
                algae_result, fading_result, damage_result, tilt_result
            ),
            "maintenance_needed": visibility_score < 70
        }
    
    def _detect_algae(self, region: np.ndarray) -> Dict:
        """Detect algae coverage on gauge"""
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Green/algae range
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate coverage percentage
        coverage = np.sum(mask > 0) / mask.size
        
        return {
            "algae_present": coverage > 0.15,
            "coverage": round(coverage * 100, 1)
        }
    
    def _detect_fading(self, region: np.ndarray) -> Dict:
        """Detect faded numbers on gauge"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Compute contrast metrics
        min_val, max_val = np.min(gray), np.max(gray)
        contrast = (max_val - min_val) / 255.0
        
        # Local contrast (standard deviation)
        std = np.std(gray) / 128.0
        
        # Combined contrast score
        contrast_score = (contrast + std) / 2
        
        return {
            "faded": contrast_score < 0.3,
            "contrast_score": round(contrast_score, 3)
        }
    
    def _detect_damage(self, region: np.ndarray) -> Dict:
        """Detect physical damage (cracks, breaks) on gauge"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for irregular edge patterns (cracks)
        # Use morphological operations
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours for irregularity
        irregular_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                # Irregular shapes have low circularity
                if circularity < 0.3 and area > 100:
                    irregular_count += 1
        
        damage_score = min(1.0, irregular_count / 10)
        
        return {
            "broken": damage_score > 0.4,
            "damage_score": round(damage_score, 3),
            "irregular_patterns": irregular_count
        }
    
    def _detect_tilt(self, region: np.ndarray) -> Dict:
        """Detect gauge tilt using Hough line transform"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return {"tilt_angle": 0.0}
        
        # Find near-vertical lines (gauge lines)
        vertical_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(x2 - x1, y2 - y1) * 180 / np.pi
            # Angle from vertical (0 = perfectly vertical)
            if abs(angle) < 30:  # Near vertical
                vertical_angles.append(angle)
        
        if vertical_angles:
            avg_tilt = np.median(vertical_angles)
        else:
            avg_tilt = 0.0
        
        return {
            "tilt_angle": round(avg_tilt, 1),
            "lines_detected": len(vertical_angles)
        }
    
    def _generate_notes(
        self, 
        algae: Dict, 
        fading: Dict, 
        damage: Dict, 
        tilt: Dict
    ) -> str:
        """Generate human-readable condition notes"""
        notes = []
        
        if algae["algae_present"]:
            notes.append(f"Algae buildup detected ({algae['coverage']:.0f}% coverage)")
        
        if fading["faded"]:
            notes.append("Gauge numbers appear faded - low contrast")
        
        if damage["broken"]:
            notes.append("Possible physical damage or cracks detected")
        
        if abs(tilt["tilt_angle"]) > 5:
            notes.append(f"Gauge tilted {abs(tilt['tilt_angle']):.1f}° from vertical")
        
        if not notes:
            notes.append("Gauge appears in good condition")
        
        return "; ".join(notes)


def analyze_gauge_health(
    image_path: str,
    gauge_roi: Optional[Tuple[int, int, int, int]] = None
) -> Dict:
    """
    Convenience function to analyze gauge health from image file.
    
    Args:
        image_path: Path to image file
        gauge_roi: Optional gauge region
        
    Returns:
        Gauge health analysis results
    """
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}
    
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not read image"}
    
    analyzer = GaugeHealthAnalyzer()
    return analyzer.analyze(image, gauge_roi)
