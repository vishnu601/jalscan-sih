"""
Riverbank Erosion Detection Module
Tracks riverbank changes over time using image comparison

Output:
- bank_status: stable / minor_erosion / heavy_erosion / unknown
- erosion_change_pct: percentage change from baseline
- ssim_score: structural similarity with baseline
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
import os
import json

logger = logging.getLogger(__name__)


class BankErosionAnalyzer:
    """
    Analyzes riverbank erosion by comparing images over time.
    Uses SSIM and contour analysis for change detection.
    """
    
    # Erosion thresholds
    EROSION_THRESHOLDS = {
        "stable": (0.9, 1.0),        # SSIM > 0.9
        "minor_erosion": (0.7, 0.9),  # SSIM 0.7-0.9
        "heavy_erosion": (0.0, 0.7)   # SSIM < 0.7
    }
    
    def analyze(
        self,
        current_image: np.ndarray,
        baseline_image: Optional[np.ndarray] = None,
        bank_roi_polygon: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        """
        Analyze riverbank erosion by comparing current image to baseline.
        
        Args:
            current_image: Current BGR image
            baseline_image: Previous/baseline BGR image for comparison
            bank_roi_polygon: List of (x, y) points defining bank region
            
        Returns:
            Erosion analysis results
        """
        if current_image is None:
            return {"bank_status": "unknown", "error": "No image provided"}
        
        h, w = current_image.shape[:2]
        
        # Extract bank region
        if bank_roi_polygon:
            current_bank = self._extract_polygon_region(current_image, bank_roi_polygon)
            if baseline_image is not None:
                baseline_bank = self._extract_polygon_region(baseline_image, bank_roi_polygon)
            else:
                baseline_bank = None
        else:
            # Default: left side of image (typically where bank is visible)
            x_end = int(w * 0.3)
            current_bank = current_image[:, :x_end]
            if baseline_image is not None:
                baseline_bank = baseline_image[:, :x_end]
            else:
                baseline_bank = None
        
        # If no baseline, analyze current image for erosion signs
        if baseline_bank is None:
            return self._analyze_single_image(current_bank)
        
        # Compare with baseline
        return self._compare_images(current_bank, baseline_bank)
    
    def _extract_polygon_region(
        self, 
        image: np.ndarray, 
        polygon: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Extract region defined by polygon"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        # Apply mask
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Crop to bounding rectangle
        x, y, w, h = cv2.boundingRect(points)
        return result[y:y+h, x:x+w]
    
    def _compare_images(
        self, 
        current: np.ndarray, 
        baseline: np.ndarray
    ) -> Dict:
        """Compare current and baseline images for changes"""
        # Resize to same dimensions
        if current.shape != baseline.shape:
            baseline = cv2.resize(baseline, (current.shape[1], current.shape[0]))
        
        # Convert to grayscale
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM (Structural Similarity Index)
        ssim_score = self._compute_ssim(current_gray, baseline_gray)
        
        # Compute absolute difference
        diff = cv2.absdiff(current_gray, baseline_gray)
        diff_mean = np.mean(diff)
        diff_max = np.max(diff)
        
        # Find changed regions
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        changed_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        change_pct = (changed_pixels / total_pixels) * 100
        
        # Classify erosion status
        bank_status = "unknown"
        for status, (low, high) in self.EROSION_THRESHOLDS.items():
            if low <= ssim_score < high:
                bank_status = status
                break
        
        # Additional analysis: water-land boundary shift
        boundary_shift = self._analyze_boundary_shift(current_gray, baseline_gray)
        
        return {
            "bank_status": bank_status,
            "ssim_score": round(ssim_score, 3),
            "erosion_change_pct": round(change_pct, 1),
            "diff_mean": round(diff_mean, 2),
            "boundary_shift_pixels": boundary_shift,
            "erosion_detected": bank_status in ["minor_erosion", "heavy_erosion"],
            "method": "image_comparison"
        }
    
    def _compute_ssim(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray
    ) -> float:
        """Compute Structural Similarity Index"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
    
    def _analyze_boundary_shift(
        self, 
        current: np.ndarray, 
        baseline: np.ndarray
    ) -> int:
        """Analyze shift in water-land boundary"""
        # Apply threshold to find water edge
        _, current_thresh = cv2.threshold(current, 100, 255, cv2.THRESH_BINARY)
        _, baseline_thresh = cv2.threshold(baseline, 100, 255, cv2.THRESH_BINARY)
        
        # Find contours
        current_contours, _ = cv2.findContours(
            current_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        baseline_contours, _ = cv2.findContours(
            baseline_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not current_contours or not baseline_contours:
            return 0
        
        # Get largest contour (likely water boundary)
        current_largest = max(current_contours, key=cv2.contourArea)
        baseline_largest = max(baseline_contours, key=cv2.contourArea)
        
        # Compare centroids
        M1 = cv2.moments(current_largest)
        M2 = cv2.moments(baseline_largest)
        
        if M1["m00"] > 0 and M2["m00"] > 0:
            cx1 = int(M1["m10"] / M1["m00"])
            cx2 = int(M2["m10"] / M2["m00"])
            return cx1 - cx2  # Positive = erosion (bank moved inward)
        
        return 0
    
    def _analyze_single_image(self, image: np.ndarray) -> Dict:
        """Analyze single image for erosion signs (no baseline)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for erosion indicators:
        # 1. Irregular edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Color variation (exposed soil vs vegetation)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brown_mask = cv2.inRange(hsv, (10, 50, 50), (30, 200, 200))
        exposed_soil_pct = np.sum(brown_mask > 0) / brown_mask.size
        
        # Heuristic classification
        erosion_score = edge_density * 0.4 + exposed_soil_pct * 0.6
        
        if erosion_score > 0.4:
            bank_status = "heavy_erosion"
        elif erosion_score > 0.2:
            bank_status = "minor_erosion"
        else:
            bank_status = "stable"
        
        return {
            "bank_status": bank_status,
            "erosion_score": round(erosion_score, 3),
            "exposed_soil_pct": round(exposed_soil_pct * 100, 1),
            "edge_density": round(edge_density, 3),
            "erosion_detected": bank_status != "stable",
            "method": "single_image_heuristic",
            "note": "No baseline available - using heuristic analysis"
        }


def analyze_bank_erosion(
    image_path: str,
    baseline_path: Optional[str] = None,
    bank_roi_polygon: Optional[List[Tuple[int, int]]] = None
) -> Dict:
    """
    Convenience function to analyze bank erosion from image files.
    
    Args:
        image_path: Path to current image
        baseline_path: Optional path to baseline image
        bank_roi_polygon: Optional polygon defining bank region
        
    Returns:
        Erosion analysis results
    """
    if not os.path.exists(image_path):
        return {"bank_status": "unknown", "error": f"File not found: {image_path}"}
    
    current_image = cv2.imread(image_path)
    if current_image is None:
        return {"bank_status": "unknown", "error": "Could not read image"}
    
    baseline_image = None
    if baseline_path and os.path.exists(baseline_path):
        baseline_image = cv2.imread(baseline_path)
    
    analyzer = BankErosionAnalyzer()
    return analyzer.analyze(current_image, baseline_image, bank_roi_polygon)
