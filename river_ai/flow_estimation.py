"""
Flow Speed Estimation Module
Estimates water flow speed from surface texture and optical flow

Classes:
- still: Very slow or stagnant water
- low: Gentle flow
- moderate: Normal river flow
- high: Fast-moving water
- turbulent: Turbulent (flash flood precursor)
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class FlowEstimator:
    """
    Estimates water flow speed from images.
    Uses optical flow (if multiple frames) or texture analysis (single frame).
    """
    
    # Flow class thresholds (based on optical flow magnitude)
    FLOW_THRESHOLDS = {
        "still": (0, 2),
        "low": (2, 8),
        "moderate": (8, 20),
        "high": (20, 40),
        "turbulent": (40, float('inf'))
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize flow estimator.
        
        Args:
            model_path: Optional path to trained flow classifier
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained ML model"""
        try:
            import joblib
            self.model = joblib.load(model_path)
            logger.info(f"Loaded flow classifier from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load flow model: {e}")
    
    def estimate_from_frames(
        self, 
        frames: List[np.ndarray],
        water_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Estimate flow from multiple consecutive frames using optical flow.
        
        Args:
            frames: List of BGR images (at least 2)
            water_roi: Optional water region
            
        Returns:
            Flow estimation results
        """
        if len(frames) < 2:
            return self.estimate_from_texture(frames[0] if frames else None, water_roi)
        
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        
        # Extract water region
        if water_roi:
            x, y, w, h = water_roi
            gray_frames = [g[y:y+h, x:x+w] for g in gray_frames]
        else:
            # Use bottom half for water
            height = gray_frames[0].shape[0]
            gray_frames = [g[height//2:, :] for g in gray_frames]
        
        # Compute optical flow between consecutive frames
        flow_magnitudes = []
        flow_angles = []
        
        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Compute magnitude and angle
            mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
            flow_magnitudes.append(np.mean(mag))
            flow_angles.append(np.mean(ang))
        
        # Aggregate statistics
        mean_magnitude = np.mean(flow_magnitudes)
        std_magnitude = np.std(flow_magnitudes)
        mean_angle = np.mean(flow_angles)
        direction_coherence = 1.0 - (np.std(flow_angles) / np.pi)
        
        # Classify flow
        flow_class = self._classify_flow(mean_magnitude, std_magnitude)
        
        # Compute turbulence score (0-100)
        turbulence_score = min(100, int(std_magnitude * 5 + mean_magnitude * 2))
        
        return {
            "flow_class": flow_class,
            "flow_magnitude": round(mean_magnitude, 2),
            "flow_std": round(std_magnitude, 2),
            "direction_coherence": round(direction_coherence, 2),
            "turbulence_score": turbulence_score,
            "flash_flood_risk": "high" if flow_class == "turbulent" else "medium" if flow_class == "high" else "low",
            "confidence": 0.8,
            "method": "optical_flow",
            "frames_analyzed": len(frames)
        }
    
    def estimate_from_texture(
        self, 
        image: np.ndarray,
        water_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Estimate flow from single image using texture analysis.
        Less accurate than optical flow but works with single frame.
        
        Args:
            image: BGR image
            water_roi: Optional water region
            
        Returns:
            Flow estimation results
        """
        if image is None:
            return {"flow_class": "unknown", "confidence": 0, "error": "No image"}
        
        h, w = image.shape[:2]
        
        # Extract water region
        if water_roi:
            x, y, rw, rh = water_roi
            water_patch = image[y:y+rh, x:x+rw]
        else:
            water_patch = image[h//2:, :]
        
        if water_patch.size == 0:
            return {"flow_class": "unknown", "confidence": 0, "error": "Empty patch"}
        
        # Convert to grayscale
        gray = cv2.cvtColor(water_patch, cv2.COLOR_BGR2GRAY)
        
        # Texture analysis using multiple methods
        
        # 1. Laplacian variance (edge sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # 2. Gabor filter response (texture orientation)
        gabor_responses = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            gabor_responses.append(np.std(filtered))
        gabor_energy = np.mean(gabor_responses)
        gabor_anisotropy = np.std(gabor_responses) / (np.mean(gabor_responses) + 1e-6)
        
        # 3. Local Binary Pattern-like texture (simplified)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        texture_complexity = np.mean(gradient_mag)
        
        # Combine features
        texture_score = (laplacian_var / 500) + (gabor_energy / 50) + (texture_complexity / 30)
        
        # Classify based on texture score
        if texture_score < 0.5:
            flow_class = "still"
        elif texture_score < 1.5:
            flow_class = "low"
        elif texture_score < 3.0:
            flow_class = "moderate"
        elif texture_score < 5.0:
            flow_class = "high"
        else:
            flow_class = "turbulent"
        
        # Compute turbulence score
        turbulence_score = min(100, int(texture_score * 15))
        
        return {
            "flow_class": flow_class,
            "texture_score": round(texture_score, 2),
            "laplacian_variance": round(laplacian_var, 2),
            "gabor_energy": round(gabor_energy, 2),
            "turbulence_score": turbulence_score,
            "flash_flood_risk": "high" if flow_class == "turbulent" else "medium" if flow_class == "high" else "low",
            "confidence": 0.5,  # Lower confidence for single-frame
            "method": "texture_analysis"
        }
    
    def _classify_flow(self, magnitude: float, std: float) -> str:
        """Classify flow based on optical flow magnitude"""
        for flow_class, (low, high) in self.FLOW_THRESHOLDS.items():
            if low <= magnitude < high:
                return flow_class
        return "unknown"


def estimate_flow_speed(
    image_path: str,
    additional_frames: Optional[List[str]] = None,
    water_roi: Optional[Tuple[int, int, int, int]] = None,
    model_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to estimate flow from image file(s).
    
    Args:
        image_path: Path to main image
        additional_frames: Optional list of paths to additional frames
        water_roi: Optional water region
        model_path: Optional path to trained classifier
        
    Returns:
        Flow estimation results
    """
    if not os.path.exists(image_path):
        return {"flow_class": "unknown", "error": f"File not found: {image_path}"}
    
    image = cv2.imread(image_path)
    if image is None:
        return {"flow_class": "unknown", "error": "Could not read image"}
    
    estimator = FlowEstimator(model_path)
    
    if additional_frames:
        frames = [image]
        for frame_path in additional_frames:
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
        
        if len(frames) >= 2:
            return estimator.estimate_from_frames(frames, water_roi)
    
    return estimator.estimate_from_texture(image, water_roi)
