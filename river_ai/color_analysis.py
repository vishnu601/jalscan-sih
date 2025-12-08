"""
Water Color Analysis Module
Analyzes water color to detect sediment type, pollution, algae

Classes:
- clear: Clean, blue-green water
- silt: Brown/tan sediment
- muddy: Heavy brown sediment (flood indicator)
- green: Algae bloom
- dark: Deep/murky water
- polluted: Industrial pollution (unusual colors)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


class WaterColorAnalyzer:
    """
    Analyzes water color from river images.
    Uses HSV color space for robust detection.
    """
    
    # Color class definitions (HSV ranges)
    COLOR_CLASSES = {
        "clear": {
            "hsv_low": (85, 30, 80),
            "hsv_high": (130, 180, 255),
            "description": "Clear blue-green water"
        },
        "silt": {
            "hsv_low": (10, 30, 60),
            "hsv_high": (30, 150, 200),
            "description": "Silt-laden brownish water"
        },
        "muddy": {
            "hsv_low": (5, 50, 40),
            "hsv_high": (25, 200, 150),
            "description": "Heavy mud/flood sediment"
        },
        "green": {
            "hsv_low": (35, 40, 40),
            "hsv_high": (85, 255, 255),
            "description": "Algae bloom present"
        },
        "dark": {
            "hsv_low": (0, 0, 0),
            "hsv_high": (180, 100, 80),
            "description": "Dark/murky water"
        },
        "polluted": {
            "hsv_low": (130, 50, 50),
            "hsv_high": (180, 255, 255),
            "description": "Possible industrial pollution"
        }
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize color analyzer.
        
        Args:
            model_path: Optional path to trained ML classifier
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained ML model for color classification"""
        try:
            import joblib
            self.model = joblib.load(model_path)
            logger.info(f"Loaded color classifier from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load color model: {e}")
    
    def analyze(
        self, 
        image: np.ndarray,
        water_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Analyze water color from image.
        
        Args:
            image: BGR image (OpenCV format)
            water_roi: Optional (x, y, width, height) for water region
            
        Returns:
            Dict with color_class, color_index, rgb, hsv, confidence
        """
        if image is None:
            return {"color_class": "unknown", "color_index": 0.5, "error": "No image"}
        
        h, w = image.shape[:2]
        
        # Extract water region (default: bottom third, center portion)
        if water_roi:
            x, y, rw, rh = water_roi
            water_patch = image[y:y+rh, x:x+rw]
        else:
            y_start = int(h * 0.5)
            x_start = int(w * 0.25)
            x_end = int(w * 0.75)
            water_patch = image[y_start:, x_start:x_end]
        
        if water_patch.size == 0:
            return {"color_class": "unknown", "color_index": 0.5, "error": "Empty water patch"}
        
        # Convert to HSV
        hsv = cv2.cvtColor(water_patch, cv2.COLOR_BGR2HSV)
        
        # Compute statistics
        mean_bgr = np.mean(water_patch, axis=(0, 1))
        mean_hsv = np.mean(hsv, axis=(0, 1))
        std_bgr = np.std(water_patch, axis=(0, 1))
        
        # Compute color variance (higher = more turbid)
        variance = np.mean(std_bgr)
        
        # Use ML model if available
        if self.model is not None:
            color_class, confidence = self._classify_with_model(mean_hsv, variance)
        else:
            # Rule-based classification
            color_class, confidence = self._classify_rule_based(mean_hsv, variance)
        
        # Compute color index (0 = clear, 1 = highly turbid/polluted)
        color_index = self._compute_color_index(mean_hsv, variance, color_class)
        
        return {
            "color_class": color_class,
            "color_index": round(color_index, 3),
            "confidence": round(confidence, 2),
            "rgb": [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])],
            "hsv": [int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])],
            "variance": round(variance, 2),
            "method": "ml_classifier" if self.model else "rule_based"
        }
    
    def _classify_rule_based(
        self, 
        mean_hsv: np.ndarray, 
        variance: float
    ) -> Tuple[str, float]:
        """Rule-based color classification using HSV thresholds"""
        h, s, v = mean_hsv
        
        # Check each class
        scores = {}
        
        for class_name, ranges in self.COLOR_CLASSES.items():
            low = np.array(ranges["hsv_low"])
            high = np.array(ranges["hsv_high"])
            
            # Check if mean HSV is in range
            in_range = all(low[i] <= mean_hsv[i] <= high[i] for i in range(3))
            
            if in_range:
                # Compute distance from center of range
                center = (low + high) / 2
                dist = np.linalg.norm(mean_hsv - center)
                max_dist = np.linalg.norm(high - low) / 2
                scores[class_name] = 1 - (dist / max_dist) if max_dist > 0 else 1.0
        
        # Special rules
        if variance > 50:
            scores["muddy"] = scores.get("muddy", 0) + 0.3
        
        if s < 30 and v > 150:
            scores["clear"] = scores.get("clear", 0) + 0.2
        
        if 35 <= h <= 85 and s > 50:
            scores["green"] = scores.get("green", 0) + 0.3
        
        if scores:
            best_class = max(scores, key=scores.get)
            confidence = min(1.0, scores[best_class])
            return best_class, confidence
        
        return "unknown", 0.3
    
    def _classify_with_model(
        self, 
        mean_hsv: np.ndarray, 
        variance: float
    ) -> Tuple[str, float]:
        """ML-based classification using trained model"""
        features = np.array([mean_hsv[0], mean_hsv[1], mean_hsv[2], variance]).reshape(1, -1)
        
        try:
            prediction = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            confidence = max(proba)
            return prediction, confidence
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return self._classify_rule_based(mean_hsv, variance)
    
    def _compute_color_index(
        self, 
        mean_hsv: np.ndarray, 
        variance: float,
        color_class: str
    ) -> float:
        """
        Compute normalized color index (0 = clear, 1 = polluted/flood)
        """
        # Base index from saturation and variance
        s_factor = mean_hsv[1] / 255.0  # Higher saturation = more sediment
        v_factor = 1 - (mean_hsv[2] / 255.0)  # Lower value = darker
        var_factor = min(1.0, variance / 100)  # Higher variance = more turbid
        
        # Class-based adjustments
        class_weights = {
            "clear": 0.0,
            "silt": 0.4,
            "muddy": 0.7,
            "green": 0.5,
            "dark": 0.6,
            "polluted": 1.0,
            "unknown": 0.5
        }
        
        class_weight = class_weights.get(color_class, 0.5)
        
        # Weighted combination
        color_index = 0.2 * s_factor + 0.2 * v_factor + 0.3 * var_factor + 0.3 * class_weight
        
        return min(1.0, max(0.0, color_index))


def analyze_water_color(
    image_path: str,
    water_roi: Optional[Tuple[int, int, int, int]] = None,
    model_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to analyze water color from image file.
    
    Args:
        image_path: Path to image file
        water_roi: Optional ROI for water region
        model_path: Optional path to trained classifier
        
    Returns:
        Color analysis result dictionary
    """
    if not os.path.exists(image_path):
        return {"color_class": "unknown", "error": f"File not found: {image_path}"}
    
    image = cv2.imread(image_path)
    if image is None:
        return {"color_class": "unknown", "error": "Could not read image"}
    
    analyzer = WaterColorAnalyzer(model_path)
    return analyzer.analyze(image, water_roi)
