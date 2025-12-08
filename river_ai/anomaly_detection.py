"""
Anomaly Detection Module
Detects unusual water behavior and sudden changes

Anomaly Types:
- rapid_rise: Water level rising too fast
- rapid_fall: Water level dropping too fast  
- color_change: Sudden water color shift
- flow_spike: Sudden turbulence increase
- combined_alert: Multiple indicators

Output:
- anomaly_score: 0-1 (0 = normal, 1 = severe anomaly)
- anomaly_reason: Human-readable explanation
- anomaly_type: Classification of anomaly
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies by comparing current readings with historical data.
    Uses rule-based detection with configurable thresholds.
    """
    
    # Default thresholds
    THRESHOLDS = {
        "water_level_change_3h": 50.0,      # cm change in 3 hours
        "water_level_change_1h": 30.0,      # cm change in 1 hour
        "water_level_change_24h": 100.0,    # cm change in 24 hours
        "color_index_change": 0.3,          # color index change
        "turbulence_spike": 40,             # turbulence score jump
        "flow_class_jump": 2,               # flow class levels jumped
    }
    
    # Flow class numeric mapping
    FLOW_CLASS_VALUES = {
        "still": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
        "turbulent": 4,
        "unknown": 2
    }
    
    def detect(
        self,
        current_features: Dict,
        historical_features: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Detect anomalies by comparing current with historical data.
        
        Args:
            current_features: Current reading features
            historical_features: List of recent historical readings
            
        Returns:
            Anomaly detection results
        """
        anomalies = []
        scores = []
        
        if not historical_features:
            # No history - can only check for extreme values
            return self._check_extreme_values(current_features)
        
        # Sort historical by timestamp (newest first)
        sorted_history = sorted(
            historical_features,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        # Get readings at different time horizons
        reading_1h = self._get_reading_at_horizon(sorted_history, hours=1)
        reading_3h = self._get_reading_at_horizon(sorted_history, hours=3)
        reading_24h = self._get_reading_at_horizon(sorted_history, hours=24)
        
        current_level = current_features.get("water_level_cm")
        
        # Check water level changes
        if current_level is not None:
            # 1-hour change
            if reading_1h and reading_1h.get("water_level_cm"):
                delta_1h = current_level - reading_1h["water_level_cm"]
                if abs(delta_1h) > self.THRESHOLDS["water_level_change_1h"]:
                    score = min(1.0, abs(delta_1h) / (self.THRESHOLDS["water_level_change_1h"] * 2))
                    anomaly_type = "rapid_rise" if delta_1h > 0 else "rapid_fall"
                    anomalies.append({
                        "type": anomaly_type,
                        "message": f"Water level changed {delta_1h:+.1f}cm in 1 hour",
                        "score": score
                    })
                    scores.append(score)
            
            # 3-hour change
            if reading_3h and reading_3h.get("water_level_cm"):
                delta_3h = current_level - reading_3h["water_level_cm"]
                if abs(delta_3h) > self.THRESHOLDS["water_level_change_3h"]:
                    score = min(1.0, abs(delta_3h) / (self.THRESHOLDS["water_level_change_3h"] * 2))
                    anomaly_type = "rapid_rise" if delta_3h > 0 else "rapid_fall"
                    anomalies.append({
                        "type": anomaly_type,
                        "message": f"Water level changed {delta_3h:+.1f}cm in 3 hours",
                        "score": score
                    })
                    scores.append(score)
        
        # Check color index change
        current_color = current_features.get("color_index", 0)
        if reading_3h:
            prev_color = reading_3h.get("color_index", 0)
            color_delta = abs(current_color - prev_color)
            if color_delta > self.THRESHOLDS["color_index_change"]:
                score = min(1.0, color_delta / 0.6)
                anomalies.append({
                    "type": "color_change",
                    "message": f"Water color index changed by {color_delta:.2f}",
                    "score": score
                })
                scores.append(score)
        
        # Check turbulence spike
        current_turb = current_features.get("turbulence_score", 0)
        if reading_1h:
            prev_turb = reading_1h.get("turbulence_score", 0)
            turb_delta = current_turb - prev_turb
            if turb_delta > self.THRESHOLDS["turbulence_spike"]:
                score = min(1.0, turb_delta / 80)
                anomalies.append({
                    "type": "flow_spike",
                    "message": f"Turbulence increased by {turb_delta} points",
                    "score": score
                })
                scores.append(score)
        
        # Check flow class jump
        current_flow = self.FLOW_CLASS_VALUES.get(
            current_features.get("flow_class", "unknown"), 2
        )
        if reading_1h:
            prev_flow = self.FLOW_CLASS_VALUES.get(
                reading_1h.get("flow_class", "unknown"), 2
            )
            flow_jump = current_flow - prev_flow
            if flow_jump >= self.THRESHOLDS["flow_class_jump"]:
                score = min(1.0, flow_jump / 3)
                anomalies.append({
                    "type": "flow_spike",
                    "message": f"Flow class jumped from {reading_1h.get('flow_class')} to {current_features.get('flow_class')}",
                    "score": score
                })
                scores.append(score)
        
        # Check for combined flash flood indicators
        if (current_features.get("flow_class") == "turbulent" and
            current_features.get("color_class") in ["muddy", "dark"]):
            score = 0.9
            anomalies.append({
                "type": "combined_alert",
                "message": "⚠️ Multiple flash flood indicators: turbulent flow + muddy water",
                "score": score
            })
            scores.append(score)
        
        # Compute overall anomaly score
        if scores:
            anomaly_score = max(scores)  # Take worst anomaly
            anomaly_detected = anomaly_score > 0.3
            primary_anomaly = max(anomalies, key=lambda x: x["score"])
            anomaly_type = primary_anomaly["type"]
            anomaly_reason = "; ".join([a["message"] for a in anomalies])
        else:
            anomaly_score = 0.0
            anomaly_detected = False
            anomaly_type = None
            anomaly_reason = "No anomalies detected"
        
        return {
            "anomaly_detected": anomaly_detected,
            "anomaly_score": round(anomaly_score, 3),
            "anomaly_type": anomaly_type,
            "anomaly_reason": anomaly_reason,
            "anomaly_count": len(anomalies),
            "severity": self._score_to_severity(anomaly_score),
            "all_anomalies": anomalies
        }
    
    def _get_reading_at_horizon(
        self, 
        history: List[Dict], 
        hours: int
    ) -> Optional[Dict]:
        """Get the reading closest to the specified hours ago"""
        if not history:
            return None
        
        # Try to find reading closest to target time
        # For simplicity, take Nth item where N = hours * 4 (assuming ~15min intervals)
        target_index = min(hours * 4, len(history) - 1)
        return history[target_index] if history else None
    
    def _check_extreme_values(self, features: Dict) -> Dict:
        """Check for extreme values when no history is available"""
        anomalies = []
        
        # Check for extreme water level
        level = features.get("water_level_cm")
        if level and level > 500:  # > 5 meters
            anomalies.append({
                "type": "extreme_level",
                "message": f"Extreme water level: {level}cm",
                "score": min(1.0, level / 800)
            })
        
        # Check for turbulent flow
        if features.get("flow_class") == "turbulent":
            anomalies.append({
                "type": "turbulent_flow",
                "message": "Turbulent water flow detected",
                "score": 0.7
            })
        
        # Check for pollution
        if features.get("color_class") == "polluted":
            anomalies.append({
                "type": "pollution",
                "message": "Possible water pollution detected",
                "score": 0.6
            })
        
        if anomalies:
            max_anomaly = max(anomalies, key=lambda x: x["score"])
            return {
                "anomaly_detected": True,
                "anomaly_score": max_anomaly["score"],
                "anomaly_type": max_anomaly["type"],
                "anomaly_reason": "; ".join([a["message"] for a in anomalies]),
                "severity": self._score_to_severity(max_anomaly["score"]),
                "all_anomalies": anomalies
            }
        
        return {
            "anomaly_detected": False,
            "anomaly_score": 0.0,
            "anomaly_type": None,
            "anomaly_reason": "No anomalies detected (no historical data for comparison)",
            "severity": "low",
            "all_anomalies": []
        }
    
    def _score_to_severity(self, score: float) -> str:
        """Convert anomaly score to severity level"""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"


def detect_anomalies(
    current_features: Dict,
    historical_features: Optional[List[Dict]] = None
) -> Dict:
    """
    Convenience function for anomaly detection.
    
    Args:
        current_features: Current reading features
        historical_features: List of historical readings
        
    Returns:
        Anomaly detection results
    """
    detector = AnomalyDetector()
    return detector.detect(current_features, historical_features)
