"""
River Memory AI - Main Pipeline
Orchestrates the full computer vision analysis pipeline

This module coordinates:
1. Gauge detection (water level)
2. Color analysis (sediment, pollution)
3. Flow estimation (speed, turbulence)
4. Gauge health (algae, damage)
5. Bank erosion (change detection)
6. Anomaly detection (alerts)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

import cv2
import numpy as np

from .gauge_detection import GaugeDetector, detect_water_level
from .color_analysis import WaterColorAnalyzer, analyze_water_color
from .flow_estimation import FlowEstimator, estimate_flow_speed
from .gauge_health import GaugeHealthAnalyzer, analyze_gauge_health
from .bank_erosion import BankErosionAnalyzer, analyze_bank_erosion
from .anomaly_detection import AnomalyDetector, detect_anomalies

logger = logging.getLogger(__name__)


class RiverMemoryPipeline:
    """
    Main pipeline for River Memory AI.
    Coordinates all analysis modules and persists results.
    """
    
    def __init__(self, db_session=None, models_dir: str = "river_ai/models"):
        """
        Initialize the pipeline.
        
        Args:
            db_session: SQLAlchemy database session
            models_dir: Directory containing trained ML models
        """
        self.db_session = db_session
        self.models_dir = models_dir
        
        # Initialize analyzers
        self.gauge_detector = GaugeDetector()
        self.color_analyzer = WaterColorAnalyzer(
            model_path=os.path.join(models_dir, "color_classifier.joblib")
        )
        self.flow_estimator = FlowEstimator(
            model_path=os.path.join(models_dir, "flow_classifier.joblib")
        )
        self.gauge_health_analyzer = GaugeHealthAnalyzer()
        self.bank_erosion_analyzer = BankErosionAnalyzer()
        self.anomaly_detector = AnomalyDetector()
    
    def process_image(
        self,
        image_path: str,
        site_id: str,
        site_config: Optional[Dict] = None,
        manual_water_level: Optional[float] = None,
        captured_at: Optional[datetime] = None
    ) -> Dict:
        """
        Process a river image through the full analysis pipeline.
        
        Args:
            image_path: Path to the image file
            site_id: Site identifier
            site_config: Site configuration (calibration, ROIs)
            manual_water_level: Optional manually reported water level
            captured_at: Timestamp when image was captured
            
        Returns:
            Complete analysis results dictionary
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image"}
        
        timestamp = captured_at or datetime.utcnow()
        
        # Default site config
        if site_config is None:
            site_config = {}
        
        # Extract ROIs from config
        gauge_roi = site_config.get("gauge_roi")
        water_roi = site_config.get("water_roi")
        bank_polygon = site_config.get("bank_roi_polygon")
        
        # ========== 1. Gauge Detection ==========
        logger.info(f"Analyzing gauge for site {site_id}")
        
        if manual_water_level is not None:
            gauge_result = {
                "water_level_cm": manual_water_level,
                "confidence": 1.0,
                "method": "manual"
            }
        else:
            self.gauge_detector.pixels_per_cm = site_config.get(
                "gauge_calibration_pixels_per_cm", 10.0
            )
            self.gauge_detector.gauge_zero_y = site_config.get(
                "gauge_zero_pixel_y", image.shape[0] // 2
            )
            gauge_result = self.gauge_detector.detect_water_level(image, gauge_roi)
        
        # ========== 2. Color Analysis ==========
        logger.info(f"Analyzing water color for site {site_id}")
        color_result = self.color_analyzer.analyze(image, water_roi)
        
        # ========== 3. Flow Estimation ==========
        logger.info(f"Estimating flow for site {site_id}")
        flow_result = self.flow_estimator.estimate_from_texture(image, water_roi)
        
        # ========== 4. Gauge Health ==========
        logger.info(f"Analyzing gauge health for site {site_id}")
        health_result = self.gauge_health_analyzer.analyze(image, gauge_roi)
        
        # ========== 5. Bank Erosion ==========
        logger.info(f"Analyzing bank erosion for site {site_id}")
        baseline_image = self._get_baseline_image(site_id)
        erosion_result = self.bank_erosion_analyzer.analyze(
            image, baseline_image, bank_polygon
        )
        
        # ========== 6. Compile Current Features ==========
        current_features = {
            "water_level_cm": gauge_result.get("water_level_cm"),
            "flow_class": flow_result.get("flow_class"),
            "color_class": color_result.get("color_class"),
            "color_index": color_result.get("color_index", 0),
            "turbulence_score": flow_result.get("turbulence_score", 0),
            "timestamp": timestamp.isoformat()
        }
        
        # ========== 7. Anomaly Detection ==========
        logger.info(f"Checking for anomalies at site {site_id}")
        historical = self._get_historical_features(site_id, hours=24)
        anomaly_result = self.anomaly_detector.detect(current_features, historical)
        
        # ========== 8. Compute Overall Risk ==========
        overall_risk = self._compute_overall_risk(
            gauge_result, color_result, flow_result, 
            health_result, erosion_result, anomaly_result
        )
        
        # ========== Compile Results ==========
        result = {
            "site_id": site_id,
            "image_path": image_path,
            "timestamp": timestamp.isoformat(),
            
            # Water Level
            "water_level_cm": gauge_result.get("water_level_cm"),
            "water_level_confidence": gauge_result.get("confidence", 0),
            
            # Flow
            "flow_class": flow_result.get("flow_class", "unknown"),
            "turbulence_score": flow_result.get("turbulence_score", 0),
            "flash_flood_risk": flow_result.get("flash_flood_risk", "low"),
            
            # Color
            "color_class": color_result.get("color_class", "unknown"),
            "color_index": color_result.get("color_index", 0),
            "rgb": color_result.get("rgb"),
            
            # Gauge Health
            "gauge_algae_present": health_result.get("gauge_algae_present", False),
            "gauge_faded": health_result.get("gauge_faded", False),
            "gauge_broken": health_result.get("gauge_broken", False),
            "gauge_tilt_angle": health_result.get("gauge_tilt_angle", 0),
            "gauge_visibility_score": health_result.get("gauge_visibility_score", 100),
            
            # Bank Erosion
            "bank_status": erosion_result.get("bank_status", "unknown"),
            "erosion_change_pct": erosion_result.get("erosion_change_pct", 0),
            
            # Anomaly
            "anomaly_detected": anomaly_result.get("anomaly_detected", False),
            "anomaly_score": anomaly_result.get("anomaly_score", 0),
            "anomaly_type": anomaly_result.get("anomaly_type"),
            "anomaly_reason": anomaly_result.get("anomaly_reason"),
            
            # Overall
            "overall_risk": overall_risk,
            "analysis_version": "1.0.0",
            
            # Raw results for debugging
            "raw": {
                "gauge": gauge_result,
                "color": color_result,
                "flow": flow_result,
                "health": health_result,
                "erosion": erosion_result,
                "anomaly": anomaly_result
            }
        }
        
        return result
    
    def _compute_overall_risk(
        self,
        gauge: Dict,
        color: Dict,
        flow: Dict,
        health: Dict,
        erosion: Dict,
        anomaly: Dict
    ) -> str:
        """Compute overall risk level from all analyses"""
        risk_score = 0
        
        # Water level risk
        level = gauge.get("water_level_cm")
        if level:
            if level > 400:
                risk_score += 0.4
            elif level > 300:
                risk_score += 0.2
        
        # Flow risk
        flow_class = flow.get("flow_class", "")
        if flow_class == "turbulent":
            risk_score += 0.3
        elif flow_class == "high":
            risk_score += 0.15
        
        # Color risk (muddy = possible flood)
        if color.get("color_class") in ["muddy", "polluted"]:
            risk_score += 0.15
        
        # Anomaly risk
        risk_score += anomaly.get("anomaly_score", 0) * 0.3
        
        # Erosion risk
        if erosion.get("bank_status") == "heavy_erosion":
            risk_score += 0.1
        
        # Classify
        if risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.3:
            return "medium"
        else:
            return "low"
    
    def _get_baseline_image(self, site_id: str) -> Optional[np.ndarray]:
        """Get baseline image for erosion comparison"""
        # TODO: Query from database for first image of site
        # For now, return None (single-image analysis)
        return None
    
    def _get_historical_features(
        self, 
        site_id: str, 
        hours: int = 24
    ) -> list:
        """Get historical features for anomaly detection"""
        if not self.db_session:
            return []
        
        try:
            from models import RiverAnalysis
            
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            analyses = RiverAnalysis.query.filter(
                RiverAnalysis.site_id == int(site_id),
                RiverAnalysis.timestamp >= cutoff
            ).order_by(RiverAnalysis.timestamp.desc()).all()
            
            return [
                {
                    "water_level_cm": a.water_level_cm if hasattr(a, 'water_level_cm') else None,
                    "flow_class": a.flow_speed_class,
                    "color_class": a.sediment_type,
                    "color_index": a.pollution_index,
                    "turbulence_score": a.turbulence_score,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in analyses
            ]
        except Exception as e:
            logger.warning(f"Could not fetch historical data: {e}")
            return []


def process_image_for_site(
    image_path: str,
    site_id: str,
    db_session=None,
    site_config: Optional[Dict] = None,
    manual_water_level: Optional[float] = None
) -> Dict:
    """
    Main entry point for processing a river image.
    
    Args:
        image_path: Path to image file
        site_id: Site identifier
        db_session: Optional database session
        site_config: Optional site configuration
        manual_water_level: Optional manual water level reading
        
    Returns:
        Complete analysis results
    """
    pipeline = RiverMemoryPipeline(db_session)
    result = pipeline.process_image(
        image_path, site_id, site_config, manual_water_level
    )
    
    # Persist to database if session provided
    if db_session and "error" not in result:
        try:
            from models import RiverAnalysis
            import json
            
            analysis = RiverAnalysis(
                site_id=int(site_id),
                timestamp=datetime.fromisoformat(result["timestamp"]),
                water_color_rgb=json.dumps(result.get("rgb")),
                sediment_type=result.get("color_class"),
                pollution_index=result.get("color_index", 0),
                flow_speed_class=result.get("flow_class"),
                turbulence_score=result.get("turbulence_score", 0),
                gauge_visibility_score=result.get("gauge_visibility_score", 100),
                gauge_damage_detected=result.get("gauge_broken", False),
                damage_type=",".join(filter(None, [
                    "algae" if result.get("gauge_algae_present") else None,
                    "faded" if result.get("gauge_faded") else None,
                    "broken" if result.get("gauge_broken") else None
                ])),
                anomaly_detected=result.get("anomaly_detected", False),
                anomaly_type=result.get("anomaly_type"),
                anomaly_description=result.get("anomaly_reason"),
                erosion_detected=result.get("bank_status") in ["minor_erosion", "heavy_erosion"],
                overall_risk=result.get("overall_risk", "low"),
                ai_analysis_json=json.dumps(result)
            )
            db_session.add(analysis)
            db_session.commit()
            
            result["analysis_id"] = analysis.id
            result["stored"] = True
            
        except Exception as e:
            logger.error(f"Failed to store analysis: {e}")
            result["stored"] = False
            result["store_error"] = str(e)
    
    return result
