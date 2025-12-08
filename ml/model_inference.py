"""
JalScan Flood Prediction - Model Inference
Load trained model and run predictions
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import joblib

from .schemas import (
    SiteFeatures, PredictionRequest, PredictionResponse,
    RiskCategory, LABEL_TO_RISK_CATEGORY
)
from .data_pipeline import FloodDataPipeline

logger = logging.getLogger(__name__)

ML_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = ML_DIR / "models" / "flood_model.joblib"


class FloodPredictor:
    """
    Load trained model and run flood risk predictions.
    Thread-safe singleton for production use.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = None
        self.pipeline = FloodDataPipeline()
        self._initialized = True
        
    def load_model(self, model_path: str = None) -> bool:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to model artifact (default: ml/models/flood_model.joblib)
            
        Returns:
            True if loaded successfully
        """
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Using rule-based fallback.")
            return False
        
        try:
            artifact = joblib.load(model_path)
            self.model = artifact["model"]
            self.scaler = artifact["scaler"]
            self.feature_names = artifact["feature_names"]
            self.model_version = artifact.get("version", "1.0.0")
            logger.info(f"Loaded flood prediction model v{self.model_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(
        self,
        request: PredictionRequest,
        app_context=None
    ) -> PredictionResponse:
        """
        Run flood risk prediction for a monitoring site.
        
        Args:
            request: PredictionRequest with site_id and optional timestamp
            app_context: Flask app context (for DB access)
            
        Returns:
            PredictionResponse with risk category, score, and explanations
        """
        from models import MonitoringSite
        
        timestamp = request.timestamp or datetime.utcnow()
        
        # Get site info
        site = MonitoringSite.query.get(request.monitoring_site_id)
        if not site:
            return self._error_response(
                request.monitoring_site_id,
                "Unknown Site",
                timestamp,
                "Site not found"
            )
        
        # Extract features
        features = self.pipeline.get_site_features(
            request.monitoring_site_id,
            at_time=timestamp
        )
        
        if features is None:
            return self._error_response(
                request.monitoring_site_id,
                site.name,
                timestamp,
                "Insufficient data for prediction"
            )
        
        # Run prediction
        if self.model is not None:
            return self._ml_predict(features, site)
        else:
            return self._rule_based_predict(features, site)
    
    def _ml_predict(
        self,
        features: SiteFeatures,
        site
    ) -> PredictionResponse:
        """
        ML-based prediction using trained XGBoost model.
        """
        # Prepare features
        X = np.array([features.to_feature_vector()])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.model.predict(X_scaled)[0]
        y_proba = self.model.predict_proba(X_scaled)[0]
        
        risk_category = LABEL_TO_RISK_CATEGORY[y_pred]
        risk_score = float(y_proba[y_pred])
        confidence = float(np.max(y_proba))
        
        # Get feature importances for explanation
        importances = self.model.feature_importances_
        feature_values = features.to_feature_vector()
        
        key_factors = {}
        for i, (name, imp, val) in enumerate(sorted(
            zip(self.feature_names, importances, feature_values),
            key=lambda x: x[1],
            reverse=True
        )[:5]):
            key_factors[name] = float(val)
        
        # Generate explanations
        explanations = self._generate_explanations(features, risk_category, key_factors)
        recommendations = self._generate_recommendations(risk_category, features)
        
        return PredictionResponse(
            monitoring_site_id=features.site_id,
            site_name=features.site_name,
            timestamp=features.timestamp,
            risk_category=risk_category,
            risk_score=risk_score,
            confidence=confidence,
            horizon_hours=6,
            explanations=explanations,
            key_factors=key_factors,
            recommendations=recommendations,
            model_version=self.model_version or "rule-based"
        )
    
    def _rule_based_predict(
        self,
        features: SiteFeatures,
        site
    ) -> PredictionResponse:
        """
        Rule-based fallback when no ML model is available.
        Uses thresholds and rate of change.
        """
        # Determine risk based on thresholds and dynamics
        pct_danger = features.pct_of_danger_threshold
        pct_alert = features.pct_of_alert_threshold
        delta_1h = features.delta_1h
        slope = features.slope_1h
        
        # Flash flood check: rapid rise > 50cm/hour
        if slope > 50:
            risk_category = RiskCategory.FLASH_FLOOD_RISK
            risk_score = min(0.9, 0.5 + (slope / 100))
        # Flood risk: above danger threshold
        elif pct_danger >= 100:
            risk_category = RiskCategory.FLOOD_RISK
            risk_score = min(0.95, pct_danger / 150)
        # Caution: above alert threshold or rising fast
        elif pct_alert >= 100 or delta_1h > 30:
            risk_category = RiskCategory.CAUTION
            risk_score = min(0.7, max(pct_alert / 150, delta_1h / 50))
        # Safe
        else:
            risk_category = RiskCategory.SAFE
            risk_score = max(0.1, pct_alert / 200)
        
        confidence = 0.7  # Lower confidence for rule-based
        
        key_factors = {
            "water_level_cm": features.water_level_cm,
            "pct_of_danger_threshold": pct_danger,
            "delta_1h": delta_1h,
            "slope_1h": slope
        }
        
        explanations = self._generate_explanations(features, risk_category, key_factors)
        recommendations = self._generate_recommendations(risk_category, features)
        
        return PredictionResponse(
            monitoring_site_id=features.site_id,
            site_name=features.site_name,
            timestamp=features.timestamp,
            risk_category=risk_category,
            risk_score=risk_score,
            confidence=confidence,
            horizon_hours=6,
            explanations=explanations,
            key_factors=key_factors,
            recommendations=recommendations,
            model_version="rule-based-1.0"
        )
    
    def _generate_explanations(
        self,
        features: SiteFeatures,
        risk_category: RiskCategory,
        key_factors: Dict
    ) -> List[str]:
        """Generate human-readable explanations for the prediction"""
        explanations = []
        
        # Water level status
        level_cm = features.water_level_cm
        if level_cm > 0:
            explanations.append(f"Current water level is {level_cm:.0f} cm")
        
        # Threshold proximity
        if features.pct_of_danger_threshold >= 100:
            explanations.append("Water level has exceeded the danger threshold")
        elif features.pct_of_danger_threshold >= 80:
            explanations.append(f"Water level is at {features.pct_of_danger_threshold:.0f}% of danger threshold")
        elif features.pct_of_alert_threshold >= 100:
            explanations.append("Water level is above alert threshold")
        
        # Rate of change
        if features.delta_1h > 30:
            explanations.append(f"Water level rose by {features.delta_1h:.0f} cm in the last hour")
        elif features.delta_1h < -20:
            explanations.append(f"Water level fell by {abs(features.delta_1h):.0f} cm in the last hour")
        
        # Flash flood warning
        if features.slope_1h > 50:
            explanations.append("⚠️ Rapid water level rise detected - flash flood risk")
        
        # Monsoon context
        if features.is_monsoon:
            explanations.append("Currently in monsoon season - elevated baseline risk")
        
        if not explanations:
            if risk_category == RiskCategory.SAFE:
                explanations.append("Water levels are within normal range")
            else:
                explanations.append("Multiple risk factors detected")
        
        return explanations
    
    def _generate_recommendations(
        self,
        risk_category: RiskCategory,
        features: SiteFeatures
    ) -> List[str]:
        """Generate action recommendations based on risk level"""
        if risk_category == RiskCategory.FLASH_FLOOD_RISK:
            return [
                "🚨 IMMEDIATE ACTION REQUIRED",
                "Move to higher ground immediately",
                "Avoid low-lying areas and river banks",
                "Alert emergency services if trapped",
                "Do not attempt to cross flooded areas"
            ]
        elif risk_category == RiskCategory.FLOOD_RISK:
            return [
                "⚠️ HIGH RISK - Prepare for potential flooding",
                "Move valuables to higher floors",
                "Prepare emergency supplies and documents",
                "Stay tuned to official alerts",
                "Be ready to evacuate if advised"
            ]
        elif risk_category == RiskCategory.CAUTION:
            return [
                "Monitor water levels closely",
                "Avoid unnecessary travel near rivers",
                "Keep emergency contacts handy",
                "Check for updates every few hours"
            ]
        else:  # SAFE
            return [
                "No immediate action required",
                "Continue regular monitoring",
                "Stay aware of weather forecasts"
            ]
    
    def _error_response(
        self,
        site_id: int,
        site_name: str,
        timestamp: datetime,
        error_msg: str
    ) -> PredictionResponse:
        """Create error response"""
        return PredictionResponse(
            monitoring_site_id=site_id,
            site_name=site_name,
            timestamp=timestamp,
            risk_category=RiskCategory.SAFE,
            risk_score=0.0,
            confidence=0.0,
            horizon_hours=6,
            explanations=[f"Prediction unavailable: {error_msg}"],
            key_factors={},
            recommendations=["Contact system administrator"],
            model_version="error"
        )
    
    def get_all_site_risks(self) -> List[Dict]:
        """
        Get risk predictions for all active monitoring sites.
        Used by dashboard.
        """
        from models import MonitoringSite
        
        sites = MonitoringSite.query.filter_by(is_active=True).all()
        results = []
        
        for site in sites:
            try:
                request = PredictionRequest(monitoring_site_id=site.id)
                response = self.predict(request)
                results.append(response.to_dict())
            except Exception as e:
                logger.error(f"Error predicting for site {site.id}: {e}")
                results.append({
                    "monitoring_site_id": site.id,
                    "site_name": site.name,
                    "risk_category": "UNKNOWN",
                    "error": str(e)
                })
        
        return results


# Singleton instance
predictor = FloodPredictor()


def get_predictor() -> FloodPredictor:
    """Get the singleton predictor instance"""
    if predictor.model is None:
        predictor.load_model()
    return predictor
