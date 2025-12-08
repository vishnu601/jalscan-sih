"""
JalScan Flood Prediction - Data Schemas
Pydantic models for inputs/outputs
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RiskCategory(str, Enum):
    """Flood risk categories"""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    FLOOD_RISK = "FLOOD_RISK"
    FLASH_FLOOD_RISK = "FLASH_FLOOD_RISK"


# Numeric mapping for model training
RISK_CATEGORY_LABELS = {
    RiskCategory.SAFE: 0,
    RiskCategory.CAUTION: 1,
    RiskCategory.FLOOD_RISK: 2,
    RiskCategory.FLASH_FLOOD_RISK: 3
}

LABEL_TO_RISK_CATEGORY = {v: k for k, v in RISK_CATEGORY_LABELS.items()}


@dataclass
class SiteFeatures:
    """Features extracted for a monitoring site at a given time"""
    site_id: int
    site_name: str
    timestamp: datetime
    
    # Current water level
    water_level_cm: float
    pct_of_danger_threshold: float
    pct_of_alert_threshold: float
    
    # Temporal features
    hour: int
    day_of_week: int
    month: int
    is_monsoon: bool  # June-September
    
    # Water level dynamics (deltas)
    delta_1h: float = 0.0
    delta_3h: float = 0.0
    delta_6h: float = 0.0
    delta_12h: float = 0.0
    delta_24h: float = 0.0
    
    # Rate of change (slope, acceleration)
    slope_1h: float = 0.0  # cm per hour
    acceleration: float = 0.0  # change in slope
    
    # Aggregates over past 24 hours
    level_mean_24h: float = 0.0
    level_max_24h: float = 0.0
    level_min_24h: float = 0.0
    level_std_24h: float = 0.0
    submission_count_24h: int = 0
    
    # Site attributes
    site_flood_history_count: int = 0  # Historical flood events at this site
    river_type_encoded: int = 0  # 0=unknown, 1=major, 2=minor, 3=tributary
    
    # Weather features (stubbed - will integrate with external API)
    rainfall_last_3h: float = 0.0
    rainfall_last_24h: float = 0.0
    forecast_rainfall_6h: float = 0.0
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for ML model"""
        return [
            self.water_level_cm,
            self.pct_of_danger_threshold,
            self.pct_of_alert_threshold,
            float(self.hour),
            float(self.day_of_week),
            float(self.month),
            float(self.is_monsoon),
            self.delta_1h,
            self.delta_3h,
            self.delta_6h,
            self.delta_12h,
            self.delta_24h,
            self.slope_1h,
            self.acceleration,
            self.level_mean_24h,
            self.level_max_24h,
            self.level_min_24h,
            self.level_std_24h,
            float(self.submission_count_24h),
            float(self.site_flood_history_count),
            float(self.river_type_encoded),
            self.rainfall_last_3h,
            self.rainfall_last_24h,
            self.forecast_rainfall_6h
        ]
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for model interpretation"""
        return [
            "water_level_cm",
            "pct_of_danger_threshold",
            "pct_of_alert_threshold",
            "hour",
            "day_of_week",
            "month",
            "is_monsoon",
            "delta_1h",
            "delta_3h",
            "delta_6h",
            "delta_12h",
            "delta_24h",
            "slope_1h",
            "acceleration",
            "level_mean_24h",
            "level_max_24h",
            "level_min_24h",
            "level_std_24h",
            "submission_count_24h",
            "site_flood_history_count",
            "river_type_encoded",
            "rainfall_last_3h",
            "rainfall_last_24h",
            "forecast_rainfall_6h"
        ]


@dataclass
class PredictionRequest:
    """Request for flood risk prediction"""
    monitoring_site_id: int
    timestamp: Optional[datetime] = None  # Default: now
    time_window_hours: int = 24  # History to consider


@dataclass 
class PredictionResponse:
    """Response from flood risk prediction"""
    monitoring_site_id: int
    site_name: str
    timestamp: datetime
    
    # Prediction results
    risk_category: RiskCategory
    risk_score: float  # 0-1 probability
    confidence: float  # Model confidence
    horizon_hours: int  # Prediction window (default 6)
    
    # Explanations
    explanations: List[str] = field(default_factory=list)
    key_factors: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Model metadata
    model_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        return {
            "monitoring_site_id": self.monitoring_site_id,
            "site_name": self.site_name,
            "timestamp": self.timestamp.isoformat(),
            "risk_category": self.risk_category.value,
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "horizon_hours": self.horizon_hours,
            "explanations": self.explanations,
            "key_factors": {k: round(v, 4) for k, v in self.key_factors.items()},
            "recommendations": self.recommendations,
            "model_version": self.model_version
        }


@dataclass
class TrainingExample:
    """A labeled training example"""
    features: SiteFeatures
    label: int  # 0-3 corresponding to RiskCategory
    weight: float = 1.0  # Sample weight for class balancing
