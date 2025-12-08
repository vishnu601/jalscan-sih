"""
JalScan Flood Prediction - Data Pipeline
Extract features from database for training and inference
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np

from .schemas import (
    SiteFeatures, TrainingExample, RiskCategory, 
    RISK_CATEGORY_LABELS, LABEL_TO_RISK_CATEGORY
)

logger = logging.getLogger(__name__)


class FloodDataPipeline:
    """
    Data pipeline for extracting features from JalScan database.
    Handles both training data generation and real-time inference.
    """
    
    def __init__(self, app=None):
        """Initialize with Flask app context"""
        self.app = app
        
    def get_site_features(
        self, 
        site_id: int, 
        at_time: Optional[datetime] = None,
        db_session=None
    ) -> Optional[SiteFeatures]:
        """
        Extract features for a site at a given time.
        Used for both training and inference.
        """
        from models import MonitoringSite, WaterLevelSubmission
        
        if at_time is None:
            at_time = datetime.utcnow()
            
        # Get site info
        site = MonitoringSite.query.get(site_id)
        if not site:
            logger.warning(f"Site {site_id} not found")
            return None
        
        # Get recent submissions for this site
        time_24h_ago = at_time - timedelta(hours=24)
        time_12h_ago = at_time - timedelta(hours=12)
        time_6h_ago = at_time - timedelta(hours=6)
        time_3h_ago = at_time - timedelta(hours=3)
        time_1h_ago = at_time - timedelta(hours=1)
        
        # Query submissions in the 24h window
        submissions = WaterLevelSubmission.query.filter(
            WaterLevelSubmission.site_id == site_id,
            WaterLevelSubmission.timestamp >= time_24h_ago,
            WaterLevelSubmission.timestamp <= at_time
        ).order_by(WaterLevelSubmission.timestamp.desc()).all()
        
        if not submissions:
            # No recent data - return features with current level as 0
            return self._create_empty_features(site, at_time)
        
        # Current water level (most recent submission)
        current_level = submissions[0].water_level * 100  # Convert m to cm
        
        # Calculate thresholds (default if not set)
        danger_threshold = getattr(site, 'danger_level', None) or 500  # 5m default
        alert_threshold = getattr(site, 'alert_level', None) or 300   # 3m default
        
        # Calculate deltas
        levels = [(s.timestamp, s.water_level * 100) for s in submissions]
        
        delta_1h = self._calc_delta(levels, current_level, time_1h_ago)
        delta_3h = self._calc_delta(levels, current_level, time_3h_ago)
        delta_6h = self._calc_delta(levels, current_level, time_6h_ago)
        delta_12h = self._calc_delta(levels, current_level, time_12h_ago)
        delta_24h = self._calc_delta(levels, current_level, time_24h_ago)
        
        # Calculate slope and acceleration
        slope_1h = delta_1h  # cm per hour
        prev_slope = self._calc_slope(levels, time_1h_ago, time_3h_ago)
        acceleration = slope_1h - prev_slope
        
        # Aggregates
        level_values = [l[1] for l in levels]
        level_mean_24h = np.mean(level_values) if level_values else 0
        level_max_24h = np.max(level_values) if level_values else 0
        level_min_24h = np.min(level_values) if level_values else 0
        level_std_24h = np.std(level_values) if len(level_values) > 1 else 0
        
        # Temporal features
        is_monsoon = at_time.month in [6, 7, 8, 9]
        
        # Site history (count of past flood-like events)
        flood_history = self._get_flood_history_count(site_id)
        
        # Weather features (stubbed - integrate with external API later)
        rainfall_last_3h, rainfall_last_24h, forecast_6h = self._get_weather_features(
            site.latitude, site.longitude, at_time
        )
        
        return SiteFeatures(
            site_id=site_id,
            site_name=site.name,
            timestamp=at_time,
            water_level_cm=current_level,
            pct_of_danger_threshold=(current_level / danger_threshold) * 100 if danger_threshold > 0 else 0,
            pct_of_alert_threshold=(current_level / alert_threshold) * 100 if alert_threshold > 0 else 0,
            hour=at_time.hour,
            day_of_week=at_time.weekday(),
            month=at_time.month,
            is_monsoon=is_monsoon,
            delta_1h=delta_1h,
            delta_3h=delta_3h,
            delta_6h=delta_6h,
            delta_12h=delta_12h,
            delta_24h=delta_24h,
            slope_1h=slope_1h,
            acceleration=acceleration,
            level_mean_24h=level_mean_24h,
            level_max_24h=level_max_24h,
            level_min_24h=level_min_24h,
            level_std_24h=level_std_24h,
            submission_count_24h=len(submissions),
            site_flood_history_count=flood_history,
            river_type_encoded=self._encode_river_type(site),
            rainfall_last_3h=rainfall_last_3h,
            rainfall_last_24h=rainfall_last_24h,
            forecast_rainfall_6h=forecast_6h
        )
    
    def generate_training_data(
        self,
        days_back: int = 90,
        label_horizon_hours: int = 6
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Generate training data from historical submissions.
        
        Args:
            days_back: How many days of history to use
            label_horizon_hours: How many hours ahead to predict flood
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,)
            site_ids: Site ID for each sample (for stratified splits)
        """
        from models import MonitoringSite, WaterLevelSubmission, db
        
        logger.info(f"Generating training data for last {days_back} days")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        # Get all active sites
        sites = MonitoringSite.query.filter_by(is_active=True).all()
        
        X_list = []
        y_list = []
        site_ids = []
        
        for site in sites:
            # Get submissions for this site
            submissions = WaterLevelSubmission.query.filter(
                WaterLevelSubmission.site_id == site.id,
                WaterLevelSubmission.timestamp >= start_time,
                WaterLevelSubmission.timestamp <= end_time
            ).order_by(WaterLevelSubmission.timestamp).all()
            
            if len(submissions) < 5:
                continue
            
            # Generate samples at each submission timestamp
            for i, sub in enumerate(submissions[:-1]):  # Skip last (need future for label)
                features = self.get_site_features(site.id, sub.timestamp)
                if features is None:
                    continue
                
                # Generate label based on future water levels
                label = self._generate_label(
                    site, submissions, i, label_horizon_hours
                )
                
                X_list.append(features.to_feature_vector())
                y_list.append(label)
                site_ids.append(site.id)
        
        if not X_list:
            logger.warning("No training data generated")
            return np.array([]), np.array([]), []
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Generated {len(X)} training samples from {len(sites)} sites")
        logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, site_ids
    
    def _generate_label(
        self,
        site,
        submissions: List,
        current_idx: int,
        horizon_hours: int
    ) -> int:
        """
        Generate label based on future conditions.
        
        Labels:
        0 = SAFE: No significant rise
        1 = CAUTION: Moderate rise or approaching alert level
        2 = FLOOD_RISK: Exceeds danger threshold
        3 = FLASH_FLOOD_RISK: Rapid rise (>50cm in 3h)
        """
        current_sub = submissions[current_idx]
        current_level = current_sub.water_level * 100  # cm
        current_time = current_sub.timestamp
        
        # Get thresholds
        danger_level = getattr(site, 'danger_level', 500)  # Default 5m
        alert_level = getattr(site, 'alert_level', 300)   # Default 3m
        
        # Look at future submissions within horizon
        future_subs = [
            s for s in submissions[current_idx+1:]
            if s.timestamp <= current_time + timedelta(hours=horizon_hours)
        ]
        
        if not future_subs:
            return 0  # SAFE if no future data
        
        future_levels = [s.water_level * 100 for s in future_subs]
        max_future_level = max(future_levels)
        max_rise = max_future_level - current_level
        
        # Check for flash flood (rapid rise)
        if max_rise > 50 and len(future_subs) >= 2:
            # Check if rapid rise happened within 3 hours
            for sub in future_subs:
                rise = (sub.water_level * 100) - current_level
                time_diff = (sub.timestamp - current_time).total_seconds() / 3600
                if rise > 50 and time_diff <= 3:
                    return 3  # FLASH_FLOOD_RISK
        
        # Check threshold-based labels
        if max_future_level >= danger_level:
            return 2  # FLOOD_RISK
        elif max_future_level >= alert_level or max_rise > 30:
            return 1  # CAUTION
        else:
            return 0  # SAFE
    
    def _calc_delta(
        self, 
        levels: List[Tuple[datetime, float]], 
        current_level: float,
        past_time: datetime
    ) -> float:
        """Calculate water level change from past_time to now"""
        for ts, level in levels:
            if ts <= past_time:
                return current_level - level
        return 0.0
    
    def _calc_slope(
        self,
        levels: List[Tuple[datetime, float]],
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """Calculate slope (rate of change) between two time points"""
        start_level = None
        end_level = None
        
        for ts, level in levels:
            if ts <= end_time and start_level is None:
                start_level = level
            if ts <= start_time:
                end_level = level
                break
        
        if start_level is None or end_level is None:
            return 0.0
        
        time_diff = (start_time - end_time).total_seconds() / 3600
        if time_diff == 0:
            return 0.0
        
        return (end_level - start_level) / time_diff
    
    def _create_empty_features(self, site, at_time: datetime) -> SiteFeatures:
        """Create features with zero values when no data available"""
        return SiteFeatures(
            site_id=site.id,
            site_name=site.name,
            timestamp=at_time,
            water_level_cm=0,
            pct_of_danger_threshold=0,
            pct_of_alert_threshold=0,
            hour=at_time.hour,
            day_of_week=at_time.weekday(),
            month=at_time.month,
            is_monsoon=at_time.month in [6, 7, 8, 9]
        )
    
    def _get_flood_history_count(self, site_id: int) -> int:
        """Get count of historical flood events at this site"""
        from models import WaterLevelSubmission, MonitoringSite
        
        site = MonitoringSite.query.get(site_id)
        if not site:
            return 0
        
        danger_level = getattr(site, 'danger_level', 500) / 100  # Convert to meters
        
        # Count submissions where level exceeded danger threshold
        count = WaterLevelSubmission.query.filter(
            WaterLevelSubmission.site_id == site_id,
            WaterLevelSubmission.water_level >= danger_level
        ).count()
        
        return min(count, 100)  # Cap at 100
    
    def _encode_river_type(self, site) -> int:
        """Encode river type as numeric feature"""
        # This could be enhanced based on site metadata
        name = site.name.lower()
        if any(r in name for r in ['ganga', 'yamuna', 'brahmaputra', 'godavari', 'krishna']):
            return 1  # Major river
        elif any(r in name for r in ['nullah', 'drain', 'canal']):
            return 3  # Tributary/canal
        else:
            return 2  # Minor river
    
    def _get_weather_features(
        self, 
        lat: float, 
        lon: float, 
        at_time: datetime
    ) -> Tuple[float, float, float]:
        """
        Get weather features from external API (stubbed).
        
        In production, integrate with:
        - IMD (India Meteorological Department) API
        - OpenWeatherMap
        - NOAA precipitation data
        
        Returns:
            rainfall_last_3h (mm)
            rainfall_last_24h (mm)
            forecast_6h (mm)
        """
        # Stub: Return zeros until external API is integrated
        # TODO: Implement actual weather data fetch
        return 0.0, 0.0, 0.0


# Singleton instance
pipeline = FloodDataPipeline()
