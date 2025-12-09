"""
Hydrology Utilities for Flood Prediction
=========================================
Provides rate-of-rise calculation, threshold logic, and GeoJSON utilities.

Author: JalScan Team
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from shapely.geometry import Point, Polygon, mapping


def compute_rate_of_rise(
    old_level: float,
    old_timestamp: str,
    new_level: float,
    new_timestamp: str
) -> float:
    """
    Compute rate of water level rise in meters per hour.
    
    Args:
        old_level: Previous water level reading (meters)
        old_timestamp: ISO timestamp of previous reading
        new_level: Current water level reading (meters)
        new_timestamp: ISO timestamp of current reading
    
    Returns:
        Rate of rise in meters per hour (can be negative for falling water)
    """
    try:
        # Parse ISO timestamps
        if isinstance(old_timestamp, str):
            t0 = datetime.fromisoformat(old_timestamp.replace('Z', '+00:00'))
        else:
            t0 = old_timestamp
            
        if isinstance(new_timestamp, str):
            t1 = datetime.fromisoformat(new_timestamp.replace('Z', '+00:00'))
        else:
            t1 = new_timestamp
        
        # Calculate time difference in hours
        hours = max((t1 - t0).total_seconds() / 3600.0, 0.001)  # Avoid division by zero
        
        # Calculate rate
        rate = (new_level - old_level) / hours
        return round(rate, 4)
        
    except Exception as e:
        print(f"Error computing rate of rise: {e}")
        return 0.0


def predict_delta_h(
    rate_of_rise: Optional[float],
    hours_ahead: int,
    current_level: float,
    warning_level: float,
    danger_level: float,
    rate_threshold: float = 0.3  # m/hr threshold for significant rise
) -> Tuple[float, str]:
    """
    Predict water level change based on rate-of-rise and thresholds.
    
    Decision logic:
    - If rate > threshold (0.3 m/hr), predict delta = rate * hours
    - If current level >= warning level, predict conservative rise
    - Otherwise, predict minimal change
    
    Args:
        rate_of_rise: Current rate of rise (m/hr), can be None
        hours_ahead: Hours to predict ahead
        current_level: Current water level
        warning_level: Warning threshold
        danger_level: Danger threshold
        rate_threshold: Rate threshold for significant rise (default 0.3 m/hr)
    
    Returns:
        Tuple of (predicted_delta_h, prediction_basis)
    """
    if rate_of_rise and rate_of_rise > rate_threshold:
        # Significant rising - use rate-based prediction
        delta_h = rate_of_rise * hours_ahead
        basis = f"rate_based ({rate_of_rise:.2f} m/hr × {hours_ahead}h)"
    elif current_level >= warning_level:
        # Already at warning - predict conservative rise to danger
        delta_h = max(0.5, (danger_level - current_level))
        basis = "above_warning_threshold"
    elif rate_of_rise and rate_of_rise > 0:
        # Small positive rate - extrapolate with damping
        delta_h = rate_of_rise * hours_ahead * 0.5  # 50% damping for small rates
        basis = "minor_rise_damped"
    else:
        # Stable or falling - minimal prediction
        delta_h = 0.0
        basis = "stable_conditions"
    
    return round(delta_h, 2), basis


def calculate_travel_time(distance_m: float, velocity_m_per_s: float) -> float:
    """
    Calculate flood wave travel time downstream.
    
    Args:
        distance_m: Distance downstream in meters
        velocity_m_per_s: Water velocity in m/s (from Manning's equation)
    
    Returns:
        Travel time in hours
    """
    if velocity_m_per_s <= 0:
        return float('inf')
    
    travel_seconds = distance_m / velocity_m_per_s
    travel_hours = travel_seconds / 3600.0
    return round(travel_hours, 2)


def create_flood_polygon(
    lat: float,
    lon: float,
    delta_h: float,
    spread_per_meter: float = 10.0  # meters lateral spread per 1m depth
) -> Polygon:
    """
    Create a flood polygon as a buffer around a point.
    
    For PoC, uses empirical relationship: 1m depth → 10m lateral spread.
    In production, would use DEM-based flood-fill.
    
    Args:
        lat: Latitude of flood center
        lon: Longitude of flood center
        delta_h: Predicted water level rise (meters)
        spread_per_meter: Meters of lateral spread per meter of depth
    
    Returns:
        Shapely Polygon representing flooded area
    """
    buffer_m = delta_h * spread_per_meter
    
    if buffer_m <= 0:
        # Return tiny polygon for no-flood
        return Point(lon, lat).buffer(0.00005)
    
    # Convert meter buffer to degrees (approximate at latitude)
    deg_per_meter = 1.0 / 111320.0
    buffer_deg = buffer_m * deg_per_meter
    
    return Point(lon, lat).buffer(buffer_deg)


def polygon_to_geojson(polygon: Polygon) -> Dict[str, Any]:
    """
    Convert Shapely polygon to GeoJSON dictionary.
    
    Args:
        polygon: Shapely Polygon object
    
    Returns:
        GeoJSON dictionary with type "Feature"
    """
    return {
        "type": "Feature",
        "geometry": mapping(polygon),
        "properties": {
            "generated_at": datetime.utcnow().isoformat(),
            "source": "jalscan_flood_synthesis"
        }
    }


def get_flood_severity(
    current_level: float,
    warning_level: float,
    danger_level: float,
    predicted_delta: float = 0.0
) -> Dict[str, Any]:
    """
    Determine flood severity based on thresholds.
    
    Args:
        current_level: Current water level
        warning_level: Warning threshold
        danger_level: Danger threshold
        predicted_delta: Predicted rise (optional)
    
    Returns:
        Dictionary with severity info
    """
    predicted_level = current_level + predicted_delta
    
    if predicted_level >= danger_level:
        severity = "danger"
        color = "#dc3545"  # Red
        message = "DANGER: Predicted to exceed danger level"
    elif predicted_level >= warning_level:
        severity = "warning"
        color = "#ffc107"  # Yellow
        message = "WARNING: Predicted to exceed warning level"
    elif current_level >= warning_level:
        severity = "elevated"
        color = "#fd7e14"  # Orange
        message = "ELEVATED: Currently at warning level"
    else:
        severity = "normal"
        color = "#28a745"  # Green
        message = "NORMAL: Within safe limits"
    
    return {
        "severity": severity,
        "color": color,
        "message": message,
        "current_level": round(current_level, 2),
        "predicted_level": round(predicted_level, 2),
        "warning_level": round(warning_level, 2),
        "danger_level": round(danger_level, 2)
    }
