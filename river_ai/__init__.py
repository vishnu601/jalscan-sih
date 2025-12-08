"""
River Memory AI - Computer Vision Package
Complete backend for river monitoring digital twin
"""

__version__ = "1.0.0"

from .pipeline import process_image_for_site
from .gauge_detection import detect_water_level
from .color_analysis import analyze_water_color
from .flow_estimation import estimate_flow_speed
from .gauge_health import analyze_gauge_health
from .bank_erosion import analyze_bank_erosion
from .anomaly_detection import detect_anomalies
