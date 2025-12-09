"""
Flood Synthesis Module
======================
Provides flood prediction and visualization using physics-based modeling
and AI-generated flood overlay visualization.

Author: JalScan Team
"""

from .physics_engine import (
    FloodMaskGenerator,
    generate_synthetic_dem,
    calculate_mannings_velocity,
    calculate_flood_depth
)

from .model import create_simple_flood_overlay

__all__ = [
    'FloodMaskGenerator',
    'generate_synthetic_dem',
    'calculate_mannings_velocity',
    'calculate_flood_depth',
    'create_simple_flood_overlay'
]
