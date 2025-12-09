"""
Flask API for Flood Prediction and Visualization
=================================================
Provides endpoints for flood prediction using physics engine
and visualization overlay.

Author: JalScan Team
"""

import os
import base64
import json
import logging
import random
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
from flask import Blueprint, request, jsonify, current_app, url_for, send_file
from PIL import Image

from .physics_engine import FloodMaskGenerator, generate_synthetic_dem
from .model import create_simple_flood_overlay
from .hydrology import (
    compute_rate_of_rise, predict_delta_h, create_flood_polygon,
    polygon_to_geojson, get_flood_severity
)

logger = logging.getLogger(__name__)

# Create Blueprint
flood_bp = Blueprint('flood', __name__, url_prefix='/api/flood')

# Demo mode flag - set to True for hackathon demo
DEMO_MODE = True

# Demo images directory
DEMO_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'demo_images')

# Google Maps Static API Key (set via environment variable)
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', '')

# Real river locations from the monitoring sites database
DEMO_SCENARIOS = [
    {
        'id': 'ganga_haridwar',
        'name': 'Ganga River, Haridwar',
        'lat': 29.9457,
        'lon': 78.1642,
        'description': 'Holy Ganga at Haridwar - major flood-prone zone',
        'stats': {
            'flooded_area_km2': 18.5,
            'flooded_percentage': 24.3,
            'max_depth_m': 4.5,
            'mean_depth_m': 2.1
        }
    },
    {
        'id': 'yamuna_delhi',
        'name': 'Yamuna River, Delhi',
        'lat': 28.6139,
        'lon': 77.2090,
        'description': 'Yamuna flood simulation in Delhi NCR region',
        'stats': {
            'flooded_area_km2': 22.8,
            'flooded_percentage': 28.5,
            'max_depth_m': 5.2,
            'mean_depth_m': 2.4
        }
    },
    {
        'id': 'krishna_vijayawada',
        'name': 'Krishna River, Vijayawada',
        'lat': 16.5062,
        'lon': 80.6480,
        'description': 'Krishna River flood scenario at Vijayawada',
        'stats': {
            'flooded_area_km2': 15.3,
            'flooded_percentage': 21.1,
            'max_depth_m': 3.8,
            'mean_depth_m': 1.7
        }
    },
    {
        'id': 'godavari_nashik',
        'name': 'Godavari River, Nashik',
        'lat': 19.9975,
        'lon': 73.7898,
        'description': 'Godavari origin point flood simulation',
        'stats': {
            'flooded_area_km2': 12.4,
            'flooded_percentage': 18.2,
            'max_depth_m': 3.5,
            'mean_depth_m': 1.5
        }
    },
    {
        'id': 'brahmaputra_guwahati',
        'name': 'Brahmaputra River, Guwahati',
        'lat': 26.1445,
        'lon': 91.7362,
        'description': 'Mighty Brahmaputra flood scenario',
        'stats': {
            'flooded_area_km2': 35.6,
            'flooded_percentage': 38.4,
            'max_depth_m': 6.8,
            'mean_depth_m': 3.2
        }
    },
    {
        'id': 'narmada_jabalpur',
        'name': 'Narmada River, Jabalpur',
        'lat': 23.1815,
        'lon': 79.9864,
        'description': 'Narmada River flood simulation',
        'stats': {
            'flooded_area_km2': 14.7,
            'flooded_percentage': 19.6,
            'max_depth_m': 4.1,
            'mean_depth_m': 1.9
        }
    },
    {
        'id': 'kaveri_trichy',
        'name': 'Kaveri River, Tiruchirappalli',
        'lat': 10.7905,
        'lon': 78.7047,
        'description': 'Kaveri delta flood scenario',
        'stats': {
            'flooded_area_km2': 16.2,
            'flooded_percentage': 22.8,
            'max_depth_m': 3.6,
            'mean_depth_m': 1.6
        }
    },
    {
        'id': 'tapi_surat',
        'name': 'Tapi River, Surat',
        'lat': 21.1702,
        'lon': 72.8311,
        'description': 'Tapi River flood simulation in Surat',
        'stats': {
            'flooded_area_km2': 11.9,
            'flooded_percentage': 17.3,
            'max_depth_m': 3.2,
            'mean_depth_m': 1.4
        }
    },
    {
        'id': 'musi_hyderabad',
        'name': 'Musi River, Hyderabad',
        'lat': 17.3850,
        'lon': 78.4867,
        'description': 'Musi River overflow in Hyderabad city',
        'stats': {
            'flooded_area_km2': 12.5,
            'flooded_percentage': 18.3,
            'max_depth_m': 3.2,
            'mean_depth_m': 1.4
        }
    }
]


def generate_demo_flood_image(
    lat: float,
    lon: float,
    water_level_rise: float = 2.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate a demo flood image for presentation.
    Uses location coordinates to generate unique flood patterns.
    """
    # Find scenario info for stats
    scenario_index = find_closest_scenario(lat, lon)
    scenario = DEMO_SCENARIOS[scenario_index % len(DEMO_SCENARIOS)]

    # Generate location-specific DEM and river pattern
    dem, river_mask = generate_synthetic_dem(
        lat=lat,
        lon=lon,
        terrain_variation=8.0 + (abs(lat) % 5)
    )

    # Create a gradient satellite-like image with location-based colors
    height, width = dem.shape
    satellite = np.zeros((height, width, 3), dtype=np.uint8)

    # Use location to vary base colors
    color_seed = int(abs(lat * 100 + lon * 10)) % 50
    base_r = 70 + color_seed // 2
    base_g = 110 + color_seed
    base_b = 50 + color_seed // 3

    # Green/brown base (land) with terrain shading
    dem_norm = (dem - dem.min()) / (dem.max() - dem.min() + 0.01)
    satellite[:, :, 0] = np.clip(base_r + dem_norm * 40, 0, 255).astype(np.uint8)
    satellite[:, :, 1] = np.clip(base_g + dem_norm * 50, 0, 255).astype(np.uint8)
    satellite[:, :, 2] = np.clip(base_b + dem_norm * 30, 0, 255).astype(np.uint8)

    # Add river (blue variations based on location)
    river_blue = 140 + int(lat % 20)
    satellite[river_mask == 1, 0] = 35 + int(lon % 15)
    satellite[river_mask == 1, 1] = 70 + int(lat % 20)
    satellite[river_mask == 1, 2] = river_blue

    # Generate flood mask using physics engine
    generator = FloodMaskGenerator(roughness_coefficient=0.035)
    base_level = 95.0

    # Scale water level rise by user input
    actual_rise = water_level_rise * 1.5

    flood_mask, stats = generator.calculate_flood_extent(
        dem=dem,
        base_water_level=base_level,
        water_level_rise=actual_rise
    )

    # Create flood overlay with location-based color variations
    flood_r = 60 + int(abs(lat) % 20)
    flood_g = 100 + int(abs(lon) % 20)
    flood_b = 210 + int((abs(lat) + abs(lon)) % 30)

    flood_image = create_simple_flood_overlay(
        satellite_image=satellite,
        flood_mask=flood_mask,
        flood_color=(flood_r, flood_g, min(flood_b, 255)),
        opacity=0.55
    )

    # Add scenario info to stats
    stats['scenario'] = scenario
    stats['generated_at'] = datetime.utcnow().isoformat()

    return flood_image, stats


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def save_generated_image(image: np.ndarray, filename: str) -> str:
    """Save generated image and return its URL path."""
    output_dir = os.path.join(current_app.static_folder, 'flood_outputs')
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    Image.fromarray(image).save(filepath)

    return f'/static/flood_outputs/{filename}'


def find_closest_scenario(lat: float, lon: float) -> int:
    """Find the closest demo scenario to the given coordinates."""
    min_distance = float('inf')
    closest_index = 0

    for i, scenario in enumerate(DEMO_SCENARIOS):
        distance = ((lat - scenario['lat'])**2 + (lon - scenario['lon'])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index


# ============================================================================
# API ROUTES
# ============================================================================

@flood_bp.route('/predict', methods=['POST'])
def predict_flood():
    """
    Main flood prediction endpoint.

    Request JSON:
    {
        "lat": float,
        "lon": float,
        "water_level_rise": float
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

        lat = data.get('lat')
        lon = data.get('lon')
        water_level_rise = data.get('water_level_rise', 2.0)

        if lat is None or lon is None:
            return jsonify({'success': False, 'error': 'lat and lon are required'}), 400

        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'success': False, 'error': 'Invalid coordinates'}), 400

        if not (0 < water_level_rise <= 20):
            return jsonify({'success': False, 'error': 'water_level_rise must be between 0 and 20 meters'}), 400

        logger.info(f"Flood prediction request: lat={lat}, lon={lon}, rise={water_level_rise}m")

        # Generate flood image
        flood_image, stats = generate_demo_flood_image(lat, lon, water_level_rise)

        # Save and get URL
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'flood_demo_{timestamp}.png'
        image_url = save_generated_image(flood_image, filename)

        # Calculate overlay bounds (approximate 5km radius)
        delta = 0.045
        overlay_bounds = {
            'north': lat + delta,
            'south': lat - delta,
            'east': lon + delta,
            'west': lon - delta
        }

        return jsonify({
            'success': True,
            'demo_mode': DEMO_MODE,
            'image_url': image_url,
            'image_base64': image_to_base64(flood_image),
            'statistics': stats,
            'overlay_bounds': overlay_bounds,
            'message': 'Flood prediction generated successfully'
        })

    except Exception as e:
        logger.exception(f"Error in flood prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flood_bp.route('/demo', methods=['GET'])
def get_demo_scenarios():
    """Get available demo scenarios."""
    return jsonify({
        'success': True,
        'demo_mode': DEMO_MODE,
        'scenarios': DEMO_SCENARIOS
    })


@flood_bp.route('/demo/<scenario_id>', methods=['GET'])
def get_demo_scenario(scenario_id: str):
    """Get a specific demo scenario with pre-generated flood image."""
    # Find scenario
    scenario = None
    for s in DEMO_SCENARIOS:
        if s['id'] == scenario_id:
            scenario = s
            break

    if not scenario:
        return jsonify({'success': False, 'error': 'Scenario not found'}), 404

    water_level_rise = request.args.get('water_level_rise', 2.0, type=float)

    # Generate flood image
    flood_image, stats = generate_demo_flood_image(scenario['lat'], scenario['lon'], water_level_rise)

    # Save image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'flood_{scenario_id}_{timestamp}.png'
    image_url = save_generated_image(flood_image, filename)

    # Calculate overlay bounds
    delta = 0.045
    overlay_bounds = {
        'north': scenario['lat'] + delta,
        'south': scenario['lat'] - delta,
        'east': scenario['lon'] + delta,
        'west': scenario['lon'] - delta
    }

    return jsonify({
        'success': True,
        'scenario': scenario,
        'image_url': image_url,
        'image_base64': image_to_base64(flood_image),
        'statistics': stats,
        'overlay_bounds': overlay_bounds
    })


@flood_bp.route('/predict-from-site', methods=['POST'])
def predict_from_site():
    """
    Predict flood based on a monitoring site's real submission data.
    
    Uses rate-of-rise from latest submissions and site thresholds.
    
    Request JSON:
    {
        "site_id": int,
        "hours_ahead": int (default 24)
    }
    """
    try:
        # Import models here to avoid circular imports
        from models import MonitoringSite, WaterLevelSubmission
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        site_id = data.get('site_id')
        hours_ahead = int(data.get('hours_ahead', 24))
        
        if not site_id:
            return jsonify({'success': False, 'error': 'site_id is required'}), 400
        
        # Fetch site info
        site = MonitoringSite.query.get(site_id)
        if not site:
            return jsonify({'success': False, 'error': 'Site not found'}), 404
        
        # Fetch latest 2 submissions for rate-of-rise
        submissions = WaterLevelSubmission.query.filter_by(
            site_id=site_id
        ).order_by(WaterLevelSubmission.timestamp.desc()).limit(2).all()
        
        if not submissions:
            return jsonify({'success': False, 'error': 'No submissions found for this site'}), 404
        
        current_level = submissions[0].water_level
        current_ts = submissions[0].timestamp.isoformat()
        
        # Calculate rate of rise if we have 2 readings
        rate_of_rise = None
        if len(submissions) >= 2:
            old_level = submissions[1].water_level
            old_ts = submissions[1].timestamp.isoformat()
            rate_of_rise = compute_rate_of_rise(old_level, old_ts, current_level, current_ts)
        
        # Get thresholds (use site flood_threshold or defaults)
        warning_level = (site.flood_threshold or 10.0) * 0.7  # 70% of danger as warning
        danger_level = site.flood_threshold or 10.0
        
        # Predict delta_h
        delta_h, prediction_basis = predict_delta_h(
            rate_of_rise=rate_of_rise,
            hours_ahead=hours_ahead,
            current_level=current_level,
            warning_level=warning_level,
            danger_level=danger_level
        )
        
        # Get flood severity info
        severity_info = get_flood_severity(
            current_level=current_level,
            warning_level=warning_level,
            danger_level=danger_level,
            predicted_delta=delta_h
        )
        
        # Generate flood polygon
        polygon = create_flood_polygon(site.latitude, site.longitude, delta_h)
        geojson = polygon_to_geojson(polygon)
        
        # Generate flood image visualization
        water_level_rise = max(0.5, delta_h)  # Minimum for visualization
        flood_image, stats = generate_demo_flood_image(site.latitude, site.longitude, water_level_rise)
        
        # Save and get URL
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'flood_site_{site_id}_{timestamp_str}.png'
        image_url = save_generated_image(flood_image, filename)
        
        # Calculate overlay bounds
        delta = 0.045
        overlay_bounds = {
            'north': site.latitude + delta,
            'south': site.latitude - delta,
            'east': site.longitude + delta,
            'west': site.longitude - delta
        }
        
        return jsonify({
            'success': True,
            'site': {
                'id': site.id,
                'name': site.name,
                'lat': site.latitude,
                'lon': site.longitude,
                'river_code': site.river_code
            },
            'prediction': {
                'hours_ahead': hours_ahead,
                'delta_h': delta_h,
                'prediction_basis': prediction_basis,
                'rate_of_rise_m_hr': rate_of_rise
            },
            'severity': severity_info,
            'geojson': geojson,
            'image_url': image_url,
            'image_base64': image_to_base64(flood_image),
            'statistics': stats,
            'overlay_bounds': overlay_bounds
        })
        
    except Exception as e:
        logger.exception(f"Error in predict-from-site: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flood_bp.route('/sites', methods=['GET'])
def get_monitoring_sites():
    """Get all monitoring sites for the site selector dropdown."""
    try:
        from models import MonitoringSite
        
        sites = MonitoringSite.query.filter_by(is_active=True).all()
        
        return jsonify({
            'success': True,
            'sites': [{
                'id': s.id,
                'name': s.name,
                'lat': s.latitude,
                'lon': s.longitude,
                'river_code': s.river_code,
                'flood_threshold': s.flood_threshold
            } for s in sites]
        })
    except Exception as e:
        logger.exception(f"Error fetching sites: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flood_bp.route('/status', methods=['GET'])
def get_status():
    """Get the status of the flood prediction service."""
    return jsonify({
        'success': True,
        'service': 'Flood Synthesis API',
        'version': '1.0.0',
        'demo_mode': DEMO_MODE,
        'google_maps_configured': bool(GOOGLE_MAPS_API_KEY),
        'available_scenarios': len(DEMO_SCENARIOS)
    })


def init_app(app):
    """
    Initialize the flood synthesis module with the Flask app.
    """
    # Register blueprint
    app.register_blueprint(flood_bp)

    # Create output directory
    output_dir = os.path.join(app.static_folder, 'flood_outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Create demo images directory
    os.makedirs(DEMO_IMAGES_DIR, exist_ok=True)

    logger.info("Flood Synthesis module initialized")
    logger.info(f"Demo mode: {'ON' if DEMO_MODE else 'OFF'}")
