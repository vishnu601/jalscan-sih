"""
River Memory AI - API Routes
REST API endpoints for river snapshot processing and timeline retrieval
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import os
import logging
import json
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

# Create Blueprint
river_api = Blueprint('river_api', __name__, url_prefix='/api/v1')


def register_river_ai_routes(app, db):
    """Register River Memory AI routes with Flask app"""
    
    from models import MonitoringSite, WaterLevelSubmission, RiverAnalysis
    
    UPLOAD_FOLDER = app.config.get('UPLOAD_FOLDER', 'static/uploads')
    
    @app.route('/api/v1/river-snapshot', methods=['POST'])
    def create_river_snapshot():
        """
        Process a river snapshot image through River Memory AI.
        
        Input (multipart form):
            - image: River photo file
            - site_id: Site identifier (required)
            - captured_at: Timestamp string (optional)
            - manual_water_level_cm: Manual reading (optional)
            - latitude: GPS latitude (optional)
            - longitude: GPS longitude (optional)
            
        Returns:
            Complete analysis results with all computed features
        """
        try:
            # Validate required fields
            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image provided'}), 400
            
            site_id = request.form.get('site_id')
            if not site_id:
                return jsonify({'success': False, 'error': 'site_id is required'}), 400
            
            # Get site
            site = MonitoringSite.query.get(int(site_id))
            if not site:
                return jsonify({'success': False, 'error': f'Site {site_id} not found'}), 404
            
            # Save uploaded image
            image_file = request.files['image']
            filename = secure_filename(f"river_{site_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg")
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(image_path)
            
            # Parse optional parameters
            captured_at = None
            if request.form.get('captured_at'):
                try:
                    captured_at = datetime.fromisoformat(
                        request.form['captured_at'].replace('Z', '+00:00')
                    )
                except:
                    captured_at = datetime.utcnow()
            
            manual_level = None
            if request.form.get('manual_water_level_cm'):
                try:
                    manual_level = float(request.form['manual_water_level_cm'])
                except:
                    pass
            
            # Build site config from database
            site_config = {
                'gauge_calibration_pixels_per_cm': getattr(site, 'gauge_calibration_pixels_per_cm', 10.0) or 10.0,
                'gauge_zero_pixel_y': getattr(site, 'gauge_zero_pixel_y', 400) or 400,
            }
            
            # Run River Memory AI pipeline
            from river_ai import process_image_for_site
            
            result = process_image_for_site(
                image_path=image_path,
                site_id=str(site_id),
                db_session=db.session,
                site_config=site_config,
                manual_water_level=manual_level
            )
            
            # Create alerts if anomaly detected
            if result.get('anomaly_detected') and result.get('anomaly_score', 0) > 0.5:
                try:
                    from models import FloodAlert
                    
                    alert = FloodAlert(
                        site_id=int(site_id),
                        alert_type='river_memory_anomaly',
                        message=result.get('anomaly_reason', 'Anomaly detected'),
                        severity='high' if result.get('anomaly_score', 0) > 0.7 else 'medium'
                    )
                    db.session.add(alert)
                    db.session.commit()
                    result['alert_created'] = True
                except Exception as e:
                    logger.warning(f"Could not create alert: {e}")
            
            return jsonify({
                'success': True,
                'site_id': site_id,
                'site_name': site.name,
                **result
            })
            
        except Exception as e:
            logger.error(f"River snapshot error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/v1/sites/<int:site_id>/timeline')
    def get_site_timeline(site_id):
        """
        Get analysis timeline for a site.
        
        Query params:
            - from: Start date (ISO format)
            - to: End date (ISO format)
            - limit: Max results (default 100)
            
        Returns:
            Ordered list of analyses with images and features
        """
        try:
            site = MonitoringSite.query.get(site_id)
            if not site:
                return jsonify({'success': False, 'error': 'Site not found'}), 404
            
            # Parse date range
            from_date = request.args.get('from')
            to_date = request.args.get('to')
            limit = request.args.get('limit', 100, type=int)
            
            query = RiverAnalysis.query.filter_by(site_id=site_id)
            
            if from_date:
                try:
                    from_dt = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
                    query = query.filter(RiverAnalysis.timestamp >= from_dt)
                except:
                    pass
            
            if to_date:
                try:
                    to_dt = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
                    query = query.filter(RiverAnalysis.timestamp <= to_dt)
                except:
                    pass
            
            analyses = query.order_by(RiverAnalysis.timestamp.desc()).limit(limit).all()
            
            timeline = []
            for a in analyses:
                # Get associated submission for image URL
                image_url = None
                if a.submission_id:
                    submission = WaterLevelSubmission.query.get(a.submission_id)
                    if submission and submission.photo_filename:
                        image_url = f"/static/uploads/{submission.photo_filename}"
                
                timeline.append({
                    'id': a.id,
                    'timestamp': a.timestamp.isoformat() if a.timestamp else None,
                    'image_url': image_url,
                    'water_level_cm': getattr(a, 'water_level_cm', None),
                    'flow_class': a.flow_speed_class,
                    'color_class': a.sediment_type,
                    'color_index': a.pollution_index,
                    'turbulence_score': a.turbulence_score,
                    'gauge_visibility_score': a.gauge_visibility_score,
                    'gauge_damage': a.gauge_damage_detected,
                    'damage_types': a.damage_type.split(',') if a.damage_type else [],
                    'bank_status': 'erosion' if a.erosion_detected else 'stable',
                    'anomaly_detected': a.anomaly_detected,
                    'anomaly_type': a.anomaly_type,
                    'anomaly_message': a.anomaly_description,
                    'overall_risk': a.overall_risk
                })
            
            return jsonify({
                'success': True,
                'site_id': site_id,
                'site_name': site.name,
                'count': len(timeline),
                'timeline': timeline
            })
            
        except Exception as e:
            logger.error(f"Timeline error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/v1/sites/<int:site_id>/summary')
    def get_site_summary(site_id):
        """
        Get current summary and risk for a site.
        
        Returns:
            - Latest reading
            - Water level changes (3h, 24h)
            - Active alerts
            - Overall risk score
        """
        try:
            site = MonitoringSite.query.get(site_id)
            if not site:
                return jsonify({'success': False, 'error': 'Site not found'}), 404
            
            # Get latest analysis
            latest = RiverAnalysis.query.filter_by(site_id=site_id)\
                .order_by(RiverAnalysis.timestamp.desc()).first()
            
            if not latest:
                return jsonify({
                    'success': True,
                    'site_id': site_id,
                    'site_name': site.name,
                    'has_data': False,
                    'message': 'No analysis data available for this site'
                })
            
            # Get readings for delta calculation
            now = datetime.utcnow()
            reading_3h = RiverAnalysis.query.filter(
                RiverAnalysis.site_id == site_id,
                RiverAnalysis.timestamp <= now - timedelta(hours=3)
            ).order_by(RiverAnalysis.timestamp.desc()).first()
            
            reading_24h = RiverAnalysis.query.filter(
                RiverAnalysis.site_id == site_id,
                RiverAnalysis.timestamp <= now - timedelta(hours=24)
            ).order_by(RiverAnalysis.timestamp.desc()).first()
            
            # Calculate deltas
            current_level = getattr(latest, 'water_level_cm', None)
            delta_3h = None
            delta_24h = None
            
            if current_level:
                if reading_3h and hasattr(reading_3h, 'water_level_cm'):
                    level_3h = getattr(reading_3h, 'water_level_cm', None)
                    if level_3h:
                        delta_3h = current_level - level_3h
                
                if reading_24h and hasattr(reading_24h, 'water_level_cm'):
                    level_24h = getattr(reading_24h, 'water_level_cm', None)
                    if level_24h:
                        delta_24h = current_level - level_24h
            
            # Get active alerts
            from models import FloodAlert
            active_alerts = FloodAlert.query.filter(
                FloodAlert.site_id == site_id,
                FloodAlert.created_at >= now - timedelta(hours=24)
            ).order_by(FloodAlert.created_at.desc()).limit(5).all()
            
            # Compute risk score (0-100)
            risk_score = 0
            if current_level and current_level > 300:
                risk_score += min(40, (current_level - 300) / 5)
            if latest.flow_speed_class == 'turbulent':
                risk_score += 30
            elif latest.flow_speed_class == 'high':
                risk_score += 15
            if latest.anomaly_detected:
                risk_score += latest.turbulence_score * 0.3 if latest.turbulence_score else 20
            risk_score = min(100, int(risk_score))
            
            return jsonify({
                'success': True,
                'site_id': site_id,
                'site_name': site.name,
                'has_data': True,
                'latest': {
                    'timestamp': latest.timestamp.isoformat() if latest.timestamp else None,
                    'water_level_cm': current_level,
                    'flow_class': latest.flow_speed_class,
                    'color_class': latest.sediment_type,
                    'turbulence_score': latest.turbulence_score,
                    'overall_risk': latest.overall_risk
                },
                'changes': {
                    'delta_3h_cm': round(delta_3h, 1) if delta_3h else None,
                    'delta_24h_cm': round(delta_24h, 1) if delta_24h else None
                },
                'risk_score': risk_score,
                'risk_level': 'high' if risk_score >= 60 else 'medium' if risk_score >= 30 else 'low',
                'active_alerts': [
                    {
                        'type': a.alert_type,
                        'message': a.message,
                        'severity': a.severity,
                        'created_at': a.created_at.isoformat() if a.created_at else None
                    }
                    for a in active_alerts
                ],
                'gauge_health': {
                    'visibility_score': latest.gauge_visibility_score,
                    'damage_detected': latest.gauge_damage_detected,
                    'maintenance_needed': latest.gauge_visibility_score < 70 if latest.gauge_visibility_score else False
                }
            })
            
        except Exception as e:
            logger.error(f"Summary error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500


def init_river_ai_routes(app, db):
    """Initialize River AI routes with app"""
    register_river_ai_routes(app, db)
    logger.info("River Memory AI routes registered")
