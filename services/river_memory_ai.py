"""
River Memory AI - Digital Twin for River Monitoring
AI-powered analysis of water images to build long-term river memory

Features:
1. Water Color Signature Analysis - sediment, algae, pollution detection
2. Flow Speed Estimation - surface texture analysis
3. Gauge Damage Detection - condition monitoring
4. Unusual Behavior Detection - anomaly flagging
5. Erosion Tracking - bank change detection
"""

import os
import logging
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

# Gemini API for image analysis
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. River Memory AI will use mock analysis.")


class RiverMemoryAI:
    """
    AI engine for analyzing river images and building digital memory.
    Uses Gemini Vision for advanced image understanding.
    """
    
    # Sediment type color signatures (approximate RGB ranges)
    SEDIMENT_SIGNATURES = {
        "clear": {"description": "Clear water, low sediment", "color_hint": "blue-green"},
        "silt": {"description": "Silt-laden, moderate sediment", "color_hint": "brown-tan"},
        "muddy": {"description": "Heavy mud/flood sediment", "color_hint": "dark brown"},
        "algae": {"description": "Algae presence", "color_hint": "green tint"},
        "pollution": {"description": "Possible industrial pollution", "color_hint": "unusual colors"},
    }
    
    # Flow speed classifications
    FLOW_CLASSES = {
        "still": {"score": 0, "risk": "low", "description": "Very slow or stagnant"},
        "low": {"score": 1, "risk": "low", "description": "Gentle flow"},
        "moderate": {"score": 2, "risk": "medium", "description": "Normal flow"},
        "high": {"score": 3, "risk": "medium", "description": "Fast flow"},
        "turbulent": {"score": 4, "risk": "high", "description": "Turbulent - possible flash flood precursor"},
    }
    
    def __init__(self):
        self.model = None
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini Vision model"""
        if not GEMINI_AVAILABLE:
            return
        
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set. Using mock analysis.")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("River Memory AI initialized with Gemini Vision")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    
    def analyze_image(self, image_path: str, site_id: int = None) -> Dict:
        """
        Perform comprehensive river analysis on an image.
        
        Args:
            image_path: Path to the image file
            site_id: Optional site ID for historical comparison
            
        Returns:
            Dictionary with all analysis results
        """
        if not os.path.exists(image_path):
            return self._error_result(f"Image not found: {image_path}")
        
        # Read image
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
        except Exception as e:
            return self._error_result(f"Could not read image: {e}")
        
        # Use Gemini if available, otherwise mock
        if self.model:
            return self._analyze_with_gemini(image_data, site_id)
        else:
            return self._mock_analysis(site_id)
    
    def _analyze_with_gemini(self, image_data: bytes, site_id: int = None) -> Dict:
        """Analyze image using Gemini Vision API"""
        
        prompt = """Analyze this river/water gauge image for the following aspects. 
        Provide a JSON response with these exact keys:

        {
            "water_color": {
                "dominant_color": "describe the main water color",
                "rgb_estimate": [R, G, B],
                "sediment_type": "clear|silt|muddy|algae|pollution",
                "sediment_level": "low|medium|high",
                "pollution_detected": true/false,
                "pollution_description": "description if detected"
            },
            "flow_analysis": {
                "flow_class": "still|low|moderate|high|turbulent",
                "surface_texture": "smooth|rippled|choppy|turbulent",
                "turbulence_score": 0-100,
                "flash_flood_risk": "low|medium|high"
            },
            "gauge_condition": {
                "gauge_visible": true/false,
                "visibility_score": 0-100,
                "damage_detected": true/false,
                "damage_types": ["faded_numbers", "broken_tiles", "algae_buildup", "tilt", "submergence"],
                "condition_notes": "description of gauge condition"
            },
            "anomalies": {
                "anomaly_detected": true/false,
                "anomaly_type": "unusual_color|unusual_level|debris|obstruction|none",
                "anomaly_description": "detailed description",
                "severity": "low|medium|high"
            },
            "water_level": {
                "estimated_level_cm": number or null,
                "confidence": 0-100
            },
            "riverbank": {
                "visible": true/false,
                "erosion_signs": true/false,
                "erosion_description": "description if visible"
            },
            "overall_risk": "low|medium|high",
            "summary": "One-line summary of the analysis"
        }

        Be accurate and objective. If you cannot determine something, use null or "unknown"."""

        try:
            # Create image part for Gemini
            image_part = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_data).decode('utf-8')
            }
            
            response = self.model.generate_content([prompt, image_part])
            
            # Parse JSON from response
            text = response.text
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                result = json.loads(json_match.group())
                result['analysis_source'] = 'gemini'
                result['analyzed_at'] = datetime.utcnow().isoformat()
                return result
            else:
                logger.warning("Could not parse JSON from Gemini response")
                return self._mock_analysis(site_id)
                
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self._mock_analysis(site_id)
    
    def _mock_analysis(self, site_id: int = None) -> Dict:
        """Return mock analysis when Gemini is unavailable"""
        import random
        
        return {
            "water_color": {
                "dominant_color": "brown-green",
                "rgb_estimate": [random.randint(80, 150), random.randint(100, 160), random.randint(60, 120)],
                "sediment_type": random.choice(["clear", "silt", "muddy"]),
                "sediment_level": random.choice(["low", "medium", "high"]),
                "pollution_detected": random.choice([True, False]),
                "pollution_description": None
            },
            "flow_analysis": {
                "flow_class": random.choice(["low", "moderate", "high"]),
                "surface_texture": random.choice(["smooth", "rippled", "choppy"]),
                "turbulence_score": random.randint(10, 60),
                "flash_flood_risk": "low"
            },
            "gauge_condition": {
                "gauge_visible": True,
                "visibility_score": random.randint(60, 95),
                "damage_detected": random.choice([True, False]),
                "damage_types": random.choice([[], ["faded_numbers"], ["algae_buildup"]]),
                "condition_notes": "Gauge appears in acceptable condition"
            },
            "anomalies": {
                "anomaly_detected": random.choice([True, False]),
                "anomaly_type": random.choice(["none", "unusual_color", "debris"]),
                "anomaly_description": "Minor debris observed near gauge",
                "severity": "low"
            },
            "water_level": {
                "estimated_level_cm": random.randint(100, 400),
                "confidence": random.randint(70, 95)
            },
            "riverbank": {
                "visible": True,
                "erosion_signs": random.choice([True, False]),
                "erosion_description": None
            },
            "overall_risk": random.choice(["low", "medium"]),
            "summary": "Standard river conditions observed with minor variations",
            "analysis_source": "mock",
            "analyzed_at": datetime.utcnow().isoformat()
        }
    
    def _error_result(self, error_msg: str) -> Dict:
        """Return error result"""
        return {
            "error": True,
            "error_message": error_msg,
            "analyzed_at": datetime.utcnow().isoformat()
        }
    
    def compare_with_baseline(
        self, 
        current_analysis: Dict, 
        baseline_analysis: Dict
    ) -> Dict:
        """
        Compare current analysis with baseline to detect changes.
        
        Returns:
            Dictionary with change detection results
        """
        if not baseline_analysis or "error" in baseline_analysis:
            return {"comparison_available": False}
        
        changes = {
            "comparison_available": True,
            "changes_detected": False,
            "water_color_change": False,
            "flow_change": False,
            "gauge_condition_change": False,
            "erosion_change": False,
            "alerts": []
        }
        
        # Compare water color
        curr_sediment = current_analysis.get("water_color", {}).get("sediment_type", "")
        base_sediment = baseline_analysis.get("water_color", {}).get("sediment_type", "")
        if curr_sediment != base_sediment:
            changes["water_color_change"] = True
            changes["changes_detected"] = True
            changes["alerts"].append(f"Water color changed from {base_sediment} to {curr_sediment}")
        
        # Compare flow
        curr_flow = current_analysis.get("flow_analysis", {}).get("flow_class", "")
        base_flow = baseline_analysis.get("flow_analysis", {}).get("flow_class", "")
        if curr_flow != base_flow:
            changes["flow_change"] = True
            changes["changes_detected"] = True
            flow_risk = self.FLOW_CLASSES.get(curr_flow, {}).get("risk", "unknown")
            if flow_risk == "high":
                changes["alerts"].append(f"⚠️ Flow increased to TURBULENT - flash flood risk!")
            else:
                changes["alerts"].append(f"Flow changed from {base_flow} to {curr_flow}")
        
        # Compare gauge condition
        curr_visibility = current_analysis.get("gauge_condition", {}).get("visibility_score", 100)
        base_visibility = baseline_analysis.get("gauge_condition", {}).get("visibility_score", 100)
        if abs(curr_visibility - base_visibility) > 20:
            changes["gauge_condition_change"] = True
            changes["changes_detected"] = True
            changes["alerts"].append(f"Gauge visibility changed: {base_visibility}% → {curr_visibility}%")
        
        # Check erosion
        curr_erosion = current_analysis.get("riverbank", {}).get("erosion_signs", False)
        base_erosion = baseline_analysis.get("riverbank", {}).get("erosion_signs", False)
        if curr_erosion and not base_erosion:
            changes["erosion_change"] = True
            changes["changes_detected"] = True
            changes["alerts"].append("🚨 New erosion signs detected at riverbank!")
        
        return changes
    
    def get_site_memory(self, site_id: int, days: int = 30) -> Dict:
        """
        Get aggregated memory for a site over time.
        
        Args:
            site_id: Monitoring site ID
            days: Number of days to look back
            
        Returns:
            Aggregated site memory with trends
        """
        from models import RiverAnalysis, MonitoringSite
        
        site = MonitoringSite.query.get(site_id)
        if not site:
            return {"error": "Site not found"}
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        analyses = RiverAnalysis.query.filter(
            RiverAnalysis.site_id == site_id,
            RiverAnalysis.timestamp >= cutoff
        ).order_by(RiverAnalysis.timestamp.desc()).all()
        
        if not analyses:
            return {
                "site_id": site_id,
                "site_name": site.name,
                "memory_available": False,
                "message": "No historical data available for this site"
            }
        
        # Aggregate statistics
        sediment_counts = {}
        flow_counts = {}
        anomaly_count = 0
        avg_visibility = 0
        erosion_detected = False
        
        for a in analyses:
            analysis_data = json.loads(a.ai_analysis_json) if a.ai_analysis_json else {}
            
            # Sediment
            sed_type = analysis_data.get("water_color", {}).get("sediment_type", "unknown")
            sediment_counts[sed_type] = sediment_counts.get(sed_type, 0) + 1
            
            # Flow
            flow_class = analysis_data.get("flow_analysis", {}).get("flow_class", "unknown")
            flow_counts[flow_class] = flow_counts.get(flow_class, 0) + 1
            
            # Anomalies
            if analysis_data.get("anomalies", {}).get("anomaly_detected"):
                anomaly_count += 1
            
            # Visibility
            avg_visibility += analysis_data.get("gauge_condition", {}).get("visibility_score", 0)
            
            # Erosion
            if analysis_data.get("riverbank", {}).get("erosion_signs"):
                erosion_detected = True
        
        avg_visibility = avg_visibility / len(analyses) if analyses else 0
        
        return {
            "site_id": site_id,
            "site_name": site.name,
            "memory_available": True,
            "analysis_count": len(analyses),
            "period_days": days,
            "sediment_distribution": sediment_counts,
            "flow_distribution": flow_counts,
            "anomaly_frequency": anomaly_count / len(analyses) if analyses else 0,
            "average_gauge_visibility": round(avg_visibility, 1),
            "erosion_ever_detected": erosion_detected,
            "latest_analysis": analyses[0].to_dict() if analyses else None,
            "trend_summary": self._generate_trend_summary(analyses)
        }
    
    def _generate_trend_summary(self, analyses: List) -> str:
        """Generate a human-readable trend summary"""
        if not analyses:
            return "No data available"
        
        if len(analyses) < 3:
            return "Insufficient data for trend analysis"
        
        # Simple trend detection
        recent = analyses[:len(analyses)//3]
        older = analyses[len(analyses)//3:]
        
        summaries = []
        
        # Check flow trend
        recent_turbulent = sum(1 for a in recent if 
            json.loads(a.ai_analysis_json or "{}").get("flow_analysis", {}).get("flow_class") == "turbulent")
        older_turbulent = sum(1 for a in older if 
            json.loads(a.ai_analysis_json or "{}").get("flow_analysis", {}).get("flow_class") == "turbulent")
        
        if recent_turbulent > older_turbulent:
            summaries.append("🔺 Flow intensity increasing")
        elif recent_turbulent < older_turbulent:
            summaries.append("🔻 Flow intensity decreasing")
        
        return "; ".join(summaries) if summaries else "Stable conditions observed"


# Singleton instance
river_memory_ai = RiverMemoryAI()


def analyze_submission(submission_id: int) -> Dict:
    """
    Analyze a water level submission and store results.
    
    Args:
        submission_id: WaterLevelSubmission ID
        
    Returns:
        Analysis results
    """
    from models import WaterLevelSubmission, RiverAnalysis, db
    import json
    
    submission = WaterLevelSubmission.query.get(submission_id)
    if not submission:
        return {"error": "Submission not found"}
    
    # Build image path
    image_path = os.path.join('static', 'uploads', submission.photo_filename)
    if not os.path.exists(image_path):
        image_path = os.path.join('uploads', submission.photo_filename)
    
    # Run analysis
    analysis = river_memory_ai.analyze_image(image_path, submission.site_id)
    
    if "error" in analysis:
        return analysis
    
    # Get baseline for comparison
    baseline = RiverAnalysis.query.filter(
        RiverAnalysis.site_id == submission.site_id,
        RiverAnalysis.id != submission_id
    ).order_by(RiverAnalysis.timestamp.desc()).first()
    
    baseline_data = json.loads(baseline.ai_analysis_json) if baseline and baseline.ai_analysis_json else None
    comparison = river_memory_ai.compare_with_baseline(analysis, baseline_data)
    
    # Store analysis
    try:
        river_analysis = RiverAnalysis(
            submission_id=submission_id,
            site_id=submission.site_id,
            timestamp=submission.timestamp,
            water_color_rgb=json.dumps(analysis.get("water_color", {}).get("rgb_estimate", [])),
            sediment_type=analysis.get("water_color", {}).get("sediment_type"),
            pollution_index=1.0 if analysis.get("water_color", {}).get("pollution_detected") else 0.0,
            flow_speed_class=analysis.get("flow_analysis", {}).get("flow_class"),
            turbulence_score=analysis.get("flow_analysis", {}).get("turbulence_score", 0),
            gauge_visibility_score=analysis.get("gauge_condition", {}).get("visibility_score", 0),
            gauge_damage_detected=analysis.get("gauge_condition", {}).get("damage_detected", False),
            damage_type=",".join(analysis.get("gauge_condition", {}).get("damage_types", [])),
            anomaly_detected=analysis.get("anomalies", {}).get("anomaly_detected", False),
            anomaly_type=analysis.get("anomalies", {}).get("anomaly_type"),
            anomaly_description=analysis.get("anomalies", {}).get("anomaly_description"),
            erosion_detected=analysis.get("riverbank", {}).get("erosion_signs", False),
            overall_risk=analysis.get("overall_risk", "unknown"),
            ai_analysis_json=json.dumps(analysis)
        )
        db.session.add(river_analysis)
        db.session.commit()
        
        analysis["stored"] = True
        analysis["analysis_id"] = river_analysis.id
        analysis["comparison"] = comparison
        
    except Exception as e:
        logger.error(f"Failed to store analysis: {e}")
        analysis["stored"] = False
        analysis["store_error"] = str(e)
    
    return analysis
