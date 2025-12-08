"""
JalScan GPT - Conversational Flood Risk Assistant
Natural language interface for flood predictions and water level queries
"""

import re
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple, List

logger = logging.getLogger(__name__)


class JalScanGPT:
    """
    Conversational assistant for flood risk queries.
    Wraps the ML prediction API with natural language understanding.
    """
    
    # Common site name aliases
    SITE_ALIASES = {
        "ganga": ["ganges", "ganga river", "ganga nadi"],
        "yamuna": ["jamuna", "yamuna river"],
        "godavari": ["godavari river"],
        "krishna": ["krishna river"],
        "brahmaputra": ["brahmaputra river"],
        "musi": ["musi river", "moosi"],
        "sabarmati": ["sabarmati river"],
        "narmada": ["narmada river"]
    }
    
    # Intent patterns
    INTENT_PATTERNS = {
        "flood_risk": [
            r"flood\s*risk",
            r"is\s*there\s*(a\s*)?flood",
            r"flooding",
            r"will\s*it\s*flood",
            r"chance\s*of\s*flood"
        ],
        "flash_flood": [
            r"flash\s*flood",
            r"sudden\s*flood",
            r"rapid\s*rise"
        ],
        "water_level": [
            r"water\s*level",
            r"current\s*level",
            r"how\s*(high|much)",
            r"water\s*height"
        ],
        "prediction": [
            r"predict",
            r"forecast",
            r"next\s*(\d+)?\s*hours?",
            r"today",
            r"tomorrow"
        ],
        "explanation": [
            r"why",
            r"reason",
            r"explain",
            r"how\s*confident",
            r"confidence"
        ],
        "help": [
            r"help",
            r"what\s*can\s*you",
            r"commands?"
        ]
    }
    
    def __init__(self):
        self.predictor = None
    
    def _get_predictor(self):
        """Lazy load the predictor"""
        if self.predictor is None:
            from ml.model_inference import get_predictor
            self.predictor = get_predictor()
        return self.predictor
    
    def answer_flood_query(
        self,
        user_message: str,
        user_context: Optional[Dict] = None
    ) -> str:
        """
        Main entry point for answering user queries.
        
        Args:
            user_message: Natural language query from user
            user_context: Optional context (location, preferences, etc.)
            
        Returns:
            Response string
        """
        message = user_message.lower().strip()
        
        # Check for help intent
        if self._matches_intent(message, "help"):
            return self._help_response()
        
        # Extract site name from message
        site_name, site_id = self._extract_site(message, user_context)
        
        if site_id is None:
            # Try to use context or ask for site
            if user_context and user_context.get("last_site_id"):
                site_id = user_context["last_site_id"]
                site_name = user_context.get("last_site_name", "your location")
            else:
                return self._ask_for_site()
        
        # Determine intent
        if self._matches_intent(message, "explanation"):
            return self._explain_prediction(site_id, site_name)
        elif self._matches_intent(message, "flash_flood"):
            return self._flash_flood_check(site_id, site_name)
        elif self._matches_intent(message, "water_level"):
            return self._water_level_query(site_id, site_name)
        else:
            # Default: flood risk prediction
            return self._flood_risk_response(site_id, site_name)
    
    def _matches_intent(self, message: str, intent: str) -> bool:
        """Check if message matches an intent pattern"""
        patterns = self.INTENT_PATTERNS.get(intent, [])
        for pattern in patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True
        return False
    
    def _extract_site(
        self,
        message: str,
        context: Optional[Dict]
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Extract monitoring site from message.
        
        Returns:
            (site_name, site_id) or (None, None) if not found
        """
        from models import MonitoringSite
        
        # Check aliases first
        for canonical, aliases in self.SITE_ALIASES.items():
            if canonical in message.lower():
                sites = MonitoringSite.query.filter(
                    MonitoringSite.name.ilike(f"%{canonical}%")
                ).first()
                if sites:
                    return sites.name, sites.id
            
            for alias in aliases:
                if alias in message.lower():
                    sites = MonitoringSite.query.filter(
                        MonitoringSite.name.ilike(f"%{canonical}%")
                    ).first()
                    if sites:
                        return sites.name, sites.id
        
        # Try to find any site name in the message
        all_sites = MonitoringSite.query.filter_by(is_active=True).all()
        for site in all_sites:
            site_words = site.name.lower().split()
            for word in site_words:
                if len(word) > 3 and word in message.lower():
                    return site.name, site.id
        
        return None, None
    
    def _flood_risk_response(self, site_id: int, site_name: str) -> str:
        """Generate flood risk prediction response"""
        from ml.schemas import PredictionRequest, RiskCategory
        
        predictor = self._get_predictor()
        request = PredictionRequest(monitoring_site_id=site_id)
        response = predictor.predict(request)
        
        category = response.risk_category
        score = response.risk_score
        
        # Format response based on risk level
        if category == RiskCategory.FLASH_FLOOD_RISK:
            msg = f"""🚨 *FLASH FLOOD ALERT* at {site_name}!

⚠️ Risk Level: CRITICAL
📊 Confidence: {score*100:.0f}%

{chr(10).join(response.explanations[:3])}

*Immediate Actions:*
{chr(10).join('• ' + r for r in response.recommendations[:3])}

_This is a model prediction. Follow official alerts._"""
        
        elif category == RiskCategory.FLOOD_RISK:
            msg = f"""⚠️ *HIGH FLOOD RISK* at {site_name}

Risk Level: HIGH
Confidence: {score*100:.0f}%

{chr(10).join(response.explanations[:3])}

*Recommended Actions:*
{chr(10).join('• ' + r for r in response.recommendations[:3])}

_Model prediction. Stay alert to official updates._"""
        
        elif category == RiskCategory.CAUTION:
            msg = f"""🟡 *CAUTION* for {site_name}

Risk Level: MODERATE
Confidence: {score*100:.0f}%

{response.explanations[0] if response.explanations else 'Elevated water levels detected.'}

*Recommendations:*
{chr(10).join('• ' + r for r in response.recommendations[:2])}

_Monitor conditions and check back later._"""
        
        else:  # SAFE
            msg = f"""✅ *{site_name}* - Currently Safe

Risk Level: LOW
{response.explanations[0] if response.explanations else 'Water levels are within normal range.'}

No immediate flood risk detected. Continue normal activities.

_Predictions update every 6 hours._"""
        
        return msg
    
    def _flash_flood_check(self, site_id: int, site_name: str) -> str:
        """Specific flash flood risk check"""
        from ml.schemas import PredictionRequest, RiskCategory
        
        predictor = self._get_predictor()
        request = PredictionRequest(monitoring_site_id=site_id)
        response = predictor.predict(request)
        
        if response.risk_category == RiskCategory.FLASH_FLOOD_RISK:
            return f"""🚨 *YES - Flash Flood Risk Detected* at {site_name}!

Rapid water level rise detected.
{response.explanations[0] if response.explanations else ''}

Take immediate precautions!"""
        else:
            slope = response.key_factors.get("slope_1h", 0)
            if slope > 20:
                return f"""⚠️ Water levels at {site_name} are rising faster than normal.

Rate: ~{slope:.0f} cm/hour

Not yet flash flood territory, but monitoring closely."""
            else:
                return f"""✅ No flash flood risk at {site_name} currently.

Water levels are stable or rising slowly."""
    
    def _water_level_query(self, site_id: int, site_name: str) -> str:
        """Current water level query"""
        from ml.schemas import PredictionRequest
        
        predictor = self._get_predictor()
        request = PredictionRequest(monitoring_site_id=site_id)
        response = predictor.predict(request)
        
        level = response.key_factors.get("water_level_cm", 0)
        pct_danger = response.key_factors.get("pct_of_danger_threshold", 0)
        
        if level > 0:
            return f"""📏 *Water Level at {site_name}*

Current Level: {level:.0f} cm
Danger Threshold: {pct_danger:.0f}%

Status: {response.risk_category.value}

Last updated: {response.timestamp.strftime('%H:%M %d-%b')}"""
        else:
            return f"""📏 *Water Level at {site_name}*

No recent readings available.

Please check back later or verify sensor status."""
    
    def _explain_prediction(self, site_id: int, site_name: str) -> str:
        """Explain the prediction in detail"""
        from ml.schemas import PredictionRequest
        
        predictor = self._get_predictor()
        request = PredictionRequest(monitoring_site_id=site_id)
        response = predictor.predict(request)
        
        explanations = response.explanations
        factors = response.key_factors
        
        factors_str = "\n".join([
            f"• {k.replace('_', ' ').title()}: {v:.1f}"
            for k, v in list(factors.items())[:5]
        ])
        
        return f"""🔍 *Prediction Explanation for {site_name}*

Risk: {response.risk_category.value}
Confidence: {response.confidence*100:.0f}%
Model Version: {response.model_version}

*Why this prediction:*
{chr(10).join('• ' + e for e in explanations[:4])}

*Key Factors:*
{factors_str}

_Model evaluates 24 features including water dynamics, temporal patterns, and site history._"""
    
    def _help_response(self) -> str:
        """Return help message"""
        return """🌊 *JalScan GPT - Flood Risk Assistant*

I can help you with:

📊 *Flood Risk Queries*
• "What is the flood risk at Ganga River?"
• "Is there flood risk at [site name]?"

⚡ *Flash Flood Alerts*
• "Is there flash flood risk near Yamuna?"

📏 *Water Level Info*
• "Current water level at [site]"
• "How high is the water at [location]?"

🔍 *Explanations*
• "Why are you predicting high risk?"
• "Explain the prediction for [site]"

Just mention the river or site name in your question!

_Powered by JalScan AI_"""
    
    def _ask_for_site(self) -> str:
        """Ask user to specify a site"""
        from models import MonitoringSite
        
        sites = MonitoringSite.query.filter_by(is_active=True).limit(5).all()
        site_names = [s.name for s in sites]
        
        return f"""Please specify a monitoring site. 

For example:
• "Flood risk at Ganga River"
• "Water level at {site_names[0] if site_names else 'Site Name'}"

Available sites include: {', '.join(site_names[:5])}..."""


# Singleton instance
jalscan_gpt = JalScanGPT()


def answer_query(message: str, context: Optional[Dict] = None) -> str:
    """Convenience function for answering queries"""
    return jalscan_gpt.answer_flood_query(message, context)
