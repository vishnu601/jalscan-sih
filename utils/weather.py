import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_rainfall_prediction(lat, lon):
    """Fetch rainfall forecast from Open-Meteo and predict water level rise"""
    try:
        # Open-Meteo API: forecast for next 2 hours
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation&forecast_days=1"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            logger.error(f"Open-Meteo API error: {response.status_code}")
            return None

        data = response.json()
        
        if 'hourly' in data and 'precipitation' in data['hourly']:
            # Get current hour index (approximate)
            current_hour = datetime.now().hour
            
            # Get precipitation for next 2 hours (current hour + next hour)
            precip_data = data['hourly']['precipitation']
            
            next_2_hours_rain = 0
            if current_hour < len(precip_data):
                next_2_hours_rain += (precip_data[current_hour] or 0)
            if current_hour + 1 < len(precip_data):
                next_2_hours_rain += (precip_data[current_hour + 1] or 0)
            
            # Heuristic: 10mm rain -> 1m rise
            predicted_rise = (next_2_hours_rain / 10.0) * 1.0
            
            return {
                'rainfall_mm': round(next_2_hours_rain, 2),
                'predicted_rise_m': round(predicted_rise, 2),
                'period': 'Next 2 hours'
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return None
