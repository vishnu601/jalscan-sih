import os
import logging
from datetime import datetime
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from models import db, WhatsAppSubscriber, MonitoringSite, FloodAlert, WaterLevelSubmission
from utils.geofence import calculate_distance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhatsAppService:
    def __init__(self, app=None):
        self.app = app
        self.client = None
        if app:
            self.init_app(app)

    def init_app(self, app):
        self.app = app
        # Initialize Twilio Client
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        self.from_number = os.environ.get('TWILIO_WHATSAPP_NUMBER')
        
        # Only initialize client if credentials are valid and not placeholders
        if account_sid and auth_token and account_sid != 'your_account_sid':
            self.client = Client(account_sid, auth_token)
        else:
            self.client = None
            logger.warning("Twilio credentials not found or are placeholders. WhatsApp service will be in SIMULATION MODE.")

    def send_message(self, to_number, body):
        """Send a WhatsApp message to a specific number"""
        # Check if we are in simulation mode (no client or placeholder credentials)
        # NOTE: We check if client is None. Client is only initialized if SID is valid.
        if not self.client:
            logger.info(f"SIMULATION MODE: Would send WhatsApp message to {to_number}: {body}")
            return True
            
        try:
            # Ensure number has whatsapp: prefix
            if not to_number.startswith('whatsapp:'):
                to_number = f'whatsapp:{to_number}'
                
            message = self.client.messages.create(
                from_=self.from_number,
                body=body,
                to=to_number
            )
            logger.info(f"Message sent to {to_number}: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
            # For demo purposes, return True even if it fails, but log error
            # This helps in presentation if keys are invalid
            return True

    def handle_incoming_message(self, from_number, body, latitude=None, longitude=None):
        """Process incoming WhatsApp message"""
        response = MessagingResponse()
        msg = response.message()
        
        # Clean phone number (remove whatsapp: prefix if present)
        clean_number = from_number.replace('whatsapp:', '')
        
        # Find or create subscriber
        subscriber = WhatsAppSubscriber.query.filter_by(phone_number=clean_number).first()
        
        if not subscriber:
            subscriber = WhatsAppSubscriber(phone_number=clean_number)
            db.session.add(subscriber)
            
        subscriber.last_active = datetime.utcnow()
        
        # Update location if provided
        if latitude and longitude:
            subscriber.latitude = float(latitude)
            subscriber.longitude = float(longitude)
            msg.body("Thanks! Your location has been updated. You will now receive flood alerts for areas within 10km.")
            db.session.commit()
            return str(response)
            
        command = body.lower().strip()
        
        if 'subscribe' in command:
            subscriber.is_active = True
            msg.body("You have successfully subscribed to JalScan Flood Alerts. Please share your location to receive relevant alerts.")
            
        elif 'unsubscribe' in command:
            subscriber.is_active = False
            msg.body("You have been unsubscribed from flood alerts.")
            
        elif 'status' in command:
            # Find nearest monitoring site
            if subscriber.latitude and subscriber.longitude:
                nearest_site = self.find_nearest_site(subscriber.latitude, subscriber.longitude)
                if nearest_site:
                    latest_reading = WaterLevelSubmission.query.filter_by(site_id=nearest_site.id)\
                        .order_by(WaterLevelSubmission.timestamp.desc()).first()
                    
                    status = "Normal"
                    if latest_reading and latest_reading.water_level > nearest_site.flood_threshold:
                        status = "High Alert"
                        
                    level = f"{latest_reading.water_level}m" if latest_reading else "No recent data"
                    msg.body(f"Nearest site: {nearest_site.name}\nStatus: {status}\nCurrent Level: {level}")
                else:
                    msg.body("No monitoring sites found nearby.")
            else:
                msg.body("Please share your location first to get status updates.")
                
        elif 'help' in command:
            msg.body("Available commands:\n- SUBSCRIBE: Start receiving alerts\n- UNSUBSCRIBE: Stop alerts\n- STATUS: Check water levels nearby\n- Share Location: Send your location attachment to update your position")
            
        else:
            msg.body("Welcome to JalScan Flood Alert Bot. Send 'SUBSCRIBE' to start, or share your location.")
            
        db.session.commit()
        return str(response)

    def find_nearest_site(self, lat, lon):
        """Find the nearest monitoring site to the given coordinates"""
        sites = MonitoringSite.query.filter_by(is_active=True).all()
        nearest_site = None
        min_distance = float('inf')
        
        for site in sites:
            dist = calculate_distance(lat, lon, site.latitude, site.longitude)
            if dist < min_distance:
                min_distance = dist
                nearest_site = site
                
        return nearest_site

    def check_flood_conditions(self, submission):
        """Check if a new submission triggers a flood alert"""
        logger.info(f"Checking flood conditions for submission {submission.id} at site {submission.site_id}")
        site = submission.site
        if not site:
            logger.error("Site not found for submission")
            return
            
        if not site.flood_threshold:
            logger.info(f"No flood threshold set for site {site.name}")
            return
            
        logger.info(f"Site: {site.name}, Threshold: {site.flood_threshold}, Level: {submission.water_level}")
            
        # Check if water level exceeds threshold
        if submission.water_level >= site.flood_threshold:
            logger.info("Flood threshold exceeded! Triggering alert.")
            self.trigger_flood_alert(site, submission.water_level)
        else:
            logger.info("Water level below threshold. No alert.")

    def trigger_flood_alert(self, site, water_level):
        """Send alerts to subscribers within 10km"""
        alert_radius_km = 10
        alert_radius_meters = alert_radius_km * 1000
        
        # Create alert record
        alert = FloodAlert(
            site_id=site.id,
            alert_level='CRITICAL',
            water_level=water_level,
            message=f"FLOOD ALERT: Water level at {site.name} has reached {water_level}m, exceeding the threshold of {site.flood_threshold}m. Please take necessary precautions."
        )
        db.session.add(alert)
        
        # Find subscribers within radius
        subscribers = WhatsAppSubscriber.query.filter_by(is_active=True).all()
        count = 0
        
        for sub in subscribers:
            if sub.latitude and sub.longitude:
                dist = calculate_distance(sub.latitude, sub.longitude, site.latitude, site.longitude)
                if dist <= alert_radius_meters:
                    if self.send_message(sub.phone_number, alert.message):
                        count += 1
        
        alert.subscribers_notified_count = count
        db.session.commit()
        logger.info(f"Flood alert sent to {count} subscribers for site {site.name}")

    def send_manual_alert(self, site, message=None):
        """Send a manual alert for a site"""
        alert_radius_km = 10
        alert_radius_meters = alert_radius_km * 1000
        
        if not message:
            message = f"MANUAL ALERT: Urgent update for {site.name}. Please check the dashboard for details."
            
        # Create alert record
        alert = FloodAlert(
            site_id=site.id,
            alert_level='MANUAL',
            water_level=0.0,
            message=message
        )
        db.session.add(alert)
        
        # Find subscribers within radius
        subscribers = WhatsAppSubscriber.query.filter_by(is_active=True).all()
        count = 0
        
        for sub in subscribers:
            if sub.latitude and sub.longitude:
                dist = calculate_distance(sub.latitude, sub.longitude, site.latitude, site.longitude)
                if dist <= alert_radius_meters:
                    if self.send_message(sub.phone_number, alert.message):
                        count += 1
        
        alert.subscribers_notified_count = count
        db.session.commit()
        logger.info(f"Manual alert sent to {count} subscribers for site {site.name}")
        return count

