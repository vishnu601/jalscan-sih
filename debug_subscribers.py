from app import create_app
from models import db, WhatsAppSubscriber, MonitoringSite, FloodAlert
from utils.geofence import calculate_distance

def debug_subscribers():
    app = create_app()
    with app.app_context():
        print("--- Debugging Subscribers ---")
        subscribers = WhatsAppSubscriber.query.all()
        print(f"Total Subscribers: {len(subscribers)}")
        
        for sub in subscribers:
            print(f"ID: {sub.id}, Phone: {sub.phone_number}, Active: {sub.is_active}")
            print(f"  Location: {sub.latitude}, {sub.longitude}")
            
        print("\n--- Debugging Sites ---")
        sites = MonitoringSite.query.filter_by(is_active=True).all()
        for site in sites:
            print(f"Site: {site.name} ({site.latitude}, {site.longitude})")
            
            # Check distance to subscribers
            print("  Distances to subscribers:")
            for sub in subscribers:
                if sub.latitude and sub.longitude:
                    dist = calculate_distance(sub.latitude, sub.longitude, site.latitude, site.longitude)
                    print(f"    -> {sub.phone_number}: {dist:.2f} meters ({dist/1000:.2f} km)")
                    if dist <= 10000:
                        print("       [MATCH] Within 10km radius")
                else:
                    print(f"    -> {sub.phone_number}: No location data")

        print("\n--- Recent Flood Alerts ---")
        alerts = FloodAlert.query.order_by(FloodAlert.created_at.desc()).limit(5).all()
        for alert in alerts:
            print(f"Alert ID: {alert.id}, Site: {alert.site.name}, Level: {alert.alert_level}")
            print(f"  Message: {alert.message}")
            print(f"  Notified: {alert.subscribers_notified_count}")
            print(f"  Created: {alert.created_at}")

if __name__ == "__main__":
    debug_subscribers()
