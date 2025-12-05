from app import create_app
from models import WhatsAppSubscriber, MonitoringSite
from utils.geofence import calculate_distance

def list_subscribers():
    app = create_app()
    with app.app_context():
        site = MonitoringSite.query.filter(MonitoringSite.name.like('%Musi%')).first()
        print(f"--- Checking Subscribers for {site.name} ---")
        print(f"Site Location: {site.latitude}, {site.longitude}")
        
        subscribers = WhatsAppSubscriber.query.all()
        for sub in subscribers:
            dist_km = -1
            if sub.latitude and sub.longitude:
                dist_meters = calculate_distance(sub.latitude, sub.longitude, site.latitude, site.longitude)
                dist_km = dist_meters / 1000.0
                
            status = "✅ IN RANGE" if dist_km >= 0 and dist_km <= 10 else "❌ OUT OF RANGE"
            print(f"User: {sub.phone_number} | Dist: {dist_km:.2f} km | {status}")

if __name__ == "__main__":
    list_subscribers()
