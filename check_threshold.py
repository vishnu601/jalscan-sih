from app import create_app
from models import MonitoringSite

def check_site_threshold():
    app = create_app()
    with app.app_context():
        site = MonitoringSite.query.filter(MonitoringSite.name.like('%Musi%')).first()
        if site:
            print(f"Site: {site.name}")
            print(f"Flood Threshold: {site.flood_threshold} meters")
        else:
            print("Musi River site not found.")

if __name__ == "__main__":
    check_site_threshold()
