from app import create_app
from models import db, WhatsAppSubscriber, MonitoringSite

def setup_demo_users():
    app = create_app()
    with app.app_context():
        print("--- Setting up Demo Subscribers ---")
        
        # Get Musi River site as reference
        site = MonitoringSite.query.filter(MonitoringSite.name.like('%Musi%')).first()
        if not site:
            print("Error: Musi River site not found!")
            return

        print(f"Reference Site: {site.name}")
        print(f"Location: {site.latitude}, {site.longitude}")

        # 1. Add "Near" Subscriber (approx 1km away)
        # Adding 0.01 to lat is roughly 1.1km
        near_lat = site.latitude + 0.005
        near_lon = site.longitude + 0.005
        
        sub_near = WhatsAppSubscriber.query.filter_by(phone_number='+91_TEST_NEAR').first()
        if not sub_near:
            sub_near = WhatsAppSubscriber(phone_number='+91_TEST_NEAR')
            print("Creating 'Near' subscriber...")
        
        sub_near.latitude = near_lat
        sub_near.longitude = near_lon
        sub_near.is_active = True
        db.session.add(sub_near)
        print(f"Subscriber 'Near' (+91_TEST_NEAR) set to: {near_lat}, {near_lon}")

        # 2. Add "Far" Subscriber (approx 50km away)
        far_lat = site.latitude + 0.5
        far_lon = site.longitude + 0.5
        
        sub_far = WhatsAppSubscriber.query.filter_by(phone_number='+91_TEST_FAR').first()
        if not sub_far:
            sub_far = WhatsAppSubscriber(phone_number='+91_TEST_FAR')
            print("Creating 'Far' subscriber...")
            
        sub_far.latitude = far_lat
        sub_far.longitude = far_lon
        sub_far.is_active = True
        db.session.add(sub_far)
        print(f"Subscriber 'Far' (+91_TEST_FAR) set to: {far_lat}, {far_lon}")

        db.session.commit()
        print("\n✅ Test subscribers added successfully!")
        print("You can now trigger an alert for 'Musi River' and check the logs.")
        print("Only +91_TEST_NEAR should receive the alert.")

if __name__ == "__main__":
    setup_demo_users()
