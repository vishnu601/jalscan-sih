import requests
from app import create_app, db
from models import User, MonitoringSite, UserSite

def test_submission():
    app = create_app()
    with app.app_context():
        # 1. Get or Create a Test User
        from werkzeug.security import generate_password_hash
        user = User.query.filter_by(username='test_admin').first()
        if not user:
            user = User(username='test_admin', email='test@example.com', role='admin')
            user.password_hash = generate_password_hash('password')
            db.session.add(user)
            db.session.commit()
            print("Created test_admin user")
        
        # 2. Get Musi River Site
        site = MonitoringSite.query.filter(MonitoringSite.name.like('%Musi%')).first()
        if not site:
            print("Error: Musi River site not found")
            return
            
        print(f"Submitting for site: {site.name} (ID: {site.id})")
        
        # 3. Simulate Login to get Session Cookie
        session = requests.Session()
        login_url = 'http://127.0.0.1:5000/auth/login'
        login_data = {'username': 'test_admin', 'password': 'password'}
        
        # We need to handle CSRF if enabled, but for now let's try direct login
        # Note: This might fail if CSRF protection is strict. 
        # A better way is to use the app's test client.
        
    # Configure logging to see app logs
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Using Flask Test Client instead of requests to avoid running server issues
    with app.test_client() as client:
        # Login
        print("Logging in...")
        login_resp = client.post('/login', data={'username': 'test_admin', 'password': 'password'}, follow_redirects=True)
        print(f"Login Response: {login_resp.status_code}")
        
        # Submit Reading
        data = {
            'site_id': site.id,
            'water_level': 12.0,  # Above 10.0 threshold
            'latitude': site.latitude,
            'longitude': site.longitude,
            'verification_method': 'gps',
            'notes': 'Test High Water Level',
            'quality_rating': 5
        }
        
        # We need to simulate a file upload too
        from io import BytesIO
        data['photo'] = (BytesIO(b"fake image data"), 'test.jpg')
        
        print("Sending POST request to /api/submit-reading...")
        response = client.post('/api/submit-reading', data=data, content_type='multipart/form-data', follow_redirects=True)
        
        print(f"Response Status: {response.status_code}")
        # print(f"Response Body: {response.get_data(as_text=True)}") # Too verbose
        
        if response.status_code == 200:
            print("✅ Submission request completed (followed redirects).")
            print("Check the logs above for 'Checking flood conditions...'")
        else:
            print("❌ Submission failed!")

if __name__ == "__main__":
    test_submission()
