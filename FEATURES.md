# JalScan - Water Level Monitoring System

A comprehensive AI-powered water level monitoring and flood prediction system for the Smart India Hackathon (SIH).

---

## 🚀 Features

### 📷 Water Level Capture & Submission
- **GPS-verified photo capture** with live camera feed
- **AI-powered water level detection** using Google Gemini Vision API
- **AI Integration**: Uses Google Gemini (`GOOGLE_API_KEY`) for water gauge reading and chatbot assistance.
- **Geofencing validation** ensures submissions from authorized locations
- **Offline-first architecture** with automatic sync when online
- **QR code verification** for site authentication

### 📊 Analytics Dashboard
- **Interactive charts** using Chart.js
  - Submissions over time
  - Water level trends
  - Top contributors
  - Submissions by site (pie chart)
  - Quality rating distribution
- **Date range filtering** (7, 14, 30, 90 days)
- **Real-time data updates**

### 🛡️ Tamper Detection & Security
- **AI-powered tamper analysis** on submissions
- **Tamper score calculation** (0-1 scale)
- **Review workflow** for pending/confirmed/false positives
- **Trend visualization** for security metrics
- **Detection by type and severity** breakdown charts
- **Batch analysis** capability

### 📞 AI Call Reporting (Voice Integration)
- **Twilio voice call integration** for field reports
- **Speech-to-text water level input**
- **External service sync** from Replit-hosted call agent
- **Auto-import** of voice submissions to database
- **Separate Twilio credentials** for voice vs WhatsApp

### 💬 WhatsApp Bot
- **Twilio WhatsApp integration**
- **Flood alert notifications** to subscribers
- **Interactive commands** for water level queries
- **10km radius flood predictions**

### 🗺️ Site Management
- **Monitoring site CRUD operations**
- **QR code generation** for each site
- **User assignment** to sites
- **Manual flood alerts** triggering
- **GPS coordinates** with map integration

### ☁️ Cloud Dashboard
- **Real-time sync status**
- **Cloud submission tracking**
- **Sync failure monitoring**

### 👥 User Management
- **Role-based access control** (Admin, Supervisor, Central Analyst, Field Agent)
- **Permission management**
- **User activity tracking**
- **Public image submissions** for citizen reports

### 🔔 Flood Alerts
- **Threshold-based alerts** per site
- **Multi-channel notifications** (WhatsApp, Email)
- **Alert history tracking**

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.13** | Core language |
| **Flask** | Web framework |
| **Flask-SQLAlchemy** | ORM for database |
| **SQLite** | Development database |
| **Flask-Login** | Authentication |
| **Flask-Migrate** | Database migrations |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5/CSS3** | Structure & styling |
| **Bootstrap 5** | UI framework |
| **Bootstrap Icons** | Icon library |
| **JavaScript (ES6+)** | Client-side logic |
| **Chart.js** | Interactive charts |
| **Jinja2** | Templating engine |

### AI & APIs
| Technology | Purpose |
|------------|---------|
| **Google Gemini API** | Water level detection from images |
| **Twilio Voice API** | Phone call integration |
| **Twilio WhatsApp API** | Messaging integration |
| **Ngrok** | Local tunnel for webhooks |

### Storage & Sync
| Technology | Purpose |
|------------|---------|
| **Local file storage** | Image uploads |
| **SQLite** | Submission data |
| **Background sync service** | Offline-to-cloud sync |

### Development Tools
| Tool | Purpose |
|------|---------|
| **Git/GitHub** | Version control |
| **ngrok** | Expose local server |
| **dotenv** | Environment variables |

---

## 📱 PWA Features
- **Service Worker** for offline support
- **Web App Manifest** for installability
- **Responsive design** for mobile
- **Bottom navigation** for mobile users

---

## 🔐 Security Features
- Password hashing with Werkzeug
- Session-based authentication
- Role-based permissions
- Tamper detection engine
- Location verification

---

## 📁 Project Structure

```
jalscan-sih/
├── app.py                 # Main Flask application
├── models.py              # Database models
├── tamper_detection.py    # Tamper detection engine
├── sync_service.py        # Cloud sync service
├── whatsapp_service.py    # WhatsApp integration
├── templates/             # Jinja2 HTML templates
├── static/                # CSS, JS, images
├── uploads/               # User uploaded images
├── instance/              # SQLite database
└── .env                   # Environment variables
```

---

## 🌐 Environment Variables

```env
GOOGLE_API_KEY=your_gemini_api_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=your_twilio_number
VOICE_TWILIO_ACCOUNT_SID=voice_twilio_sid
VOICE_TWILIO_AUTH_TOKEN=voice_twilio_token
VOICE_TWILIO_PHONE_NUMBER=voice_phone_number
```

---

## 🚀 Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env

# Run the application
python3 app.py

# Expose via ngrok (for webhooks)
ngrok http 80
```

---

## 👨‍💻 Author

**Vishnu M** | Smart India Hackathon 2024

---

*Last updated: December 2024*
