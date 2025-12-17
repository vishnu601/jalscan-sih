# JalScan - AI-Powered Water Level Monitoring System

> **Smart India Hackathon 2024** | Advanced Flood Prediction & River Monitoring Platform

JalScan is a next-generation water monitoring solution designed to provide real-time flood risk assessment, secure data collection, and predictive analytics. It combines offline-first PWA capabilities with cutting-edge AI to empower field agents and decision-makers.

---

## 🔄 Application Flow

### System Architecture
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           JalScan System Flow                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Field Agent │────▶│  PWA Client  │────▶│ Flask Server │────▶│   Database   │
│   (Mobile)   │     │  (Browser)   │     │   (Backend)  │     │   (SQLite)   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  ┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐
  │ GPS/    │         │ IndexedDB│         │ Gemini  │         │ Reports │
  │ Camera  │         │ (Offline)│         │ AI API  │         │  (CSV)  │
  └─────────┘         └─────────┘         └─────────┘         └─────────┘
```

### Data Collection Flow
```
1. AUTHENTICATION
   └──▶ User logs in with role-based credentials
        └──▶ Role determines accessible features & sites

2. SITE VERIFICATION
   └──▶ Field agent navigates to monitoring site
        └──▶ GPS geofence check (±50m radius)
             └──▶ QR code scan (optional site verification)

3. DATA CAPTURE
   └──▶ Camera captures water gauge image
        └──▶ Gemini AI reads water level (OCR)
             └──▶ Agent confirms/corrects reading
                  └──▶ Photo saved with metadata

4. SUBMISSION
   └──▶ Data packaged (level, GPS, timestamp, photo)
        └──▶ Offline? → Store in IndexedDB
        └──▶ Online? → Send to /api/submit-reading
             └──▶ Tamper detection analysis
                  └──▶ Quality score calculation
                       └──▶ Flood risk check

5. SYNC & ALERTS
   └──▶ Background sync when online
        └──▶ If flood risk detected:
             └──▶ WhatsApp alert to subscribers
             └──▶ Dashboard notification
             └──▶ Risk level updated
```

### Analytics Flow
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Analytics Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Submissions ──▶ [Aggregation] ──▶ [ML Models] ──▶ [Dashboards]            │
│                                                                              │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐         │
│  │ Water Level│──▶│ Rate of    │──▶│RandomForest│──▶│ Flood Risk │         │
│  │ Readings   │   │ Rise Calc  │   │ Prediction │   │ Dashboard  │         │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘         │
│                                                                              │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐         │
│  │ Photo      │──▶│ OpenCV +   │──▶│ Tamper     │──▶│ Security   │         │
│  │ Submissions│   │ Gemini AI  │   │ Detection  │   │ Dashboard  │         │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘         │
│                                                                              │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐         │
│  │ Site       │──▶│ Manning's  │──▶│ Flood      │──▶│ Synthesis  │         │
│  │ Data       │   │ Equation   │   │ Polygons   │   │ Map View   │         │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### User Journey by Role

#### 👷 Field Agent Flow
```
Login → Select Site → Verify Location (GPS) → Capture Photo → 
AI Reads Gauge → Confirm Level → Submit → View My Submissions
```

#### 👨‍💼 Supervisor Flow
```
Login → Cloud Dashboard → View Team Agents → Manage Sites → 
Review Submissions → Assign Sites to Analysts → Export Reports
```

#### 📊 Central Analyst Flow
```
Login → View Assigned Sites Only → Analyze Trends → 
View Flood Risk → Access River Memory AI → Generate Reports
```

#### 🔧 Admin Flow
```
Login → Full Dashboard Access → User Management → 
Site Configuration → Tamper Detection Review → System Settings
```

---

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| **Admin** | `admin` | `admin123` | Full system access, user management, all dashboards |
| **Supervisor** | `supervisor_ganga` | `supervisor123` | Team oversight, site management, analytics |
| **Supervisor** | `supervisor_musi` | `supervisor123` | Team oversight, site management, analytics |
| **Central Analyst** | `analyst_north` | `analyst123` | View-only access to **assigned sites only** |
| **Field Agent** | `field_agent` | `password123` | Capture submissions for assigned sites |

---

## 🌟 Key Features

### 1. 📱 Offline-First Progressive Web App (PWA)
- **Zero-Connectivity Capture**: Field agents can capture water levels and photos even without internet access
- **IndexedDB Storage**: Local data persistence with automatic sync when online
- **Geofenced Verification**: GPS-enforced submissions within ±50m of assigned site
- **Cross-Platform**: Installable on Android, iOS, and Desktop

### 2. 🔐 Role-Based Access Control (RBAC)
- **Admin**: Full system access, user management, global dashboard
- **Supervisor**: Team management, site assignment, regional analytics
- **Central Analyst**: View-only access **restricted to assigned sites**
- **Field Agent**: Data capture for assigned monitoring sites only

### 3. 🧠 AI/ML Capabilities

#### Flood Risk Prediction
- **Algorithm**: RandomForest Classifier (100 estimators)
- **Features**: water_level, rate_of_rise, rainfall_3h, slope_1h, month, river_type
- **Output**: Risk Level (Safe, Caution, Flood Risk, Flash Flood) + Confidence Score

#### Gemini Vision Integration
- **Water Gauge Reading**: Automatic OCR of staff gauges
- **Scene Validation**: Detects phone displays vs real gauges
- **Model**: `gemini-2.5-flash`

#### River Memory AI (Digital Twin)
- **Color Analysis**: HSV classification (Clear, Silty, Muddy, Algae)
- **Flow Estimation**: Optical flow velocity analysis
- **Erosion Tracking**: SSIM comparison for bank erosion detection

#### Tamper Detection
- **Photo Validation**: Detects obstructions, blur, fake inputs
- **Confidence Scoring**: Automated quality assessment
- **Review Workflow**: Admin approval for suspicious submissions

### 4. 🌊 Flood Synthesis Engine
- **Manning's Equation**: Physics-based velocity calculation
- **Rate of Rise**: Real-time trend analysis from submissions
- **GeoJSON Polygons**: Predicted flood extent visualization
- **Severity Alerts**: Warning/Danger threshold monitoring

### 5. 📊 Analytics Dashboards
- **Cloud Dashboard**: Global view for admins
- **Flood Risk Dashboard**: Real-time risk assessment
- **My Submissions**: Personal submission history
- **Tamper Detection**: Security monitoring

### 6. 💬 Communication
- **Crisis Assistant Chatbot**: AI-powered flood safety guidance
- **WhatsApp Integration**: Flood alerts via WhatsApp bot
- **Twilio Voice**: Voice-based submissions

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.13 | Core runtime |
| Flask | 3.0+ | Web framework |
| SQLAlchemy | 2.0 | ORM |
| SQLite | 3 | Database |
| Google Gemini | 2.5-flash | Vision AI |

### Frontend
| Technology | Purpose |
|------------|---------|
| Bootstrap 5 | UI framework |
| Leaflet.js | Interactive maps |
| Chart.js | Data visualization |
| Service Workers | PWA offline support |
| Vanilla JS | Application logic |

### AI/ML
| Library | Purpose |
|---------|---------|
| scikit-learn | Flood prediction models |
| OpenCV | Image processing |
| Shapely | Geospatial operations |
| NumPy | Numerical computing |
| PIL/Pillow | Image manipulation |

---

## 📁 Project Structure

```
jalscan-sih/
├── app.py                 # Main Flask application
├── models.py              # SQLAlchemy models
├── auth.py                # Authentication routes
├── config.py              # Configuration
├── flood_synthesis/       # Flood prediction engine
│   ├── physics_engine.py  # Manning's equation
│   ├── hydrology.py       # Rate-of-rise utilities
│   └── flood_api.py       # REST endpoints
├── river_ai/              # AI analysis modules
│   ├── water_level_detection.py
│   ├── anomaly_detection.py
│   └── bank_erosion.py
├── ml/                    # Machine learning
│   ├── model_train.py
│   └── model_inference.py
├── templates/             # HTML templates
├── static/                # CSS, JS, assets
└── instance/              # Database
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Google Gemini API Key
- ngrok (for mobile testing)

### Installation

```bash
# Clone the repository
git clone https://github.com/vishnu601/jalscan-sih.git
cd jalscan-sih

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python init_db.py

# Run the application
python app.py
```

### Access Points
- **Local**: http://localhost:80
- **Mobile**: Use ngrok: `ngrok http 80`

---

## 🔌 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/login` | User login |
| GET | `/logout` | User logout |

### Core APIs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/submit-reading` | Submit water level reading |
| POST | `/api/analyze-gauge` | AI gauge analysis |
| GET | `/api/flood-risk/all-sites` | Get all sites with risk |
| POST | `/api/flood/predict` | Generate flood prediction |
| POST | `/api/flood/predict-from-site` | Predict from site data |

### Admin APIs
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cloud-dashboard/stats` | Dashboard statistics |
| GET | `/api/tamper-detection/overview` | Tamper detection stats |
| POST | `/api/tamper-detection/run-batch-analysis` | Run batch analysis |

---

## 📱 PWA Features

- **Installable**: Add to home screen on mobile
- **Offline Mode**: Full functionality without internet
- **Auto Sync**: Background sync when online
- **Push Ready**: Notification support (with VAPID keys)

---

## 🔒 Security Features

- Password hashing (Werkzeug)
- Session-based authentication
- CSRF protection
- Role-based route guards
- Geofence verification
- Tamper detection AI

---

## 📊 Monitoring Sites

| River | Location | Site Code |
|-------|----------|-----------|
| Musi River | Hyderabad | MUSI_HYDERABAD_001 |
| Krishna River | Kanchipuram | KRISHNA_RIVER_003 |
| Ganga River | Haridwar | ganga_haridwar |
| Yamuna River | Delhi | yamuna_delhi |
| Godavari River | Nashik | godavari_nashik |
| Brahmaputra | Guwahati | brahmaputra_guwahati |

---

## 👨‍💻 Team

**Developed for Smart India Hackathon 2024**

---

## 📄 License

This project is developed for educational and demonstration purposes for the Smart India Hackathon 2025.

---

*Version 3.5 | Last updated: December 2024*
