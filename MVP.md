# JalScan - Minimum Viable Product (MVP)

## 🎯 Problem Statement

India faces recurring flood disasters causing loss of life, property damage, and displacement. Current water level monitoring systems suffer from:
- **Manual data collection** prone to errors and delays
- **Lack of real-time monitoring** in remote areas
- **Data tampering** and fraudulent submissions
- **Disconnected notification systems** for flood alerts
- **Limited accessibility** for citizen participation

---

## 💡 Solution: JalScan

**JalScan** is an AI-powered water level monitoring and flood prediction system that enables:
- Field agents to capture GPS-verified water level readings
- AI analysis of water gauge images
- Real-time data sync to central dashboard
- Automated flood alerts via WhatsApp
- Voice-based reporting for accessibility
- Tamper detection to ensure data integrity

---

## 🚀 MVP Features

### 1. Water Level Data Capture
| Feature | Description |
|---------|-------------|
| **GPS-Verified Photo Capture** | Live camera with embedded GPS coordinates |
| **AI Gauge Reading** | Google Gemini Vision extracts water level from images |
| **QR Code Verification** | Site authentication via unique QR codes |
| **Geofencing** | Validates agent is within 500m of site |
| **Offline Support** | Captures work offline, auto-sync when connected |

### 2. Analytics & Visualization
| Feature | Description |
|---------|-------------|
| **Interactive Charts** | Submissions over time, water level trends |
| **Multi-Site Comparison** | Compare water levels across monitoring sites |
| **Quality Metrics** | Track submission quality ratings |
| **Export Reports** | PDF/CSV export functionality |

### 3. Tamper Detection Engine
| Feature | Description |
|---------|-------------|
| **AI-Powered Analysis** | Detects anomalies in submissions |
| **Tamper Score** | 0-1 scale risk assessment |
| **Review Workflow** | Pending → Confirmed/False Positive |
| **Agent Behavior Monitoring** | Track suspicious patterns |

### 4. Multi-Channel Communication
| Channel | Capability |
|---------|------------|
| **WhatsApp Bot** | Flood alerts, water level queries, subscription |
| **Voice Calls** | Twilio-powered voice reporting |
| **External AI Agent** | Replit-hosted voice call data sync |

### 5. User Management & RBAC
| Role | Capabilities |
|------|--------------|
| **Admin** | Full system access, user management |
| **Supervisor** | River assignment, agent management, alerts |
| **Central Analyst** | Analytics, reports, tamper review |
| **Field Agent** | Site capture, submission, sync |

### 6. Public Participation
| Feature | Description |
|---------|-------------|
| **Citizen Image Upload** | Public can submit flood photos |
| **ID Verification** | Government ID validates citizen |
| **Review Queue** | Admins approve/reject submissions |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       FRONTEND (Web App)                     │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐        │
│  │Dashboard│ Capture │Analytics│ Cloud   │ Tamper  │        │
│  │         │   Page  │         │Dashboard│Detection│        │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘        │
└───────┼─────────┼─────────┼─────────┼─────────┼─────────────┘
        │         │         │         │         │
        ▼         ▼         ▼         ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK BACKEND (app.py)                    │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │   Auth   │Submission│ Analytics│  Sync    │  Tamper  │   │
│  │  Routes  │  Routes  │  Routes  │ Service  │  Engine  │   │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘   │
└───────┼──────────┼──────────┼──────────┼──────────┼─────────┘
        │          │          │          │          │
        ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA & SERVICES LAYER                     │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │  SQLite  │  Google  │  Twilio  │  Twilio  │  External│   │
│  │ Database │  Gemini  │ WhatsApp │  Voice   │ AI Agent │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Models

| Model | Purpose |
|-------|---------|
| `User` | Authentication, roles, permissions |
| `MonitoringSite` | River/site coordinates, thresholds |
| `WaterLevelSubmission` | Core data: water level, GPS, photo, timestamp |
| `TamperDetection` | Flagged anomalies with review status |
| `WhatsAppSubscriber` | Flood alert subscriptions |
| `SyncLog` | Cloud synchronization history |
| `AppConfig` | System configuration (CSV URLs, intervals) |

---

## 🔐 Security Features

- **Password Hashing**: Werkzeug secure hashing
- **Session Authentication**: Flask-Login
- **Role-Based Access Control**: 4-tier permission system
- **Geofence Verification**: GPS-based location validation
- **QR Code Authentication**: Site-specific codes
- **Tamper Detection**: AI-powered anomaly detection

---

## 📱 User Flows

### Field Agent Flow
```
Login → Select Site → Verify Location → Capture Photo → 
AI Gauge Reading → Submit → Auto-Sync → Dashboard Update
```

### Supervisor Flow
```
Login → Cloud Dashboard → View Submissions → 
Trigger Manual Alert → Manage Field Agents
```

### Public Citizen Flow
```
Upload Page → Submit Photo + Location → 
ID Verification → Admin Review → Approved/Rejected
```

---

## 🌐 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/submit-reading` | POST | Submit water level data |
| `/api/verify-location` | POST | Validate GPS within geofence |
| `/api/sync/status` | GET | Check sync status |
| `/api/analytics/*` | GET | Dashboard data APIs |
| `/api/tamper-detection/*` | GET/POST | Tamper review APIs |
| `/whatsapp/webhook` | POST | WhatsApp message handler |
| `/voice/webhook` | POST | Twilio voice call handler |

---

## 🚀 Deployment

### Requirements
- Python 3.10+
- SQLite (development) / PostgreSQL (production)
- Google Gemini API key
- Twilio account (WhatsApp + Voice)
- Ngrok (for webhook testing)

### Quick Start
```bash
git clone https://github.com/vishnu601/jalscan-sih.git
cd jalscan-sih
pip install -r requirements.txt
cp .env.example .env  # Configure API keys
python3 app.py
```

### Environment Variables
```env
GOOGLE_API_KEY=your_gemini_key
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
```

---

## 📈 Success Metrics

| Metric | Target |
|--------|--------|
| Submission Latency | < 5 seconds |
| AI Gauge Accuracy | > 90% |
| Sync Success Rate | > 99% |
| Tamper Detection Rate | > 95% |
| Alert Delivery Time | < 30 seconds |

---

## 🔮 Future Enhancements

- **ML Flood Prediction**: Predictive models using historical data
- **Multi-Language Support**: Hindi, Telugu, etc.
- **Mobile Apps**: Native iOS/Android apps
- **Satellite Integration**: Remote sensing data
- **Drone Surveys**: Automated aerial monitoring

---

## 👨‍💻 Team

**Developed for Smart India Hackathon 2024**

---

*Document Version: 1.0 | December 2024*
