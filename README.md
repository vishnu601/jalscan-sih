# JalScan - AI-Powered Water Level Monitoring System

> **Smart India Hackathon 2024** | Advanced Flood Prediction & River Monitoring Platform

JalScan is a next-generation water monitoring solution designed to provide real-time flood risk assessment, secure data collection, and predictive analytics. It combines offline-first PWA capabilities with cutting-edge AI to empower field agents and decision-makers.

---

## 🌟 Key Features

### 1. 📱 Offline-First Progressive Web App (PWA)
*   **Zero-Connectivity Capture**: Field agents can capture water levels and photos even without internet access. Data is stored locally (IndexedDB) and automatically synced when connectivity returns.
*   **Geofenced Verification**: GPS-enforced submissions ensure data is collected within ±50m of the assigned site.
*   **Cross-Platform**: Installable on Android, iOS, and Desktop with a native-like experience.

### 2. 🔐 Advanced Role-Based Access Control (RBAC)
*   **Field Agents**: Restricted to capturing data for their specific sites. Cannot view global analytics.
*   **Central Analysts**: View-only access strictly limited to **assigned monitoring sites**. Can visualize data and risk models but cannot alter system configuration.
*   **Supervisors/Admins**: Full system oversight, including user management, site assignment, and global "Cloud Dashboard" access.

### 3. 🧠 Artificial Intelligence Suite
*   **Flood Risk Prediction (ML)**: Random Forest models analyze 24+ hydrological features (slope, rate of rise, rainfall) to predict flood risks up to 6 hours in advance.
*   **River Memory (Digital Twin)**: "Digital Twin" technology tracks river morphology changes over time using historical imagery.
*   **Tamper Detection (Computer Vision)**: Automated analysis of submission photos to detect obstructions, blurry images, or fake inputs.
*   **Gemini AI Integration**: Optical Character Recognition (OCR) for reading water gauge levels from photos.

### 4. 📊 Real-Time Analytics Dashboard
*   **Interactive Maps**: Heatmaps showing water level intensity across river basins.
*   **Trend Analysis**: Historical water level charts with predictive trend lines.
*   **Quality Metrics**: Automated scoring of submission quality based on GPS accuracy and image clarity.

---

## 🛠️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        JalScan System                           │
├───────────────────┬───────────────────┬─────────────────────────┤
│   Data Ingestion  │   AI Processing   │      User Interface     │
├───────────────────┼───────────────────┼─────────────────────────┤
│ • GPS Camera      │ • Gemini Vision   │ • PWA Dashboard         │
│ • QR Verification │ • OpenCV Pipeline │ • Mobile Bottom Nav     │
│ • Twilio Voice    │ • RandomForest ML │ • Role-Based Access     │
│ • Public Upload   │ • Rule-Based NLU  │ • WhatsApp Bot          │
└───────────────────┴───────────────────┴─────────────────────────┘
```

### Tech Stack
*   **Backend**: Python 3.13, Flask, SQLAlchemy, SQLite
*   **Frontend**: Bootstrap 5, Vanilla JS, Service Workers (Workbox)
*   **Database**: SQLite (Development) / PostgreSQL (Production ready)
*   **AI/ML**: scikit-learn, OpenCV, Google Gemini API, TensorFlow (Lite)

---

## 🤖 Detailed AI Capabilities

### Flood Risk Prediction
*   **Algorithm**: RandomForest Classifier (100 estimators)
*   **Features**: `water_level`, `rate_of_rise`, `rainfall_3h`, `slope_1h`, `month`, `river_type`
*   **Output**: Risk Level (Safe, Caution, Flood Risk, Flash Flood) + Confidence Score

### River Memory Analysis
*   **Color Analysis**: HSV color space classification (Clear, Silty, Muddy, Algae).
*   **Flow Estimation**: Optical flow and texture analysis to estimate water velocity.
*   **Erosion Tracking**: SSIM (Structural Similarity) comparison to detect bank erosion.

### JalScan GPT
*   **Intent Detection**: Regex and NLU-based intent recognition ("Flash Flood", "Water Level").
*   **Context Aware**: Provides answers based on specific site data and current risk levels.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Google Gemini API Key
*   Twilio Account (Optional for SMS/Voice)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/vishnu601/jalscan-sih.git
    cd jalscan-sih
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file:
    ```env
    FLASK_APP=app.py
    FLASK_ENV=development
    SECRET_KEY=your-secret-key
    GOOGLE_API_KEY=your-gemini-key
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```

5.  **Access the PWA**
    Open `http://localhost:5000` (or your ngrok URL) in a mobile browser.

---

## 👨‍💻 Team

**Developed for Smart India Hackathon 2024**
*   **Vishnu M** - Lead Developer & Architect

---
*Last updated: December 2024*
