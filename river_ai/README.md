# River Memory AI

**India's First AI-Powered Digital Twin for River Monitoring**

A comprehensive computer vision pipeline for analyzing river images, detecting water levels, estimating flow speed, assessing gauge health, tracking erosion, and detecting anomalies.

---

## 📁 Directory Structure

```
river_ai/
├── __init__.py              # Package initialization
├── gauge_detection.py       # Water level detection from gauge
├── color_analysis.py        # Water color/sediment analysis
├── flow_estimation.py       # Flow speed estimation
├── gauge_health.py          # Gauge condition assessment
├── bank_erosion.py          # Riverbank erosion tracking
├── anomaly_detection.py     # Unusual behavior detection
├── pipeline.py              # Main orchestration pipeline
├── api_routes.py            # Flask REST API endpoints
├── generate_mock_data.py    # Demo data generator
├── train_color_classifier.py    # ML training for color
├── train_flow_classifier.py     # ML training for flow
└── models/                  # Trained ML models
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install opencv-python numpy scikit-learn joblib
```

### 2. Generate Mock Data (for Demo)

```bash
python -m river_ai.generate_mock_data
```

### 3. (Optional) Train ML Models

```bash
# Train color classifier
python -m river_ai.train_color_classifier --synthetic --samples 1000

# Train flow classifier
python -m river_ai.train_flow_classifier --synthetic --samples 800
```

---

## 🔌 API Endpoints

### POST `/api/v1/river-snapshot`
Process a river image through the full AI pipeline.

**Request (multipart/form-data):**
```
image: [file]
site_id: 1
captured_at: 2024-01-15T10:30:00Z  (optional)
manual_water_level_cm: 250         (optional)
```

**Response:**
```json
{
  "success": true,
  "site_id": "1",
  "water_level_cm": 245.3,
  "flow_class": "moderate",
  "color_class": "silt",
  "turbulence_score": 35,
  "gauge_visibility_score": 85,
  "anomaly_detected": false,
  "overall_risk": "low"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/api/v1/river-snapshot \
  -F "image=@river_photo.jpg" \
  -F "site_id=1"
```

---

### GET `/api/v1/sites/{site_id}/timeline`
Get historical analysis data for a site.

**Query Parameters:**
- `from`: Start date (ISO format)
- `to`: End date (ISO format)  
- `limit`: Max results (default: 100)

**Response:**
```json
{
  "success": true,
  "site_id": 1,
  "site_name": "Ganga River - Varanasi",
  "count": 50,
  "timeline": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "water_level_cm": 245,
      "flow_class": "moderate",
      "color_class": "silt",
      "anomaly_detected": false
    }
  ]
}
```

**cURL Example:**
```bash
curl "http://localhost:5000/api/v1/sites/1/timeline?from=2024-01-01&limit=50"
```

---

### GET `/api/v1/sites/{site_id}/summary`
Get current status and risk assessment for a site.

**Response:**
```json
{
  "success": true,
  "site_id": 1,
  "site_name": "Ganga River - Varanasi",
  "latest": {
    "water_level_cm": 245,
    "flow_class": "moderate",
    "overall_risk": "low"
  },
  "changes": {
    "delta_3h_cm": 12.5,
    "delta_24h_cm": -8.0
  },
  "risk_score": 25,
  "risk_level": "low",
  "active_alerts": []
}
```

---

## 🧠 CV Analysis Modules

### 1. Gauge Detection (`gauge_detection.py`)
- Edge-based waterline detection using Canny + Hough
- Color-based fallback for difficult images
- Pixel-to-cm conversion with site calibration

### 2. Color Analysis (`color_analysis.py`)
- HSV color space analysis
- Sediment classification: clear, silt, muddy, green, dark, polluted
- Color index (0-1) for turbidity measurement
- ML classifier support (optional)

### 3. Flow Estimation (`flow_estimation.py`)
- Optical flow for multi-frame analysis
- Texture analysis for single-frame estimation
- Classes: still, low, moderate, high, turbulent
- Flash flood risk indicator

### 4. Gauge Health (`gauge_health.py`)
- Algae detection (green mask analysis)
- Fading detection (contrast measurement)
- Damage detection (contour analysis)
- Tilt detection (Hough line transform)
- Visibility score (0-100)

### 5. Bank Erosion (`bank_erosion.py`)
- SSIM comparison with baseline images
- Boundary shift analysis
- Status: stable, minor_erosion, heavy_erosion

### 6. Anomaly Detection (`anomaly_detection.py`)
- Rule-based detection with configurable thresholds
- Level change detection (1h, 3h, 24h)
- Color/flow spike detection
- Combined flash flood indicators

---

## 🗄️ Database Schema

### RiverAnalysis Model
```python
class RiverAnalysis(db.Model):
    id: Integer (PK)
    submission_id: FK -> water_level_submissions
    site_id: FK -> monitoring_sites
    timestamp: DateTime
    
    # Water Level
    water_level_cm: Float
    water_level_confidence: Float
    
    # Color
    water_color_rgb: String (JSON)
    sediment_type: String
    pollution_index: Float
    
    # Flow
    flow_speed_class: String
    turbulence_score: Integer
    
    # Gauge Health
    gauge_visibility_score: Integer
    gauge_damage_detected: Boolean
    damage_type: String
    
    # Anomaly
    anomaly_detected: Boolean
    anomaly_type: String
    anomaly_description: Text
    
    # Erosion
    erosion_detected: Boolean
    erosion_change_pct: Float
    
    # Overall
    overall_risk: String
    ai_analysis_json: Text
```

---

## 🎓 Training ML Models

### Color Classifier
```bash
# With labeled data
python -m river_ai.train_color_classifier --data data/color_labels.csv

# With synthetic data
python -m river_ai.train_color_classifier --synthetic --samples 1000

# Output: river_ai/models/color_classifier.joblib
```

### Flow Classifier
```bash
# With labeled data
python -m river_ai.train_flow_classifier --data data/flow_labels.csv

# With synthetic data
python -m river_ai.train_flow_classifier --synthetic --samples 800

# Output: river_ai/models/flow_classifier.joblib
```

---

## 🎯 WOW Factor

> *"Sir, our app doesn't just read water levels. It remembers the river. 
> It learns how the river behaves, what changes, what's abnormal, 
> and what predicts danger. This is India's first AI-powered 
> digital twin for river monitoring."*

---

## 📊 Demo Features

1. **Time-lapse Timeline**: See water level changes over 30 days
2. **AI Annotations**: Every reading has color, flow, health indicators
3. **Risk Scores**: Real-time flood risk assessment
4. **Anomaly Alerts**: Automatic detection of unusual behavior
5. **Erosion Tracking**: Long-term riverbank monitoring

---

## 🔧 Configuration

Environment variables:
```bash
GOOGLE_API_KEY=...     # For Gemini Vision (optional enhancement)
```

Site calibration (per site):
```python
site_config = {
    "gauge_calibration_pixels_per_cm": 10.0,
    "gauge_zero_pixel_y": 400,
    "gauge_roi": [x, y, width, height],
    "water_roi": [x, y, width, height],
    "bank_roi_polygon": [[x1,y1], [x2,y2], ...]
}
```
