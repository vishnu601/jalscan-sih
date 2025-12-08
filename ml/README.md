# JalScan Flood Prediction ML Module

## Overview

This module provides machine learning-based flood risk predictions for monitoring sites.

## Features

- **4 Risk Categories**: SAFE, CAUTION, FLOOD_RISK, FLASH_FLOOD_RISK
- **24 Engineered Features**: Water dynamics, temporal patterns, site attributes
- **XGBoost Classifier**: With class balancing and cross-validation
- **Rule-Based Fallback**: Works even without a trained model
- **JalScan GPT**: Natural language chatbot interface

## Quick Start

### 1. Install Dependencies

```bash
pip install scikit-learn xgboost joblib numpy
```

### 2. Train the Model

```bash
# From project root
python -m ml.model_train --days-back 90

# With custom output path
python -m ml.model_train --days-back 180 --output ml/models/flood_v2.joblib
```

### 3. Test the API

```bash
# Get prediction for a site
curl -X POST http://localhost/api/flood-risk/predict \
  -H "Content-Type: application/json" \
  -d '{"monitoring_site_id": 1}'
```

### 4. WhatsApp Queries

Send to the WhatsApp bot:
- "What is the flood risk at Ganga River?"
- "Is there flash flood risk near Yamuna?"
- "Current water level at [site]"

## API Endpoints

### POST /api/flood-risk/predict

Request:
```json
{
  "monitoring_site_id": 1,
  "timestamp": "2024-12-08T10:00:00Z"  // optional
}
```

Response:
```json
{
  "monitoring_site_id": 1,
  "site_name": "Ganga River",
  "timestamp": "2024-12-08T10:00:00",
  "risk_category": "CAUTION",
  "risk_score": 0.65,
  "confidence": 0.72,
  "horizon_hours": 6,
  "explanations": [
    "Water level is at 85% of danger threshold",
    "Water level rose by 25 cm in the last hour"
  ],
  "key_factors": {
    "water_level_cm": 425.0,
    "pct_of_danger_threshold": 85.0,
    "delta_1h": 25.0
  },
  "recommendations": [
    "Monitor water levels closely",
    "Avoid unnecessary travel near rivers"
  ],
  "model_version": "1.0.0"
}
```

## Module Structure

```
ml/
├── __init__.py
├── schemas.py          # Data classes (SiteFeatures, PredictionRequest/Response)
├── data_pipeline.py    # Feature extraction from DB
├── model_train.py      # XGBoost training script
├── model_inference.py  # Prediction service
├── evaluation.py       # Metrics and reporting
├── models/             # Saved model artifacts
│   └── flood_model.joblib
└── reports/            # Training reports
```

## Feature Set

| Feature | Description |
|---------|-------------|
| water_level_cm | Current water level in centimeters |
| pct_of_danger_threshold | Percentage of danger threshold |
| pct_of_alert_threshold | Percentage of alert threshold |
| hour, day_of_week, month | Temporal features |
| is_monsoon | True if June-September |
| delta_1h/3h/6h/12h/24h | Water level changes |
| slope_1h | Rate of rise (cm/hour) |
| acceleration | Change in slope |
| level_mean/max/min/std_24h | 24-hour aggregates |
| submission_count_24h | Data density |
| site_flood_history_count | Historical flood events |
| river_type_encoded | Major/minor/tributary |
| rainfall_* | Weather features (stubbed) |

## Labeling Strategy

Labels are generated based on future water levels:

- **SAFE (0)**: No significant rise expected
- **CAUTION (1)**: Approaching alert level or >30cm rise
- **FLOOD_RISK (2)**: Exceeds danger threshold
- **FLASH_FLOOD_RISK (3)**: Rapid rise >50cm in 3 hours

## Future Enhancements

1. **Weather Integration**: Connect to IMD/OpenWeatherMap APIs
2. **LSTM Models**: Temporal sequence modeling
3. **Online Learning**: Incremental model updates
4. **Multi-language**: Hindi, Telugu support in chatbot
