#!/usr/bin/env python3
"""
Train Flow Classifier for River Memory AI
Trains a model to classify water flow speed from texture features
"""

import os
import sys
import argparse
import json
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(data_path: str):
    """
    Load training data from CSV or JSON.
    
    Expected format:
        - CSV: magnitude,std,variance,gabor_energy,flow_class
        - JSON: [{"features": [...], "flow_class": "..."}]
    """
    if data_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(data_path)
        feature_cols = [c for c in df.columns if c != 'flow_class']
        X = df[feature_cols].values
        y = df['flow_class'].values
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        X = np.array([d['features'] for d in data])
        y = np.array([d['flow_class'] for d in data])
    else:
        raise ValueError("Data file must be .csv or .json")
    
    return X, y


def generate_synthetic_data(n_samples: int = 800):
    """Generate synthetic training data when real data is unavailable"""
    logger.info("Generating synthetic training data...")
    
    # Flow class feature distributions
    # Features: [magnitude, std, laplacian_var, gabor_energy]
    class_params = {
        "still": {"mag": (1, 0.5), "std": (0.5, 0.3), "lap": (100, 50), "gab": (10, 5)},
        "low": {"mag": (5, 2), "std": (2, 1), "lap": (300, 100), "gab": (25, 10)},
        "moderate": {"mag": (15, 5), "std": (5, 2), "lap": (600, 150), "gab": (50, 15)},
        "high": {"mag": (30, 8), "std": (10, 3), "lap": (1000, 200), "gab": (80, 20)},
        "turbulent": {"mag": (60, 15), "std": (20, 5), "lap": (1500, 300), "gab": (120, 30)},
    }
    
    X = []
    y = []
    
    samples_per_class = n_samples // len(class_params)
    
    for flow_class, params in class_params.items():
        for _ in range(samples_per_class):
            mag = max(0, np.random.normal(params["mag"][0], params["mag"][1]))
            std = max(0, np.random.normal(params["std"][0], params["std"][1]))
            lap = max(0, np.random.normal(params["lap"][0], params["lap"][1]))
            gab = max(0, np.random.normal(params["gab"][0], params["gab"][1]))
            
            X.append([mag, std, lap, gab])
            y.append(flow_class)
    
    return np.array(X), np.array(y)


def train_model(X, y, model_type: str = "random_forest"):
    """Train the flow classifier"""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=100, random_state=42)
        except ImportError:
            logger.warning("XGBoost not available, using RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # Train
    logger.info(f"Training {model_type} classifier...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = (y_pred == y_test).mean()
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_names = ['magnitude', 'std', 'laplacian_var', 'gabor_energy']
        importances = dict(zip(feature_names, model.feature_importances_))
        logger.info(f"Feature Importances: {importances}")
    
    return model, scaler, le, accuracy


def save_model(model, scaler, label_encoder, output_path: str):
    """Save trained model and preprocessing artifacts"""
    import joblib
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "classes": list(label_encoder.classes_),
        "feature_names": ['magnitude', 'std', 'laplacian_var', 'gabor_energy'],
        "version": "1.0.0"
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info(f"Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train water flow classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to training data (CSV/JSON)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic training data")
    parser.add_argument("--samples", type=int, default=800, help="Number of synthetic samples")
    parser.add_argument("--model", type=str, default="random_forest", 
                       choices=["random_forest", "gradient_boosting", "xgboost"])
    parser.add_argument("--output", type=str, default="river_ai/models/flow_classifier.joblib")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("River Memory AI - Flow Classifier Training")
    print("=" * 60)
    
    # Load or generate data
    if args.data and os.path.exists(args.data):
        logger.info(f"Loading data from {args.data}")
        X, y = load_training_data(args.data)
    else:
        logger.info("Using synthetic training data")
        X, y = generate_synthetic_data(args.samples)
    
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"Classes: {np.unique(y)}")
    
    # Train model
    model, scaler, le, accuracy = train_model(X, y, args.model)
    
    # Save model
    save_model(model, scaler, le, args.output)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
