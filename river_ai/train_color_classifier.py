#!/usr/bin/env python3
"""
Train Color Classifier for River Memory AI
Trains a model to classify water color from HSV features
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
        - CSV: h,s,v,variance,color_class
        - JSON: [{"hsv": [h,s,v], "variance": x, "color_class": "..."}]
    """
    if data_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(data_path)
        X = df[['h', 's', 'v', 'variance']].values
        y = df['color_class'].values
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        X = np.array([[d['hsv'][0], d['hsv'][1], d['hsv'][2], d['variance']] for d in data])
        y = np.array([d['color_class'] for d in data])
    else:
        raise ValueError("Data file must be .csv or .json")
    
    return X, y


def generate_synthetic_data(n_samples: int = 1000):
    """Generate synthetic training data when real data is unavailable"""
    logger.info("Generating synthetic training data...")
    
    # Color class HSV centers
    class_params = {
        "clear": {"h": (90, 20), "s": (50, 30), "v": (180, 40), "var": (10, 5)},
        "silt": {"h": (20, 10), "s": (120, 40), "v": (140, 30), "var": (30, 10)},
        "muddy": {"h": (15, 8), "s": (150, 30), "v": (100, 30), "var": (50, 15)},
        "green": {"h": (60, 20), "s": (100, 30), "v": (120, 30), "var": (25, 10)},
        "dark": {"h": (90, 50), "s": (40, 20), "v": (50, 20), "var": (15, 8)},
        "polluted": {"h": (150, 30), "s": (100, 40), "v": (100, 30), "var": (40, 15)},
    }
    
    X = []
    y = []
    
    samples_per_class = n_samples // len(class_params)
    
    for color_class, params in class_params.items():
        for _ in range(samples_per_class):
            h = np.clip(np.random.normal(params["h"][0], params["h"][1]), 0, 180)
            s = np.clip(np.random.normal(params["s"][0], params["s"][1]), 0, 255)
            v = np.clip(np.random.normal(params["v"][0], params["v"][1]), 0, 255)
            var = np.clip(np.random.normal(params["var"][0], params["var"][1]), 0, 100)
            
            X.append([h, s, v, var])
            y.append(color_class)
    
    return np.array(X), np.array(y)


def train_model(X, y, model_type: str = "random_forest"):
    """Train the color classifier"""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    
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
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True, random_state=42)
    
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
    
    return model, scaler, le, accuracy


def save_model(model, scaler, label_encoder, output_path: str):
    """Save trained model and preprocessing artifacts"""
    import joblib
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "classes": list(label_encoder.classes_),
        "version": "1.0.0"
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info(f"Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train water color classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to training data (CSV/JSON)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic training data")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--model", type=str, default="random_forest", 
                       choices=["random_forest", "gradient_boosting", "svm"])
    parser.add_argument("--output", type=str, default="river_ai/models/color_classifier.joblib")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("River Memory AI - Color Classifier Training")
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
