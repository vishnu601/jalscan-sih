"""
JalScan Flood Prediction - Model Training
Train and save classifier for flood risk prediction
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import joblib

# ML imports
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
except ImportError:
    print("Missing ML dependencies. Install with: pip install scikit-learn joblib")
    sys.exit(1)

from .schemas import SiteFeatures, LABEL_TO_RISK_CATEGORY
from .data_pipeline import FloodDataPipeline

logger = logging.getLogger(__name__)

# Paths
ML_DIR = Path(__file__).parent
MODELS_DIR = ML_DIR / "models"
REPORTS_DIR = ML_DIR / "reports"


class FloodModelTrainer:
    """
    Train RandomForest classifier for flood risk prediction.
    Handles class imbalance, feature scaling, and model persistence.
    """
    
    def __init__(self, app=None):
        self.app = app
        self.pipeline = FloodDataPipeline(app)
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = SiteFeatures.feature_names()
        
    def train(
        self,
        days_back: int = 90,
        test_size: float = 0.2,
        n_folds: int = 5,
        output_path: str = None
    ) -> dict:
        """
        Full training pipeline.
        
        Args:
            days_back: Days of historical data to use
            test_size: Fraction for holdout test set
            n_folds: Number of cross-validation folds
            output_path: Path to save model (default: ml/models/flood_model.joblib)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("=" * 60)
        logger.info("JalScan Flood Prediction - Model Training")
        logger.info("=" * 60)
        
        # Generate training data
        X, y, site_ids = self.pipeline.generate_training_data(days_back=days_back)
        
        if len(X) < 50:
            logger.warning(f"Insufficient training data: {len(X)} samples. Need at least 50.")
            # Generate synthetic baseline data for demo
            X, y = self._generate_synthetic_data()
            site_ids = [1] * len(X)
        
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights for imbalance
        class_counts = np.bincount(y)
        class_weights = {i: len(y) / (len(class_counts) * count) 
                        for i, count in enumerate(class_counts) if count > 0}
        sample_weights = np.array([class_weights.get(label, 1.0) for label in y_train])
        
        # Train RandomForest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training RandomForest classifier...")
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)
        
        # Metrics
        report = classification_report(
            y_test, y_pred,
            target_names=[LABEL_TO_RISK_CATEGORY[i].value for i in range(4)],
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=min(n_folds, len(np.unique(y))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
        
        metrics = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": report['accuracy'],
            "weighted_f1": report['weighted avg']['f1-score'],
            "cv_f1_mean": float(np.mean(cv_scores)),
            "cv_f1_std": float(np.std(cv_scores)),
            "per_class": {
                LABEL_TO_RISK_CATEGORY[i].value: {
                    "precision": report.get(LABEL_TO_RISK_CATEGORY[i].value, {}).get('precision', 0),
                    "recall": report.get(LABEL_TO_RISK_CATEGORY[i].value, {}).get('recall', 0),
                    "f1": report.get(LABEL_TO_RISK_CATEGORY[i].value, {}).get('f1-score', 0),
                    "support": report.get(LABEL_TO_RISK_CATEGORY[i].value, {}).get('support', 0)
                }
                for i in range(4)
            },
            "confusion_matrix": conf_matrix.tolist(),
            "feature_importances": dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            )),
            "trained_at": datetime.utcnow().isoformat(),
            "model_version": "1.0.0"
        }
        
        # Log results
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        logger.info(f"CV F1: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Save model
        if output_path is None:
            output_path = MODELS_DIR / "flood_model.joblib"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": metrics,
            "version": "1.0.0"
        }
        
        joblib.dump(model_artifact, output_path)
        logger.info(f"Model saved to: {output_path}")
        
        # Save metrics report
        report_path = REPORTS_DIR / f"training_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Report saved to: {report_path}")
        
        return metrics
    
    def _generate_synthetic_data(self, n_samples: int = 500) -> tuple:
        """
        Generate synthetic training data for bootstrapping.
        Used when real data is insufficient.
        """
        logger.info("Generating synthetic training data for bootstrap...")
        
        np.random.seed(42)
        
        X = []
        y = []
        
        for _ in range(n_samples):
            # Random class
            label = np.random.choice([0, 0, 0, 1, 1, 2, 3], p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.07, 0.03])
            
            # Generate features correlated with label
            if label == 0:  # SAFE
                water_level = np.random.uniform(50, 200)
                delta_1h = np.random.uniform(-10, 10)
            elif label == 1:  # CAUTION
                water_level = np.random.uniform(200, 350)
                delta_1h = np.random.uniform(0, 20)
            elif label == 2:  # FLOOD_RISK
                water_level = np.random.uniform(350, 600)
                delta_1h = np.random.uniform(10, 40)
            else:  # FLASH_FLOOD_RISK
                water_level = np.random.uniform(300, 500)
                delta_1h = np.random.uniform(40, 100)
            
            pct_danger = (water_level / 500) * 100
            pct_alert = (water_level / 300) * 100
            
            features = [
                water_level,              # water_level_cm
                pct_danger,               # pct_of_danger_threshold
                pct_alert,                # pct_of_alert_threshold
                np.random.randint(0, 24), # hour
                np.random.randint(0, 7),  # day_of_week
                np.random.randint(1, 13), # month
                float(np.random.choice([0, 1])),  # is_monsoon
                delta_1h,                 # delta_1h
                delta_1h * 2 + np.random.uniform(-5, 5),  # delta_3h
                delta_1h * 3 + np.random.uniform(-10, 10),  # delta_6h
                delta_1h * 4 + np.random.uniform(-15, 15),  # delta_12h
                delta_1h * 5 + np.random.uniform(-20, 20),  # delta_24h
                delta_1h,                 # slope_1h
                np.random.uniform(-5, 5), # acceleration
                water_level + np.random.uniform(-20, 20),  # level_mean_24h
                water_level + np.random.uniform(0, 50),    # level_max_24h
                water_level - np.random.uniform(0, 50),    # level_min_24h
                np.random.uniform(5, 30),  # level_std_24h
                np.random.randint(1, 20),  # submission_count_24h
                np.random.randint(0, 10),  # site_flood_history_count
                np.random.randint(0, 4),   # river_type_encoded
                np.random.uniform(0, 30) if label > 0 else np.random.uniform(0, 10),  # rainfall_last_3h
                np.random.uniform(0, 100) if label > 0 else np.random.uniform(0, 30),  # rainfall_last_24h
                np.random.uniform(0, 50) if label > 1 else np.random.uniform(0, 20),  # forecast_rainfall_6h
            ]
            
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)


def main():
    """CLI entrypoint for training"""
    parser = argparse.ArgumentParser(description="Train JalScan flood prediction model")
    parser.add_argument("--days-back", type=int, default=90, help="Days of historical data")
    parser.add_argument("--output-path", type=str, default=None, help="Model output path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create Flask app context
    try:
        from app import create_app
        app = create_app()
        with app.app_context():
            trainer = FloodModelTrainer(app)
            metrics = trainer.train(
                days_back=args.days_back,
                output_path=args.output_path,
                test_size=args.test_size
            )
            print(f"\n✅ Training complete! Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
