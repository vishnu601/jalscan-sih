"""
JalScan Flood Prediction - Model Evaluation
Utilities for evaluating model performance
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import json

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        roc_auc_score, precision_recall_curve,
        f1_score, accuracy_score
    )
except ImportError:
    pass

from .schemas import LABEL_TO_RISK_CATEGORY

logger = logging.getLogger(__name__)

ML_DIR = Path(__file__).parent
REPORTS_DIR = ML_DIR / "reports"


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict:
    """
    Evaluate model predictions and return metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Per-class report
    report = classification_report(
        y_true, y_pred,
        target_names=[LABEL_TO_RISK_CATEGORY[i].value for i in range(4)],
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "per_class": report,
        "confusion_matrix": conf_matrix.tolist(),
        "n_samples": len(y_true),
        "evaluated_at": datetime.utcnow().isoformat()
    }
    
    # ROC AUC if probabilities available
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            # One-vs-rest ROC AUC
            roc_auc = roc_auc_score(
                y_true, y_proba, 
                multi_class='ovr',
                average='weighted'
            )
            metrics["roc_auc_weighted"] = float(roc_auc)
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
    
    return metrics


def generate_evaluation_report(
    metrics: Dict,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a human-readable evaluation report.
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_path: Path to save report (optional)
        
    Returns:
        Report string
    """
    lines = [
        "=" * 60,
        "JalScan Flood Prediction - Model Evaluation Report",
        "=" * 60,
        f"Evaluated at: {metrics.get('evaluated_at', 'N/A')}",
        f"Number of samples: {metrics.get('n_samples', 'N/A')}",
        "",
        "Overall Metrics:",
        f"  Accuracy:       {metrics.get('accuracy', 0):.4f}",
        f"  F1 (weighted):  {metrics.get('f1_weighted', 0):.4f}",
        f"  F1 (macro):     {metrics.get('f1_macro', 0):.4f}",
    ]
    
    if "roc_auc_weighted" in metrics:
        lines.append(f"  ROC AUC:        {metrics['roc_auc_weighted']:.4f}")
    
    lines.extend([
        "",
        "Per-Class Performance:",
        "-" * 40
    ])
    
    per_class = metrics.get("per_class", {})
    for category in ["SAFE", "CAUTION", "FLOOD_RISK", "FLASH_FLOOD_RISK"]:
        if category in per_class:
            cls = per_class[category]
            lines.append(
                f"  {category:20s} P={cls.get('precision', 0):.2f} "
                f"R={cls.get('recall', 0):.2f} F1={cls.get('f1-score', 0):.2f} "
                f"N={cls.get('support', 0)}"
            )
    
    lines.extend([
        "",
        "Confusion Matrix:",
        "-" * 40
    ])
    
    conf_matrix = metrics.get("confusion_matrix", [])
    labels = ["SAFE", "CAUT", "FLOOD", "FLASH"]
    if conf_matrix:
        lines.append("         " + " ".join(f"{l:>6s}" for l in labels))
        for i, row in enumerate(conf_matrix):
            lines.append(f"  {labels[i]:5s} " + " ".join(f"{v:>6d}" for v in row))
    
    lines.extend(["", "=" * 60])
    
    report = "\n".join(lines)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {output_path}")
    
    return report


def compute_site_performance(
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> Dict:
    """
    Compute per-site prediction performance.
    
    Args:
        predictions: List of prediction results with site_id, predicted_category
        ground_truth: List of actual outcomes with site_id, actual_category
        
    Returns:
        Dictionary with per-site metrics
    """
    # Group by site
    site_results = {}
    
    for pred, truth in zip(predictions, ground_truth):
        site_id = pred.get("monitoring_site_id")
        if site_id not in site_results:
            site_results[site_id] = {
                "site_name": pred.get("site_name", f"Site {site_id}"),
                "y_true": [],
                "y_pred": []
            }
        
        site_results[site_id]["y_true"].append(truth.get("label", 0))
        site_results[site_id]["y_pred"].append(pred.get("predicted_label", 0))
    
    # Compute metrics per site
    site_metrics = {}
    for site_id, data in site_results.items():
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        
        site_metrics[site_id] = {
            "site_name": data["site_name"],
            "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0,
            "f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            "n_samples": len(y_true)
        }
    
    return site_metrics


def log_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 10
) -> str:
    """
    Log and return top feature importances.
    """
    sorted_idx = np.argsort(importances)[::-1]
    
    lines = [
        "",
        "Top Feature Importances:",
        "-" * 40
    ]
    
    for i in range(min(top_n, len(feature_names))):
        idx = sorted_idx[i]
        lines.append(f"  {feature_names[idx]:25s} {importances[idx]:.4f}")
    
    report = "\n".join(lines)
    logger.info(report)
    return report
