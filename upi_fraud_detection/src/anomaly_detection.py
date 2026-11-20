"""
Anomaly Detection Module (One-Class Classification) for UPI Fraud Detection
Implements Isolation Forest and One-Class SVM trained on normal baseline behaviour.
Converts raw decision function anomaly scores into calibrated 0-100 anomaly scores.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def train_isolation_forest(
    X_train_normal: np.ndarray, 
    contamination: float = 0.05, 
    n_estimators: int = 150, 
    random_state: int = 42
) -> IsolationForest:
    """
    Trains an Isolation Forest on normal baseline transactions.
    Isolates anomalies by randomly selecting a feature and split value.
    """
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples='auto',
        random_state=random_state,
        n_jobs=-1
    )
    iso_forest.fit(X_train_normal)
    return iso_forest

def train_one_class_svm(
    X_train_normal: np.ndarray, 
    nu: float = 0.05, 
    kernel: str = 'rbf', 
    gamma: str = 'scale'
) -> OneClassSVM:
    """
    Trains a One-Class SVM to learn a soft boundary encapsulating normal transaction patterns.
    Transactions falling outside this hypersphere support boundary are flagged as anomalies.
    """
    # Sample up to 10,000 normal points to prevent quadratic SVM kernel memory slowdown
    if len(X_train_normal) > 10000:
        indices = np.random.choice(len(X_train_normal), size=10000, replace=False)
        X_fit = X_train_normal[indices]
    else:
        X_fit = X_train_normal
        
    oc_svm = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma
    )
    oc_svm.fit(X_fit)
    return oc_svm

def predict_anomaly(
    model, 
    X: np.ndarray, 
    model_type: str = 'isolation_forest'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes anomaly classifications and maps raw continuous decision function values 
    to a standardized relative anomaly risk score between 0.0 and 100.0.
    
    Returns:
        is_anomaly: Binary array (1 = Anomaly / Unusual, 0 = Normal)
        anomaly_scores: Continuous relative anomaly score [0.0 - 100.0]
    """
    # Scikit-learn outputs: +1 for inlier (normal), -1 for outlier (anomaly)
    preds = model.predict(X)
    is_anomaly = np.where(preds == -1, 1, 0)
    
    # Raw decision function: lower scores indicate greater abnormality
    # For IsolationForest: decision_function is the average path length deviation (around -0.5 to +0.5)
    # For OneClassSVM: signed distance to the separating hyperplane
    raw_scores = model.decision_function(X)
    
    if model_type == 'isolation_forest':
        # Typical range is [-0.3, 0.2]. Score < 0 indicates anomaly.
        # Invert and scale to [0, 100] via sigmoid/min-max clipping
        # Standard anomaly mapping:
        calibrated = 1.0 / (1.0 + np.exp(8.0 * (raw_scores + 0.05)))
        anomaly_scores = np.clip(calibrated * 100.0, 0.0, 100.0)
    else:
        # One-Class SVM decision function: negative means outlier
        calibrated = 1.0 / (1.0 + np.exp(1.5 * raw_scores))
        anomaly_scores = np.clip(calibrated * 100.0, 0.0, 100.0)
        
    return is_anomaly, np.round(anomaly_scores, 2)

def save_anomaly_model(model, filepath: str):
    """Saves anomaly detection model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_anomaly_model(filepath: str):
    """Loads anomaly detection model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)
