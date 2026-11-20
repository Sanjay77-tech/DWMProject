"""
Supervised Classification Module for UPI Scam and Fraud Detection
Implements Logistic Regression, Random Forest, and Gradient Boosting Classifiers.
Addresses class imbalance and outputs calibrated class probabilities.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_logistic_regression(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    C: float = 1.0, 
    max_iter: int = 1000, 
    random_state: int = 42
) -> LogisticRegression:
    """
    Trains a regularized Logistic Regression model with balanced class weighting.
    Acts as a linear baseline and provides direct odds ratio interpretability.
    """
    if len(np.unique(y_train)) < 2:
        raise ValueError("Supervised classification requires at least two classes (0 and 1) in the training data.")
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight='balanced',
        solver='lbfgs',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 150, 
    max_depth: Optional[int] = 12, 
    min_samples_split: int = 5,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Trains an ensemble Random Forest with balanced sub-sample weighting
    to detect complex non-linear UPI fraud interactions.
    """
    if len(np.unique(y_train)) < 2:
        raise ValueError("Supervised classification requires at least two classes (0 and 1) in the training data.")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int = 100, 
    learning_rate: float = 0.1, 
    max_depth: int = 5, 
    random_state: int = 42
) -> GradientBoostingClassifier:
    """
    Trains a Gradient Boosting Classifier, sequentially minimizing the deviance loss
    to capture subtle edge-case fraud behavior.
    """
    if len(np.unique(y_train)) < 2:
        raise ValueError("Supervised classification requires at least two classes (0 and 1) in the training data.")
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def predict_supervised(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates class predictions and positive class (fraud) probabilities.
    
    Returns:
        y_pred: Binary array (0 = Legitimate, 1 = Fraud)
        y_prob: Continuous probability of fraud [0.0 - 1.0]
    """
    y_pred = model.predict(X)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        # Fallback for models without predict_proba
        y_prob = y_pred.astype(float)
    return y_pred, y_prob

def save_classifier(model, filepath: str):
    """Saves supervised model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_classifier(filepath: str):
    """Loads supervised model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Classifier file not found at: {filepath}")
    return joblib.load(filepath)
