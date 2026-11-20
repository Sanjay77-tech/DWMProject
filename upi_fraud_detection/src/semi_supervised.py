"""
Semi-Supervised Learning Module for UPI Scam and Fraud Detection
Implements SelfTrainingClassifier, LabelSpreading, and LabelPropagation using Scikit-learn.
Handles training with small labelled subsets and large unlabelled transaction pools (-1 labels),
and provides inspection of generated pseudo-labels.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading, LabelPropagation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def create_semi_supervised_labels(
    y_true: np.ndarray, 
    labeled_ratio: float = 0.15, 
    random_state: int = 42
) -> np.ndarray:
    """
    Simulates real-world scenario where only a small subset of transactions (e.g. 15%)
    have verified fraud/legitimate investigations, and the remaining 85% are unlabelled (-1).
    Ensures stratified representation of both fraud and normal classes in the labelled subset.
    """
    np.random.seed(random_state)
    y_semi = np.copy(y_true)
    n = len(y_true)
    
    # Identify indices for class 0 and 1
    idx_0 = np.where(y_true == 0)[0]
    idx_1 = np.where(y_true == 1)[0]
    
    if len(idx_0) == 0 or len(idx_1) == 0:
        raise ValueError("Semi-supervised learning requires at least one legitimate (0) and one fraudulent (1) sample in the training set.")
    
    # Choose which ones keep labels safely bounded by actual class counts
    n_keep_0 = max(1, min(len(idx_0), int(len(idx_0) * labeled_ratio)))
    n_keep_1 = max(1, min(len(idx_1), int(len(idx_1) * labeled_ratio)))
    
    keep_0 = np.random.choice(idx_0, size=n_keep_0, replace=False)
    keep_1 = np.random.choice(idx_1, size=n_keep_1, replace=False)
    
    keep_indices = set(keep_0).union(set(keep_1))
    
    for i in range(n):
        if i not in keep_indices:
            y_semi[i] = -1  # Standard Scikit-learn unlabelled indicator
            
    return y_semi

def train_self_training_model(
    X: np.ndarray, 
    y_semi: np.ndarray, 
    base_estimator: Optional[Any] = None,
    threshold: float = 0.75, 
    max_iter: int = 10
) -> Tuple[SelfTrainingClassifier, Dict[str, Any]]:
    """
    Fits SelfTrainingClassifier by iteratively predicting on unlabelled data (-1)
    and adopting high-confidence pseudo-labels into the training set.
    """
    if base_estimator is None:
        base_estimator = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            class_weight='balanced', 
            random_state=42
        )
        
    self_trainer = SelfTrainingClassifier(
        estimator=base_estimator,
        threshold=threshold,
        max_iter=max_iter,
        criterion='threshold',
        verbose=False
    )
    
    self_trainer.fit(X, y_semi)
    
    # Extract pseudo-labeling metrics
    unlabelled_mask = (y_semi == -1)
    total_unlabelled = int(np.sum(unlabelled_mask))
    labeled_mask = (y_semi != -1)
    total_labelled = int(np.sum(labeled_mask))
    
    # SelfTrainingClassifier stores final fitted labels in transduction_
    transduced_labels = self_trainer.transduction_
    pseudo_labels_assigned = int(np.sum(transduced_labels[unlabelled_mask] != -1))
    
    pseudo_fraud_count = int(np.sum((transduced_labels == 1) & unlabelled_mask))
    pseudo_normal_count = int(np.sum((transduced_labels == 0) & unlabelled_mask))
    remaining_unlabelled = total_unlabelled - pseudo_labels_assigned
    
    stats = {
        'model_name': 'SelfTrainingClassifier',
        'initial_labelled': total_labelled,
        'initial_unlabelled': total_unlabelled,
        'pseudo_labels_assigned': pseudo_labels_assigned,
        'pseudo_fraud_count': pseudo_fraud_count,
        'pseudo_normal_count': pseudo_normal_count,
        'remaining_unlabelled': remaining_unlabelled,
        'threshold': threshold,
        'classes_': list(self_trainer.classes_)
    }
    
    return self_trainer, stats

def train_label_spreading_model(
    X: np.ndarray, 
    y_semi: np.ndarray, 
    kernel: str = 'rbf', 
    gamma: float = 20.0, 
    alpha: float = 0.2
) -> Tuple[LabelSpreading, Dict[str, Any]]:
    """
    Fits LabelSpreading, which uses normalized graph Laplacian to diffuse labels
    through the feature manifold with soft clamping (regularization parameter alpha).
    Subsamples if dataset is large to maintain academic interactivity.
    """
    max_samples = 4000
    if len(X) > max_samples:
        indices = np.random.choice(len(X), size=max_samples, replace=False)
        X_sub = X[indices]
        y_sub = y_semi[indices]
    else:
        X_sub = X
        y_sub = y_semi
        
    model = LabelSpreading(
        kernel=kernel,
        gamma=gamma,
        alpha=alpha,
        max_iter=30
    )
    model.fit(X_sub, y_sub)
    
    unlabelled_mask = (y_sub == -1)
    transduced = model.transduction_
    
    stats = {
        'model_name': 'LabelSpreading',
        'samples_used': len(X_sub),
        'pseudo_fraud_count': int(np.sum(transduced[unlabelled_mask] == 1)),
        'pseudo_normal_count': int(np.sum(transduced[unlabelled_mask] == 0)),
        'alpha': alpha,
        'gamma': gamma
    }
    
    return model, stats

def train_label_propagation_model(
    X: np.ndarray, 
    y_semi: np.ndarray, 
    kernel: str = 'rbf', 
    gamma: float = 20.0
) -> Tuple[LabelPropagation, Dict[str, Any]]:
    """
    Fits LabelPropagation with hard clamping on initial labelled points.
    """
    max_samples = 4000
    if len(X) > max_samples:
        indices = np.random.choice(len(X), size=max_samples, replace=False)
        X_sub = X[indices]
        y_sub = y_semi[indices]
    else:
        X_sub = X
        y_sub = y_semi
        
    model = LabelPropagation(
        kernel=kernel,
        gamma=gamma,
        max_iter=30
    )
    model.fit(X_sub, y_sub)
    
    unlabelled_mask = (y_sub == -1)
    transduced = model.transduction_
    
    stats = {
        'model_name': 'LabelPropagation',
        'samples_used': len(X_sub),
        'pseudo_fraud_count': int(np.sum(transduced[unlabelled_mask] == 1)),
        'pseudo_normal_count': int(np.sum(transduced[unlabelled_mask] == 0)),
        'gamma': gamma
    }
    
    return model, stats

def save_semi_supervised_model(model, filepath: str):
    """Saves semi-supervised model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_semi_supervised_model(filepath: str):
    """Loads semi-supervised model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)
