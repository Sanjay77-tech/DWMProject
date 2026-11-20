"""
Evaluation and Performance Benchmarking Module for UPI Scam and Fraud Detection
Calculates comprehensive metrics for imbalanced fraud datasets:
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR, and generates Confusion Matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

def evaluate_classification_model(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes rigorous academic evaluation metrics designed specifically
    for imbalanced fraud detection problems where Accuracy alone is deceptive.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion matrix elements - guaranteed (2, 2) by specifying labels=[0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
        
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    # AUC metrics
    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass
            
    metrics = {
        'model_name': model_name,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4) if roc_auc is not None else 'N/A',
        'pr_auc': round(pr_auc, 4) if pr_auc is not None else 'N/A',
        'false_positive_rate': round(fpr, 4),
        'false_negative_rate': round(fnr, 4),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def evaluate_multiple_models(
    models_dict: Dict[str, Any], 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Evaluates a dictionary of trained models on the same test dataset
    and returns a side-by-side comparison DataFrame.
    """
    rows = []
    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            # Anomaly or SVM models
            preds = model.predict(X_test)
            # Invert -1/1 to 1/0 for anomaly detection
            y_pred = np.where(preds == -1, 1, 0)
            dfunc = model.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(3.0 * dfunc)) # Normalized sigmoid pseudo-probability
        else:
            y_pred = model.predict(X_test)
            y_prob = None
            
        m = evaluate_classification_model(y_test, y_pred, y_prob, model_name=name)
        rows.append({
            'Model': name,
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1-Score': m['f1_score'],
            'ROC-AUC': m['roc_auc'],
            'PR-AUC': m['pr_auc'],
            'FPR': m['false_positive_rate'],
            'FNR': m['false_negative_rate']
        })
        
    return pd.DataFrame(rows).sort_values('F1-Score', ascending=False).reset_index(drop=True)
