"""
Pipeline Manager Module for UPI Scam and Fraud Detection
Coordinates data loading, feature caching, model training, model persistence,
and real-time live transaction risk inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.data_loader import load_dataset
from src.preprocessing import (
    clean_transaction_data, build_preprocessing_pipeline,
    split_chronological, prepare_feature_matrices,
    save_preprocessor, load_preprocessor, MODEL_FEATURE_COLUMNS
)
from src.feature_engineering import (
    extract_transaction_features, extract_single_transaction_features
)
from src.anomaly_detection import (
    train_isolation_forest, train_one_class_svm,
    predict_anomaly, save_anomaly_model, load_anomaly_model
)
from src.classification import train_random_forest, predict_supervised, save_classifier, load_classifier
from src.semi_supervised import (
    create_semi_supervised_labels, train_self_training_model,
    save_semi_supervised_model, load_semi_supervised_model
)
from src.risk_scoring import compute_risk_score
from src.explainability import explain_transaction

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

def get_model_paths() -> Dict[str, str]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    return {
        'preprocessor': os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl'),
        'isolation_forest': os.path.join(MODELS_DIR, 'isolation_forest.pkl'),
        'one_class_svm': os.path.join(MODELS_DIR, 'one_class_svm.pkl'),
        'random_forest': os.path.join(MODELS_DIR, 'random_forest.pkl'),
        'self_training': os.path.join(MODELS_DIR, 'self_training.pkl')
    }

def train_and_save_all_baseline_models(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Executes an end-to-end training pipeline on the benchmark dataset:
    - Feature extraction with strict temporal ordering
    - Chronological data split (70% train, 15% val, 15% test)
    - Fitting Scikit-learn RobustScaler Preprocessor
    - Training Isolation Forest on normal baseline transactions
    - Training One-Class SVM
    - Training Random Forest Classifier on labelled transactions
    - Training SelfTrainingClassifier with 15% initial labels
    - Persisting all models to models/
    """
    paths = get_model_paths()
    
    if df is None:
        df, _ = load_dataset()
        
    df_clean = clean_transaction_data(df)
    df_feat = extract_transaction_features(df_clean)
    
    # Save processed feature dataset
    processed_path = os.path.join(DATA_PROCESSED_DIR, 'upi_features_processed.csv')
    df_feat.to_csv(processed_path, index=False)
    
    # Chronological Split
    df_train, df_val, df_test = split_chronological(df_feat, 0.70, 0.15, 0.15)
    
    # Build & Fit Preprocessor on Train set only (Strict anti-leakage)
    preprocessor = build_preprocessing_pipeline(MODEL_FEATURE_COLUMNS)
    X_train, y_train, preprocessor, _ = prepare_feature_matrices(df_train, preprocessor, fit=True)
    X_val, y_val, _, _ = prepare_feature_matrices(df_val, preprocessor, fit=False)
    X_test, y_test, _, _ = prepare_feature_matrices(df_test, preprocessor, fit=False)
    
    save_preprocessor(preprocessor, paths['preprocessor'])
    
    # 1. Anomaly Detectors (Trained on Normal Inlier transactions)
    if y_train is not None:
        normal_mask = (y_train == 0)
        X_train_normal = X_train[normal_mask]
    else:
        X_train_normal = X_train
        
    iso_forest = train_isolation_forest(X_train_normal, contamination=0.045)
    save_anomaly_model(iso_forest, paths['isolation_forest'])
    
    oc_svm = train_one_class_svm(X_train_normal, nu=0.05)
    save_anomaly_model(oc_svm, paths['one_class_svm'])
    
    # 2. Supervised Classifier (Random Forest)
    if y_train is not None:
        rf = train_random_forest(X_train, y_train, n_estimators=120, max_depth=12)
        save_classifier(rf, paths['random_forest'])
    else:
        rf = None
        
    # 3. Semi-Supervised Model (Self-Training with 15% labels)
    if y_train is not None:
        y_semi = create_semi_supervised_labels(y_train, labeled_ratio=0.15)
        st_model, st_stats = train_self_training_model(X_train, y_semi, threshold=0.75)
        save_semi_supervised_model(st_model, paths['self_training'])
    else:
        st_model = None
        st_stats = {}
        
    return {
        'status': 'SUCCESS',
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'models_saved': list(paths.keys())
    }

def ensure_baseline_models_exist():
    """Checks if saved models exist; trains them if missing."""
    paths = get_model_paths()
    if not (os.path.exists(paths['preprocessor']) and os.path.exists(paths['random_forest'])):
        print("Models not found on disk. Initializing baseline training...")
        train_and_save_all_baseline_models()

def predict_single_transaction(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Live real-time inference function:
    Takes transaction parameters from GUI, transforms into feature vector,
    applies fitted pipeline, computes anomaly and fraud risk, and returns XAI explanations.
    """
    ensure_baseline_models_exist()
    paths = get_model_paths()
    
    preprocessor = load_preprocessor(paths['preprocessor'])
    
    # Extract features dictionary
    features = extract_single_transaction_features(
        amount=float(input_dict.get('amount', 500.0)),
        txn_type=str(input_dict.get('transaction_type', 'PAYMENT')),
        hour=int(input_dict.get('hour_of_day', 12)),
        sender_avg_amt=float(input_dict.get('sender_avg_amt', 500.0)),
        tx_count_5m=int(input_dict.get('tx_count_5m', 0)),
        tx_count_30m=int(input_dict.get('tx_count_30m', 0)),
        tx_count_24h=int(input_dict.get('tx_count_24h', 1)),
        time_since_prev_min=float(input_dict.get('time_since_prev_min', 120.0)),
        is_new_recipient=int(input_dict.get('is_new_recipient', 0)),
        is_device_changed=int(input_dict.get('is_device_changed', 0)),
        is_location_changed=int(input_dict.get('is_location_changed', 0)),
        failed_attempts=int(input_dict.get('failed_attempts', 0)),
        old_balance=float(input_dict.get('old_balance', 10000.0)),
        new_balance=float(input_dict.get('new_balance', 9500.0))
    )
    
    df_single = pd.DataFrame([features])
    X_scaled = preprocessor.transform(df_single)
    
    # Model 1: Supervised Random Forest
    rf = load_classifier(paths['random_forest'])
    _, rf_probs = predict_supervised(rf, X_scaled)
    supervised_prob = float(rf_probs[0])
    
    # Model 2: Isolation Forest Anomaly Detector
    iso = load_anomaly_model(paths['isolation_forest'])
    is_anomaly, anomaly_scores = predict_anomaly(iso, X_scaled, 'isolation_forest')
    anomaly_score = float(anomaly_scores[0])
    
    # Model 3: Semi-Supervised Self-Training
    if os.path.exists(paths['self_training']):
        st_model = load_semi_supervised_model(paths['self_training'])
        if hasattr(st_model, 'predict_proba'):
            semi_prob = float(st_model.predict_proba(X_scaled)[0, 1])
        else:
            semi_prob = float(st_model.predict(X_scaled)[0])
    else:
        semi_prob = None
        
    # Count rule flags
    explanations = explain_transaction(features)
    rule_flags = sum(1 for e in explanations if e['severity'] in ['HIGH', 'CRITICAL'])
    
    # Composite Risk Score Calculation
    risk_report = compute_risk_score(
        supervised_prob=supervised_prob,
        anomaly_score=anomaly_score,
        semi_supervised_prob=semi_prob,
        rule_flags_count=rule_flags
    )
    
    risk_report['explanations'] = explanations
    risk_report['is_anomaly'] = int(is_anomaly[0])
    risk_report['features'] = features
    
    return risk_report
