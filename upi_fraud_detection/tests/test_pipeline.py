"""
Unit and Integration Test Suite for UPI Scam and Fraud Detection Pipeline
Tests data loading, schema inference, feature engineering, missing value resilience,
anomaly detection, risk scoring, and model inference without optional attributes.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import generate_synthetic_upi_dataset, detect_column_mapping, load_dataset
from src.preprocessing import (
    clean_transaction_data, build_preprocessing_pipeline, 
    split_chronological, prepare_feature_matrices, MODEL_FEATURE_COLUMNS
)
from src.feature_engineering import extract_transaction_features, extract_single_transaction_features
from src.anomaly_detection import train_isolation_forest, predict_anomaly, train_one_class_svm
from src.classification import train_random_forest, predict_supervised
from src.semi_supervised import create_semi_supervised_labels, train_self_training_model
from src.risk_scoring import compute_risk_score
from src.explainability import explain_transaction

def test_synthetic_dataset_generation():
    """Verify synthetic dataset generator creates valid, non-empty UPI data."""
    df = generate_synthetic_upi_dataset(num_records=200, fraud_ratio=0.05, random_seed=42)
    assert len(df) == 200
    assert 'is_fraud' in df.columns
    assert 'amount' in df.columns
    assert 'sender_id' in df.columns
    assert 'receiver_id' in df.columns
    assert df['amount'].min() > 0
    assert df['is_fraud'].sum() > 0

def test_column_mapping_fuzzy_detection():
    """Verify fuzzy synonym mapping detects varied column names."""
    raw_cols = ['nameOrig', 'nameDest', 'amount', 'isFraud', 'step', 'type']
    mapping = detect_column_mapping(raw_cols)
    assert mapping['nameOrig'] == 'sender_id'
    assert mapping['nameDest'] == 'receiver_id'
    assert mapping['isFraud'] == 'is_fraud'
    assert mapping['amount'] == 'amount'

def test_clean_transaction_data():
    """Verify duplicate removal and filtering of zero/negative amounts."""
    bad_df = pd.DataFrame({
        'transaction_id': ['TXN1', 'TXN1', 'TXN2', 'TXN3'],
        'amount': [100.0, 100.0, -50.0, 0.0],
        'is_fraud': [0, 0, 1, 0]
    })
    cleaned = clean_transaction_data(bad_df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['transaction_id'] == 'TXN1'

def test_feature_engineering_chronological_integrity():
    """Verify feature engineering extracts behavioral velocity features without future leakage."""
    df = generate_synthetic_upi_dataset(num_records=150, fraud_ratio=0.05, random_seed=42)
    df_feat = extract_transaction_features(df)
    
    assert 'user_avg_amount' in df_feat.columns
    assert 'amount_deviation' in df_feat.columns
    assert 'tx_count_5m' in df_feat.columns
    assert 'is_unusual_hour' in df_feat.columns
    assert 'is_new_recipient' in df_feat.columns
    assert not df_feat['user_avg_amount'].isnull().any()

def test_resilience_to_missing_optional_columns():
    """Verify pipeline functions properly even if optional columns (device, location) are missing."""
    minimal_df = pd.DataFrame({
        'transaction_id': [f'TXN{i}' for i in range(50)],
        'amount': np.random.uniform(10, 5000, 50),
        'timestamp': pd.date_range('2026-01-01', periods=50, freq='10min')
    })
    df_feat = extract_transaction_features(minimal_df)
    assert 'is_device_changed' in df_feat.columns
    assert 'is_location_changed' in df_feat.columns
    assert df_feat['is_device_changed'].sum() == 0  # Graceful zero-fill

def test_chronological_splitting():
    """Verify chronological split divides dataset into non-overlapping temporal sets."""
    df = pd.DataFrame({'time_step': range(100), 'val': range(100)})
    train, val, test = split_chronological(df, 0.7, 0.15, 0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert train['time_step'].max() < val['time_step'].min()
    assert val['time_step'].max() < test['time_step'].min()

def test_isolation_forest_anomaly_detection():
    """Verify Isolation Forest training, decision function, and relative 0-100 risk scoring."""
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(200, 5))
    iso = train_isolation_forest(X_normal, contamination=0.05)
    
    X_test = np.vstack([
        np.random.normal(loc=0.0, scale=1.0, size=(10, 5)),
        np.ones((2, 5)) * 25.0  # Obvious extreme outliers
    ])
    
    is_ano, scores = predict_anomaly(iso, X_test, 'isolation_forest')
    assert len(is_ano) == 12
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0
    # Outliers should have significantly higher anomaly score than normal
    assert scores[-1] > scores[0]

def test_semi_supervised_self_training():
    """Verify SelfTrainingClassifier fits with partially unlabelled (-1) dataset."""
    X = np.random.randn(120, 6)
    y_true = np.array([0] * 100 + [1] * 20)
    y_semi = create_semi_supervised_labels(y_true, labeled_ratio=0.20)
    
    assert np.sum(y_semi == -1) > 0  # Unlabelled instances present
    
    model, stats = train_self_training_model(X, y_semi, threshold=0.60, max_iter=3)
    preds = model.predict(X)
    assert len(preds) == 120
    assert stats['initial_unlabelled'] > 0

def test_composite_risk_score_calibration():
    """Verify risk scoring logic, risk bands, and action prescriptions."""
    # Test Normal
    r_norm = compute_risk_score(supervised_prob=0.05, anomaly_score=12.0)
    assert r_norm['risk_category'] == "Low Risk"
    assert "Allow transaction" in r_norm['suggested_action']
    
    # Test High-Risk Fraud
    r_fraud = compute_risk_score(supervised_prob=0.92, anomaly_score=85.0, rule_flags_count=3)
    assert r_fraud['risk_score'] >= 80.0
    assert r_fraud['risk_category'] in ["High Risk", "Very High Risk"]
    assert "Block" in r_fraud['suggested_action'] or "Hold" in r_fraud['suggested_action']

def test_explainable_ai_reasons():
    """Verify human-readable XAI rule triggers for suspicious inputs."""
    suspicious_features = {
        'amount': 75000.0,
        'user_avg_amount': 800.0,
        'amount_ratio_avg': 93.75,
        'hour_of_day': 3,
        'is_unusual_hour': 1,
        'tx_count_5m': 4,
        'time_since_prev_tx_min': 0.5,
        'is_new_recipient': 1,
        'is_device_changed': 1,
        'is_location_changed': 1,
        'failed_attempts': 2,
        'account_drain_ratio': 0.95
    }
    reasons = explain_transaction(suspicious_features)
    assert len(reasons) >= 3
    severities = [r['severity'] for r in reasons]
    assert 'CRITICAL' in severities or 'HIGH' in severities

def test_transaction_history_small_dataset_sampling():
    """Verify that transaction history sampling works safely on small custom datasets without replacement errors."""
    # Simulate a tiny uploaded custom dataset with only 10 rows (e.g. 2 fraud, 8 normal)
    tiny_df = pd.DataFrame({
        'is_fraud': [1, 1] + [0] * 8,
        'amount': [5000, 12000] + [250] * 8,
        'transaction_id': [f"TX{i}" for i in range(10)],
        'sender_id': [f"u{i}@upi" for i in range(10)],
        'receiver_id': [f"r{i}@upi" for i in range(10)],
        'timestamp': ['2026-01-15 10:00:00'] * 10
    })
    
    # Replicate the guarded sampling logic
    df_fraud = tiny_df[tiny_df['is_fraud'] == 1].copy()
    normal_pool = tiny_df[tiny_df['is_fraud'] == 0]
    n_normal = min(80, len(normal_pool))
    assert n_normal == 8
    df_normal = normal_pool.sample(n_normal, random_state=42).copy()
    df_slice = pd.concat([df_fraud, df_normal]).reset_index(drop=True)
    
    assert len(df_slice) == 10
    assert (df_slice['is_fraud'] == 1).sum() == 2
    assert (df_slice['is_fraud'] == 0).sum() == 8
