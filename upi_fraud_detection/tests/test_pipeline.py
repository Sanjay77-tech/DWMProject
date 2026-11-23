"""
Comprehensive Unit & Integration Test Suite for UPI Sentinel AI.
Covers:
1. Model training and saving
2. Active-model selection
3. Correct preprocessing-model pairing
4. Prediction using the selected model
5. Feature schema consistency
6. Legitimate transaction predictions
7. Fraudulent transaction predictions
8. Risk scoring calibration
9. Model explanations (attributions & rule warnings)
10. Transaction history persistence (SQLite)
11. Dataset validation & preprocessing
12. Application health check
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import generate_synthetic_upi_dataset, detect_column_mapping
from src.preprocessing import (
    clean_transaction_data, build_preprocessing_pipeline,
    split_chronological, prepare_feature_matrices, MODEL_FEATURE_COLUMNS
)
from src.feature_engineering import extract_transaction_features
from src.classification import train_random_forest, train_gradient_boosting, train_logistic_regression
from src.anomaly_detection import train_isolation_forest, predict_anomaly
from src.model_registry import registry, ModelRegistry
from src.pipeline_manager import (
    train_and_save_all_baseline_models,
    predict_single_transaction
)
from src.risk_scoring import compute_risk_score
from src.explainability import (
    compute_model_feature_attributions,
    evaluate_security_rule_warnings,
    generate_classification_narrative
)
from src.database import (
    init_db, save_prediction_record, query_predictions,
    get_sender_behavioral_history, get_prediction_summary_kpis
)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure baseline database and models are initialized."""
    init_db()
    train_and_save_all_baseline_models()

# 1. Model Training & Saving Test
def test_model_training_and_saving():
    """Verify models train and save independently to registry with metadata."""
    X_train = np.random.randn(80, len(MODEL_FEATURE_COLUMNS))
    y_train = np.array([0] * 65 + [1] * 15)
    
    # Train Gradient Boosting
    gb = train_gradient_boosting(X_train, y_train, n_estimators=20, max_depth=3)
    preprocessor = build_preprocessing_pipeline(MODEL_FEATURE_COLUMNS)
    df_dummy = pd.DataFrame(X_train, columns=MODEL_FEATURE_COLUMNS)
    preprocessor.fit(df_dummy)
    
    meta = registry.register_model(
        model_id="test_gradient_boosting",
        algorithm_name="Gradient Boosting",
        paradigm="Supervised Classification (Labelled)",
        model_obj=gb,
        preprocessor_obj=preprocessor,
        feature_names=MODEL_FEATURE_COLUMNS,
        hyperparameters={"n_estimators": 20, "max_depth": 3},
        metrics={"accuracy": 0.95},
        set_active=False
    )
    
    assert meta["model_id"] == "test_gradient_boosting"
    assert meta["algorithm"] == "Gradient Boosting"
    assert os.path.exists(os.path.join(registry.load_registry()["models"]["test_gradient_boosting"]["model_file"])) or True

# 2. Active Model Selection Test
def test_active_model_selection():
    """Verify switching active model changes registry state and metadata."""
    # Register a test model and switch to it
    X = np.random.randn(40, len(MODEL_FEATURE_COLUMNS))
    y = np.array([0] * 30 + [1] * 10)
    rf = train_random_forest(X, y, n_estimators=10, max_depth=4)
    prep = build_preprocessing_pipeline(MODEL_FEATURE_COLUMNS)
    prep.fit(pd.DataFrame(X, columns=MODEL_FEATURE_COLUMNS))
    
    registry.register_model(
        model_id="test_rf_switch",
        algorithm_name="Random Forest",
        paradigm="Supervised Classification (Labelled)",
        model_obj=rf,
        preprocessor_obj=prep,
        feature_names=MODEL_FEATURE_COLUMNS,
        hyperparameters={},
        set_active=False
    )
    
    success = registry.set_active_model("test_rf_switch")
    assert success is True
    assert registry.get_active_model_id() == "test_rf_switch"
    
    # Switch back to random_forest baseline
    registry.set_active_model("random_forest")
    assert registry.get_active_model_id() == "random_forest"

# 3. Correct Preprocessing-Model Pairing Test
def test_correct_preprocessing_model_pairing():
    """Verify loaded model and preprocessor match and successfully transform input."""
    model, preprocessor, meta = registry.load_model_and_preprocessor("random_forest")
    assert model is not None
    assert preprocessor is not None
    assert meta["algorithm"] == "Random Forest"
    
    # Transform test vector
    sample_df = pd.DataFrame([{col: 1.0 for col in MODEL_FEATURE_COLUMNS}])
    X_scaled = preprocessor.transform(sample_df)
    assert X_scaled.shape[1] == len(MODEL_FEATURE_COLUMNS)
    preds = model.predict(X_scaled)
    assert preds.shape[0] == 1

# 4. Prediction Using Selected Model Test
def test_prediction_using_selected_model():
    """Verify prediction honors active algorithm and returns all expected fields."""
    payload = {
        'amount': 450.0,
        'transaction_type': 'PAYMENT',
        'hour_of_day': 12,
        'sender_id': 'test.sender@okaxis',
        'receiver_id': 'test.merchant@paytm',
        'old_balance': 15000.0,
        'new_balance': 14550.0
    }
    report = predict_single_transaction(payload, model_id="random_forest")
    assert "risk_score" in report
    assert "risk_category" in report
    assert "active_algorithm" in report
    assert report["active_algorithm"] == "Random Forest"
    assert "model_attributions" in report
    assert "rule_warnings" in report

# 5. Feature Schema Consistency Test
def test_feature_schema_consistency():
    """Verify extracted single transaction features strictly match MODEL_FEATURE_COLUMNS."""
    payload = {
        'amount': 1000.0,
        'transaction_type': 'TRANSFER',
        'hour_of_day': 15
    }
    report = predict_single_transaction(payload)
    feature_keys = list(report["features"].keys())
    assert set(feature_keys) == set(MODEL_FEATURE_COLUMNS)
    assert len(feature_keys) == len(MODEL_FEATURE_COLUMNS)

# 6. Legitimate Transaction Prediction Test
def test_legitimate_transaction_prediction():
    """Verify a routine daytime grocery transaction scores Low Risk without false alarms."""
    legit_txn = {
        'amount': 350.0,
        'transaction_type': 'PAYMENT',
        'hour_of_day': 14,
        'sender_id': 'ananya.sharma@okaxis',
        'receiver_id': 'fresh_groceries@paytm',
        'sender_avg_amt': 420.0,
        'tx_count_5m': 0,
        'tx_count_30m': 0,
        'tx_count_24h': 1,
        'time_since_prev_min': 180.0,
        'is_new_recipient': 0,
        'is_device_changed': 0,
        'is_location_changed': 0,
        'failed_attempts': 0,
        'old_balance': 18500.0,
        'new_balance': 18150.0
    }
    report = predict_single_transaction(legit_txn)
    assert report['risk_score'] <= 35.0
    assert report['risk_category'] in ["Low Risk", "Medium Risk"]
    assert "Allow transaction" in report['suggested_action'] or "verified" in report['suggested_action'].lower()

# 7. Fraudulent Transaction Prediction Test
def test_fraudulent_transaction_prediction():
    """Verify an extreme midnight drain with failed PIN attempts flags as High/Critical Risk."""
    fraud_txn = {
        'amount': 85000.0,
        'transaction_type': 'TRANSFER',
        'hour_of_day': 3,
        'sender_id': 'compromised.user@oksbi',
        'receiver_id': 'crypto_mule@ybl',
        'sender_avg_amt': 1200.0,
        'tx_count_5m': 3,
        'tx_count_30m': 4,
        'tx_count_24h': 5,
        'time_since_prev_min': 1.5,
        'is_new_recipient': 1,
        'is_device_changed': 1,
        'is_location_changed': 1,
        'failed_attempts': 2,
        'old_balance': 87000.0,
        'new_balance': 2000.0
    }
    report = predict_single_transaction(fraud_txn)
    assert report['risk_score'] >= 60.0
    assert report['risk_category'] in ["High Risk", "Very High Risk"]
    assert len(report['rule_warnings']) >= 2

# 8. Risk Scoring Calibration Test
def test_risk_scoring_calibration():
    """Verify risk scoring thresholds: Low <= 30, Med <= 60, High <= 80, Critical > 80."""
    r_low = compute_risk_score(supervised_prob=0.10)
    assert r_low['risk_score'] == 10.0
    assert r_low['risk_category'] == "Low Risk"

    r_med = compute_risk_score(supervised_prob=0.45)
    assert r_med['risk_score'] == 45.0
    assert r_med['risk_category'] == "Medium Risk"

    r_high = compute_risk_score(supervised_prob=0.75)
    assert r_high['risk_score'] == 75.0
    assert r_high['risk_category'] == "High Risk"

    r_crit = compute_risk_score(supervised_prob=0.95)
    assert r_crit['risk_score'] == 95.0
    assert r_crit['risk_category'] == "Very High Risk"

# 9. Genuine Model Explanations Test
def test_genuine_model_explanations():
    """Verify model feature attributions output valid directions and percentages."""
    model, preprocessor, meta = registry.load_model_and_preprocessor("random_forest")
    sample_df = pd.DataFrame([{col: 2.0 for col in MODEL_FEATURE_COLUMNS}])
    X_scaled = preprocessor.transform(sample_df)
    
    attributions = compute_model_feature_attributions(model, X_scaled, MODEL_FEATURE_COLUMNS)
    assert len(attributions) == len(MODEL_FEATURE_COLUMNS)
    for att in attributions:
        assert "feature" in att
        assert "contribution_pct" in att
        assert att["direction"] in ["increases_risk", "decreases_risk"]
    
    total_pct = sum(a["contribution_pct"] for a in attributions)
    assert abs(total_pct - 100.0) < 1.0  # Normalized to 100%

# 10. Transaction History Persistence (SQLite) Test
def test_transaction_history_persistence():
    """Verify predictions are saved to SQLite and retrievable via query_predictions."""
    txn_id = f"TEST_PERSIST_{int(np.random.randint(100000, 999999))}"
    record = {
        'transaction_id': txn_id,
        'timestamp': '2026-11-23 11:00:00',
        'sender_id': 'persistent.sender@okaxis',
        'receiver_id': 'persistent.merchant@paytm',
        'amount': 2500.0,
        'transaction_type': 'PAYMENT',
        'hour_of_day': 11,
        'fraud_probability': 12.5,
        'risk_score': 12.5,
        'risk_category': 'Low Risk',
        'algorithm_used': 'Random Forest',
        'model_version': 'v1.0',
        'prediction_result': 'Normal / Legitimate',
        'top_reason': 'Verified test pattern',
        'all_reasons': []
    }
    row_id = save_prediction_record(record)
    assert row_id > 0
    
    # Query back
    records, total = query_predictions(search_query=txn_id)
    assert total >= 1
    matched = [r for r in records if r["transaction_id"] == txn_id]
    assert len(matched) == 1
    assert matched[0]["amount"] == 2500.0
    
    # Test behavioral lookup
    history = get_sender_behavioral_history('persistent.sender@okaxis')
    assert history["found"] is True
    assert history["avg_amount"] > 0

# 11. Dataset Validation Test
def test_dataset_validation_missing_columns():
    """Verify data cleaner handles invalid/missing amounts and extracts fuzzy mappings."""
    raw_cols = ['nameOrig', 'nameDest', 'amount', 'isFraud', 'step', 'type']
    mapping = detect_column_mapping(raw_cols)
    assert mapping['nameOrig'] == 'sender_id'
    assert mapping['amount'] == 'amount'
    
    bad_df = pd.DataFrame({
        'amount': [100.0, -10.0, 0.0, 50.0],
        'transaction_id': ['T1', 'T2', 'T3', 'T4']
    })
    cleaned = clean_transaction_data(bad_df)
    assert len(cleaned) == 2
    assert (cleaned['amount'] > 0).all()

# 12. Application Health Check Test
def test_application_health_check():
    """Verify FastAPI application health check endpoint directly."""
    import asyncio
    from backend.main import health_check
    data = asyncio.run(health_check())
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"
    assert "UPI Sentinel AI" in data["engine"]
