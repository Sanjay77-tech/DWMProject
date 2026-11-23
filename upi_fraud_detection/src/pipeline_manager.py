"""
Pipeline Manager Module for UPI Sentinel AI.
Coordinates model registry, training, active model persistence, paired preprocessing,
historical sender behavioral lookup, live prediction, and explainability.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from src.data_loader import load_dataset
from src.preprocessing import (
    clean_transaction_data, build_preprocessing_pipeline,
    split_chronological, prepare_feature_matrices,
    MODEL_FEATURE_COLUMNS
)
from src.feature_engineering import extract_transaction_features
from src.classification import (
    train_random_forest, train_gradient_boosting, train_logistic_regression,
    predict_supervised
)
from src.anomaly_detection import (
    train_isolation_forest, train_one_class_svm, predict_anomaly
)
from src.semi_supervised import (
    create_semi_supervised_labels, train_self_training_model
)
from src.evaluation import evaluate_classification_model
from src.risk_scoring import compute_risk_score
from src.explainability import (
    compute_model_feature_attributions,
    evaluate_security_rule_warnings,
    generate_classification_narrative
)
from src.model_registry import registry
from src.database import get_sender_behavioral_history, save_prediction_record

def get_model_paths() -> Dict[str, str]:
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return {
        'preprocessor': os.path.join(models_dir, 'preprocessing_pipeline.pkl'),
        'random_forest': os.path.join(models_dir, 'random_forest.pkl'),
        'gradient_boosting': os.path.join(models_dir, 'gradient_boosting.pkl'),
        'logistic_regression': os.path.join(models_dir, 'logistic_regression.pkl'),
        'isolation_forest': os.path.join(models_dir, 'isolation_forest.pkl'),
        'self_training': os.path.join(models_dir, 'self_training.pkl')
    }

def train_and_save_all_baseline_models(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Trains and registers baseline models across all paradigms with matched preprocessors.
    """
    if df is None:
        df, _ = load_dataset()
        
    df_clean = clean_transaction_data(df)
    df_feat = extract_transaction_features(df_clean)
    
    # Chronological Split (70% train, 15% val, 15% test)
    df_train, df_val, df_test = split_chronological(df_feat, 0.70, 0.15, 0.15)
    
    # Preprocessor fit on train split only
    preprocessor = build_preprocessing_pipeline(MODEL_FEATURE_COLUMNS)
    X_train, y_train, preprocessor, fcols = prepare_feature_matrices(df_train, preprocessor, fit=True)
    X_test, y_test, _, _ = prepare_feature_matrices(df_test, preprocessor, fit=False)
    
    saved_models = []
    
    # 1. Random Forest (Supervised)
    if y_train is not None and len(np.unique(y_train)) >= 2:
        rf = train_random_forest(X_train, y_train, n_estimators=120, max_depth=12)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        rf_metrics = evaluate_classification_model(y_test, rf_pred, rf_prob, "Random Forest")
        registry.register_model(
            model_id="random_forest",
            algorithm_name="Random Forest",
            paradigm="Supervised Classification (Labelled)",
            model_obj=rf,
            preprocessor_obj=preprocessor,
            feature_names=fcols,
            hyperparameters={"n_estimators": 120, "max_depth": 12},
            metrics=rf_metrics,
            set_active=True
        )
        saved_models.append("random_forest")

    # 2. Gradient Boosting (Supervised)
    if y_train is not None and len(np.unique(y_train)) >= 2:
        gb = train_gradient_boosting(X_train, y_train, n_estimators=80, learning_rate=0.1, max_depth=4)
        gb_pred = gb.predict(X_test)
        gb_prob = gb.predict_proba(X_test)[:, 1]
        gb_metrics = evaluate_classification_model(y_test, gb_pred, gb_prob, "Gradient Boosting")
        registry.register_model(
            model_id="gradient_boosting",
            algorithm_name="Gradient Boosting",
            paradigm="Supervised Classification (Labelled)",
            model_obj=gb,
            preprocessor_obj=preprocessor,
            feature_names=fcols,
            hyperparameters={"n_estimators": 80, "learning_rate": 0.1, "max_depth": 4},
            metrics=gb_metrics,
            set_active=False
        )
        saved_models.append("gradient_boosting")

    # 3. Logistic Regression (Supervised)
    if y_train is not None and len(np.unique(y_train)) >= 2:
        lr = train_logistic_regression(X_train, y_train, C=1.0)
        lr_pred = lr.predict(X_test)
        lr_prob = lr.predict_proba(X_test)[:, 1]
        lr_metrics = evaluate_classification_model(y_test, lr_pred, lr_prob, "Logistic Regression")
        registry.register_model(
            model_id="logistic_regression",
            algorithm_name="Logistic Regression",
            paradigm="Supervised Classification (Labelled)",
            model_obj=lr,
            preprocessor_obj=preprocessor,
            feature_names=fcols,
            hyperparameters={"C": 1.0, "max_iter": 1000},
            metrics=lr_metrics,
            set_active=False
        )
        saved_models.append("logistic_regression")

    # 4. Isolation Forest (Anomaly Detection)
    normal_mask = (y_train == 0) if y_train is not None else np.ones(len(X_train), dtype=bool)
    iso = train_isolation_forest(X_train[normal_mask], contamination=0.045, n_estimators=100)
    iso_pred, iso_scores = predict_anomaly(iso, X_test, 'isolation_forest')
    iso_metrics = evaluate_classification_model(y_test, iso_pred, iso_scores / 100.0, "Isolation Forest") if y_test is not None else {}
    registry.register_model(
        model_id="isolation_forest",
        algorithm_name="Isolation Forest",
        paradigm="One-Class Anomaly Detection (Trained on Normal Baselines)",
        model_obj=iso,
        preprocessor_obj=preprocessor,
        feature_names=fcols,
        hyperparameters={"contamination": 0.045, "n_estimators": 100},
        metrics=iso_metrics,
        set_active=False
    )
    saved_models.append("isolation_forest")

    # 5. Self-Training Classifier (Semi-Supervised)
    if y_train is not None and len(np.unique(y_train)) >= 2:
        y_semi = create_semi_supervised_labels(y_train, labeled_ratio=0.15)
        st_model, st_stats = train_self_training_model(X_train, y_semi, threshold=0.75, max_iter=6)
        st_pred = st_model.predict(X_test)
        st_prob = st_model.predict_proba(X_test)[:, 1]
        st_metrics = evaluate_classification_model(y_test, st_pred, st_prob, "Self-Training Classifier")
        registry.register_model(
            model_id="self_training",
            algorithm_name="Self-Training Classifier",
            paradigm="Semi-Supervised Learning (Partial Labels + Pseudo-Labelling)",
            model_obj=st_model,
            preprocessor_obj=preprocessor,
            feature_names=fcols,
            hyperparameters={"threshold": 0.75, "labeled_ratio": 0.15},
            metrics=st_metrics,
            set_active=False
        )
        saved_models.append("self_training")

    return {
        "status": "SUCCESS",
        "registered_models": saved_models,
        "active_model": registry.get_active_model_id()
    }

def ensure_baseline_models_exist():
    """Ensures at least one active model and preprocessor are available."""
    try:
        registry.load_model_and_preprocessor()
    except Exception:
        print("Model artifacts not initialized. Training baseline models...")
        train_and_save_all_baseline_models()

def predict_single_transaction(input_dict: Dict[str, Any], model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes live transaction prediction using the chosen (or active) registered model.
    Utilizes sender behavioral history if available, extracts exact feature schema,
    computes genuine model attributions and rule warnings, and returns calibrated risk.
    """
    ensure_baseline_models_exist()

    # 1. Load active model, paired preprocessor, and metadata
    model, preprocessor, meta = registry.load_model_and_preprocessor(model_id)
    algorithm_name = meta.get("algorithm", "Random Forest")
    paradigm = meta.get("paradigm", "Supervised Classification (Labelled)")
    model_version = meta.get("version", "v1.0")

    # 2. Extract inputs with sender historical enrichment
    sender_id = str(input_dict.get('sender_id', 'unknown@upi'))
    receiver_id = str(input_dict.get('receiver_id', 'unknown@upi'))
    amount = float(input_dict.get('amount', 500.0))
    txn_type = str(input_dict.get('transaction_type', 'PAYMENT'))
    hour = int(input_dict.get('hour_of_day', 12))
    old_bal = float(input_dict.get('old_balance', 10000.0))
    new_bal = float(input_dict.get('new_balance', max(0.0, old_bal - amount)))
    failed_attempts = int(input_dict.get('failed_attempts', 0))

    # Look up sender history from database
    history_info = get_sender_behavioral_history(sender_id)
    is_cold_start = not history_info["found"]

    if history_info["found"]:
        derived_avg = float(history_info["avg_amount"])
        is_new_recipient = 1 if receiver_id not in history_info["known_recipients"] else 0
        tx_count_24h = int(input_dict.get('tx_count_24h', max(1, history_info["tx_count"])))
        derived_from_history = True
    else:
        # User entered or cold-start fallback
        derived_avg = float(input_dict.get('sender_avg_amt', amount))
        is_new_recipient = int(input_dict.get('is_new_recipient', 1))
        tx_count_24h = int(input_dict.get('tx_count_24h', 1))
        derived_from_history = False

    tx_5m = int(input_dict.get('tx_count_5m', 0))
    tx_30m = int(input_dict.get('tx_count_30m', 0))
    time_since = float(input_dict.get('time_since_prev_min', 120.0))
    dev_chg = int(input_dict.get('is_device_changed', 0))
    loc_chg = int(input_dict.get('is_location_changed', 0))

    # Calculate exact mathematical derivations
    user_avg = derived_avg if derived_avg > 0 else amount
    amount_dev = amount - user_avg
    amount_ratio = amount / (user_avg + 1.0)
    is_unusual_hour = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
    drain_ratio = (amount / (old_bal + 1.0)) if old_bal > 0 else 0.0
    bal_error = float(old_bal - amount - new_bal)

    feature_dict = {
        'amount': float(amount),
        'user_avg_amount': float(user_avg),
        'amount_deviation': float(amount_dev),
        'amount_ratio_avg': float(amount_ratio),
        'hour_of_day': int(hour),
        'is_unusual_hour': int(is_unusual_hour),
        'tx_count_5m': int(tx_5m),
        'tx_count_30m': int(tx_30m),
        'tx_count_24h': int(tx_count_24h),
        'time_since_prev_tx_min': float(time_since),
        'is_new_recipient': int(is_new_recipient),
        'is_device_changed': int(dev_chg),
        'is_location_changed': int(loc_chg),
        'failed_attempts': int(failed_attempts),
        'balance_error': float(bal_error),
        'account_drain_ratio': float(drain_ratio),
        'is_transfer': 1 if txn_type == 'TRANSFER' else 0,
        'is_cash_out': 1 if txn_type == 'CASH_OUT' else 0
    }

    # Transform through active paired preprocessor
    df_single = pd.DataFrame([feature_dict])
    X_scaled = preprocessor.transform(df_single)

    # 3. Model Inference using Active Selected Model
    supervised_prob = None
    anomaly_score = None

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_scaled)
        supervised_prob = float(probs[0, 1])
    elif hasattr(model, 'decision_function'):
        dfunc = float(model.decision_function(X_scaled)[0])
        # Calibrate decision function into [0, 100] anomaly score
        anomaly_score = float(max(0.0, min(100.0, 100.0 / (1.0 + np.exp(8.0 * (dfunc + 0.05))))))
    else:
        pred_val = model.predict(X_scaled)[0]
        supervised_prob = float(pred_val)

    # Compute rule-based warnings
    rule_warnings = evaluate_security_rule_warnings(feature_dict)
    rule_flags_count = sum(1 for w in rule_warnings if w['severity'] in ['HIGH', 'CRITICAL'])

    # 4. Risk Scoring
    risk_report = compute_risk_score(
        supervised_prob=supervised_prob,
        anomaly_score=anomaly_score,
        rule_flags_count=rule_flags_count,
        active_paradigm=paradigm,
        active_algorithm=algorithm_name
    )

    # 5. Genuine Model-Level Feature Contributions
    fnames = meta.get("feature_names", MODEL_FEATURE_COLUMNS)
    model_attributions = compute_model_feature_attributions(model, X_scaled, fnames)

    # 6. Classification Narrative
    narrative = generate_classification_narrative(
        risk_category=risk_report['risk_category'],
        algorithm_name=algorithm_name,
        top_attributions=model_attributions,
        rule_warnings=rule_warnings
    )

    # Format output for frontend & audit storage
    risk_report['narrative'] = narrative
    risk_report['model_attributions'] = model_attributions[:6]  # Top 6 significant drivers
    risk_report['rule_warnings'] = rule_warnings
    risk_report['active_model_id'] = meta.get("model_id")
    risk_report['active_algorithm'] = algorithm_name
    risk_report['model_version'] = model_version
    risk_report['is_cold_start'] = is_cold_start
    risk_report['derived_from_history'] = derived_from_history
    risk_report['features'] = feature_dict

    return risk_report
