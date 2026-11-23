"""
Explainable AI (XAI) Module for UPI Sentinel AI.
Provides dual-layer forensic explanations:
1. Model-Level Feature Contributions (Genuine local marginal attribution with direction and weight)
2. Rule-Based Forensic Security Warnings (Strictly separated from ML attributions)
3. Synthesis Narrative explaining final risk tier.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

def compute_model_feature_attributions(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    reference_baseline: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Computes genuine instance-level feature attributions for any trained classifier.
    Measures the directional impact Delta = f(x) - f(x with feature j replaced by baseline).
    """
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)

    n_features = X_sample.shape[1]
    
    # Establish a neutral baseline (zeros if scaled by RobustScaler/StandardScaler)
    if reference_baseline is None or len(reference_baseline) != n_features:
        baseline = np.zeros(n_features, dtype=np.float32)
    else:
        baseline = np.asarray(reference_baseline, dtype=np.float32).ravel()

    # Determine base probability function
    def get_prob(x_mat):
        if hasattr(model, 'predict_proba'):
            return float(model.predict_proba(x_mat)[0, 1])
        elif hasattr(model, 'decision_function'):
            dfunc = float(model.decision_function(x_mat)[0])
            return 1.0 / (1.0 + np.exp(-3.0 * dfunc))
        else:
            return float(model.predict(x_mat)[0])

    try:
        current_prob = get_prob(X_sample)
    except Exception:
        return []

    deltas = []
    for j in range(n_features):
        X_counterfactual = X_sample.copy()
        X_counterfactual[0, j] = baseline[j]
        try:
            prob_without_j = get_prob(X_counterfactual)
            # Positive delta means feature pushed probability UP (increases fraud risk)
            delta = current_prob - prob_without_j
        except Exception:
            delta = 0.0
        deltas.append(delta)

    deltas = np.array(deltas, dtype=float)
    abs_deltas = np.abs(deltas)
    sum_abs = np.sum(abs_deltas)

    attributions = []
    for j in range(n_features):
        fname = feature_names[j] if j < len(feature_names) else f"feature_{j}"
        raw_val = float(X_sample[0, j])
        delta_val = float(deltas[j])
        pct_contrib = float((abs_deltas[j] / sum_abs * 100.0) if sum_abs > 1e-6 else (100.0 / n_features))

        direction = "increases_risk" if delta_val >= 0 else "decreases_risk"
        direction_label = "Increases Fraud Risk" if delta_val >= 0 else "Lowers Risk (Legitimate Indicator)"

        attributions.append({
            "feature": fname,
            "raw_scaled_value": round(raw_val, 3),
            "marginal_impact": round(delta_val, 4),
            "contribution_pct": round(pct_contrib, 1),
            "direction": direction,
            "direction_label": direction_label
        })

    # Sort descending by contribution percentage
    attributions.sort(key=lambda x: x["contribution_pct"], reverse=True)
    return attributions

def evaluate_security_rule_warnings(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Audits forensic rule violations and returns structured security warnings.
    Kept strictly separate from machine learning model attributions.
    """
    warnings = []
    
    amount = float(features.get('amount', 0.0))
    user_avg = float(features.get('user_avg_amount', 0.0))
    amount_ratio = float(features.get('amount_ratio_avg', 1.0))
    
    # 1. Amount anomalies
    if user_avg > 0 and amount_ratio >= 3.0:
        warnings.append({
            'rule_id': 'RULE_AMOUNT_SURGE',
            'severity': 'HIGH',
            'factor': 'Unusual Transaction Amount',
            'message': f"Transaction amount (₹{amount:,.2f}) is {amount_ratio:.1f}x higher than user average (₹{user_avg:,.2f})."
        })
    elif user_avg > 0 and amount_ratio >= 1.8:
        warnings.append({
            'rule_id': 'RULE_AMOUNT_ELEVATED',
            'severity': 'MEDIUM',
            'factor': 'Elevated Transaction Amount',
            'message': f"Amount exceeds standard user baseline by ₹{amount - user_avg:,.2f}."
        })
        
    # 2. Velocity anomalies
    tx_5m = int(features.get('tx_count_5m', 0))
    tx_30m = int(features.get('tx_count_30m', 0))
    time_since = float(features.get('time_since_prev_tx_min', 999.0))
    
    if tx_5m >= 3:
        warnings.append({
            'rule_id': 'RULE_VELOCITY_5M_CRITICAL',
            'severity': 'CRITICAL',
            'factor': 'High-Velocity Burst Attack',
            'message': f"{tx_5m} rapid transactions executed within the last 5 minutes. Characteristic of automated bot attacks."
        })
    elif tx_30m >= 5:
        warnings.append({
            'rule_id': 'RULE_VELOCITY_30M_HIGH',
            'severity': 'HIGH',
            'factor': 'Accelerated Transaction Frequency',
            'message': f"{tx_30m} transactions initiated within 30 minutes, exceeding standard user cadence."
        })
    elif time_since < 1.0 and user_avg > 0:
        warnings.append({
            'rule_id': 'RULE_BURST_INTERVAL',
            'severity': 'MEDIUM',
            'factor': 'Rapid Successive Transfer',
            'message': f"Transfer initiated less than 60 seconds ({time_since * 60:.0f}s) after preceding transaction."
        })
        
    # 3. Nighttime hours
    is_unusual_hour = int(features.get('is_unusual_hour', 0))
    hour = int(features.get('hour_of_day', 12))
    if is_unusual_hour == 1:
        warnings.append({
            'rule_id': 'RULE_NOCTURNAL_HOUR',
            'severity': 'HIGH',
            'factor': 'Nocturnal Activity Window',
            'message': f"Transaction executed at {hour:02d}:00 hours (midnight to 05:00 AM window), high risk for unauthorized account access."
        })
        
    # 4. Account Drain
    drain_ratio = float(features.get('account_drain_ratio', 0.0))
    if drain_ratio >= 0.85:
        warnings.append({
            'rule_id': 'RULE_ACCOUNT_DRAIN_CRITICAL',
            'severity': 'CRITICAL',
            'factor': 'Severe Account Liquidation',
            'message': f"Transaction transfers {drain_ratio * 100:.1f}% of total account balance, leaving minimal funds."
        })
    elif drain_ratio >= 0.50:
        warnings.append({
            'rule_id': 'RULE_ACCOUNT_DRAIN_MEDIUM',
            'severity': 'MEDIUM',
            'factor': 'Substantial Balance Depletion',
            'message': f"Transaction transfers over half ({drain_ratio * 100:.1f}%) of sender's remaining balance."
        })
        
    # 5. Failed attempts
    failed = int(features.get('failed_attempts', 0))
    if failed >= 3:
        warnings.append({
            'rule_id': 'RULE_PIN_BRUTE_FORCE',
            'severity': 'CRITICAL',
            'factor': 'Multiple Failed Authentication Attempts',
            'message': f"{failed} consecutive failed PIN/OTP attempts recorded prior to transaction approval."
        })
    elif failed > 0:
        warnings.append({
            'rule_id': 'RULE_PIN_RETRY',
            'severity': 'MEDIUM',
            'factor': 'Failed Authorization Retry',
            'message': f"{failed} failed PIN authorization attempts logged."
        })
        
    # 6. Device and Location changes
    dev_chg = int(features.get('is_device_changed', 0))
    loc_chg = int(features.get('is_location_changed', 0))
    new_rcv = int(features.get('is_new_recipient', 0))
    
    if dev_chg == 1 and loc_chg == 1:
        warnings.append({
            'rule_id': 'RULE_DEVICE_LOC_ANOMALY',
            'severity': 'HIGH',
            'factor': 'Unrecognized Device & Geographical Location',
            'message': "Transaction initiated from both an unverified device hardware fingerprint and an unexpected location."
        })
    elif dev_chg == 1:
        warnings.append({
            'rule_id': 'RULE_DEVICE_CHANGE',
            'severity': 'MEDIUM',
            'factor': 'Unfamiliar Device Fingerprint',
            'message': "Transaction originated from a newly observed hardware ID not previously registered for this account."
        })
    elif loc_chg == 1:
        warnings.append({
            'rule_id': 'RULE_LOCATION_CHANGE',
            'severity': 'LOW',
            'factor': 'Geographical Discrepancy',
            'message': "Transaction IP/location differs from user's primary operating city."
        })
        
    if new_rcv == 1 and (amount >= 10000 or drain_ratio >= 0.5):
        warnings.append({
            'rule_id': 'RULE_HIGH_VALUE_NEW_PAYEE',
            'severity': 'HIGH',
            'factor': 'High-Value Transfer to First-Time Payee',
            'message': "Substantial amount transferred to a beneficiary never previously encountered in user's history."
        })
        
    return warnings

def generate_classification_narrative(
    risk_category: str,
    algorithm_name: str,
    top_attributions: List[Dict[str, Any]],
    rule_warnings: List[Dict[str, Any]]
) -> str:
    """Produces a plain-language executive summary of the prediction reasoning."""
    if risk_category in ["Low Risk"]:
        return (
            f"The transaction was cleared as Legitimate by {algorithm_name}. "
            "Behavioral amounts, interval velocity, and payee familiarity conform to historical baseline spending patterns."
        )
    elif risk_category in ["Medium Risk"]:
        factors = [a['feature'].replace('_', ' ') for a in top_attributions[:2] if a['direction'] == 'increases_risk']
        factor_str = ", ".join(factors) if factors else "moderate amount deviation"
        return (
            f"Flagged as Suspicious by {algorithm_name}. "
            f"Elevated risk driven primarily by {factor_str}. "
            f"{len(rule_warnings)} security warning(s) detected. Step-up OTP verification recommended."
        )
    else:
        top_risks = [w['factor'] for w in rule_warnings if w['severity'] in ['HIGH', 'CRITICAL']]
        risk_str = "; ".join(top_risks[:3]) if top_risks else "severe non-linear behavioral anomaly"
        return (
            f"ALERT: Classified as {risk_category} by {algorithm_name}. "
            f"Critical indicators include: {risk_str}. Immediate transaction hold and account alert advised."
        )
