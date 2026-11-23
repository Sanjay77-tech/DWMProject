"""
Risk Scoring Engine for UPI Sentinel AI.
Calibrates and synthesizes outputs from active trained models,
distinguishing calibrated supervised fraud probabilities from unsupervised anomaly scores.
Applies transparent, non-arbitrary risk scoring with verifiable component attributions.
"""

from typing import Dict, Any, Optional

# Standard FinTech Risk Thresholds
LOW_RISK_THRESHOLD = 30.0
MEDIUM_RISK_THRESHOLD = 60.0
HIGH_RISK_THRESHOLD = 80.0

def compute_risk_score(
    supervised_prob: Optional[float] = None,
    anomaly_score: Optional[float] = None,
    semi_supervised_prob: Optional[float] = None,
    rule_flags_count: int = 0,
    active_paradigm: str = "Supervised Classification (Labelled)",
    active_algorithm: str = "Random Forest"
) -> Dict[str, Any]:
    """
    Computes a transparent, calibrated fraud risk score (0-100).
    
    Principles:
    1. Distinguishes calibrated fraud probabilities from unsupervised anomaly metrics.
    2. Does not arbitrarily boost or deflate scores.
    3. Primary scoring is driven by the active selected model:
       - If Supervised/Semi-Supervised: Risk is primarily the direct calibrated fraud probability P(Fraud) * 100.
       - If Anomaly Detection: Risk is driven by the calibrated anomaly isolation score.
    4. Configurable thresholds:
       - 0.0 - 30.0: Low Risk (Legitimate)
       - 30.1 - 60.0: Medium Risk (Suspicious)
       - 60.1 - 80.0: High Risk (Potentially Fraudulent)
       - 80.1 - 100.0: Critical Risk (Severe Potential Fraud)
    """
    is_anomaly_mode = "Anomaly Detection" in active_paradigm or active_algorithm in ["Isolation Forest", "One-Class SVM"]
    
    if is_anomaly_mode:
        # Anomaly Detection paradigm: driven by anomaly score
        if anomaly_score is not None:
            raw_score = float(anomaly_score)
        else:
            raw_score = 0.0
        formula_explanation = f"Risk driven by unsupervised {active_algorithm} calibrated anomaly score ({raw_score:.1f}/100)."
        prob_display = None
    else:
        # Supervised / Semi-supervised paradigm
        if supervised_prob is not None:
            prob = float(supervised_prob)
            raw_score = prob * 100.0
            prob_display = round(prob * 100.0, 2)
            formula_explanation = f"Risk driven by {active_algorithm} predicted fraud probability ({prob_display}%)."
        elif semi_supervised_prob is not None:
            prob = float(semi_supervised_prob)
            raw_score = prob * 100.0
            prob_display = round(prob * 100.0, 2)
            formula_explanation = f"Risk driven by {active_algorithm} pseudo-labelled manifold probability ({prob_display}%)."
        elif anomaly_score is not None:
            raw_score = float(anomaly_score)
            prob_display = None
            formula_explanation = f"Fallback to anomaly isolation score ({raw_score:.1f}/100)."
        else:
            # Cold-start fallback based on rule violations count
            raw_score = min(100.0, rule_flags_count * 20.0)
            prob_display = None
            formula_explanation = f"Rule-based baseline estimate ({rule_flags_count} behavioral triggers)."

    # Clamp strictly to [0.0, 100.0]
    final_score = round(max(0.0, min(100.0, float(raw_score))), 1)

    # Determine Risk Category and Action Protocol
    if final_score <= LOW_RISK_THRESHOLD:
        category = "Low Risk"
        status = "Normal / Legitimate"
        color = "#10b981"  # Emerald Green
        badge_style = "success"
        action = "Allow transaction. Standard behavioral pattern verified."
    elif final_score <= MEDIUM_RISK_THRESHOLD:
        category = "Medium Risk"
        status = "Suspicious"
        color = "#f59e0b"  # Amber
        badge_style = "warning"
        action = "Step-up verification: Prompt SMS/App 2-Factor Authentication (OTP confirmation)."
    elif final_score <= HIGH_RISK_THRESHOLD:
        category = "High Risk"
        status = "Potentially Fraudulent"
        color = "#f97316"  # Orange
        badge_style = "error"
        action = "High-Risk Alert: Temporary transaction hold. Require biometric / in-app push verification."
    else:
        category = "Very High Risk"
        status = "Critical Potential Fraud"
        color = "#ef4444"  # Red
        badge_style = "danger"
        action = "Block transaction immediately. Freeze transfer, alert account holder, and flag for audit."

    return {
        'risk_score': final_score,
        'risk_category': category,
        'prediction_status': status,
        'color': color,
        'badge_style': badge_style,
        'suggested_action': action,
        'fraud_probability': prob_display,
        'active_algorithm': active_algorithm,
        'active_paradigm': active_paradigm,
        'formula_explanation': formula_explanation,
        'components': {
            'model_fraud_prob': prob_display,
            'anomaly_score': round(float(anomaly_score), 1) if anomaly_score is not None else None,
            'rule_violations_count': rule_flags_count
        }
    }
