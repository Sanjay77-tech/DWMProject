"""
Risk Scoring Engine for UPI Scam and Fraud Detection
Calibrates and synthesizes outputs from Supervised Classifiers, Semi-Supervised Models,
and Anomaly Detectors into an interpretable 0 to 100 Risk Score with risk bands and actions.
"""

from typing import Dict, Any, Optional

def compute_risk_score(
    supervised_prob: Optional[float] = None,
    anomaly_score: Optional[float] = None,
    semi_supervised_prob: Optional[float] = None,
    rule_flags_count: int = 0
) -> Dict[str, Any]:
    """
    Computes a composite fraud risk score (0-100) combining available model signals
    and behavioral rule violations.
    
    Weights:
    - Supervised Probability (50%): High precision on known fraud patterns.
    - Anomaly Detection Score (35%): Identifies novel unlabelled outliers & behavioral deviations.
    - Semi-Supervised Probability / Rules (15%): Pseudo-labelled manifold alignment & heuristics.
    
    Clearly documents the breakdown so anomaly indicators are not conflated with verified fraud.
    """
    scores = []
    weights = []
    
    if supervised_prob is not None:
        scores.append(supervised_prob * 100.0)
        weights.append(0.50)
        
    if anomaly_score is not None:
        # Anomaly score is already in [0, 100]
        scores.append(anomaly_score)
        weights.append(0.35)
        
    if semi_supervised_prob is not None:
        scores.append(semi_supervised_prob * 100.0)
        weights.append(0.15)
        
    if len(scores) == 0:
        # Fallback to rule-based estimate if models not loaded
        raw_score = min(100.0, rule_flags_count * 25.0)
    else:
        norm_weights = [w / sum(weights) for w in weights]
        raw_score = sum(s * w for s, w in zip(scores, norm_weights))
        # Rule boost: If 3 or more severe behavioral rules triggered, provide a safety boost
        if rule_flags_count >= 3:
            raw_score = min(100.0, raw_score + 10.0)
            
    final_score = round(float(raw_score), 1)
    
    # Determine Risk Category and Recommended Action
    if final_score <= 30.0:
        category = "Low Risk"
        status = "Normal / Legitimate"
        color = "#28a745" # Green
        badge_style = "success"
        action = "Allow transaction. No additional verification required."
    elif final_score <= 60.0:
        category = "Medium Risk"
        status = "Suspicious"
        color = "#ffc107" # Amber
        badge_style = "warning"
        action = "Step-up verification: Prompt SMS/App 2-Factor Authentication (OTP confirmation)."
    elif final_score <= 80.0:
        category = "High Risk"
        status = "Potentially Fraudulent"
        color = "#fd7e14" # Orange
        badge_style = "error"
        action = "High-Risk Alert: Temporary 15-minute transaction hold. Require biometric / in-app push verification."
    else:
        category = "Very High Risk"
        status = "Critical Potential Fraud"
        color = "#dc3545" # Red
        badge_style = "danger"
        action = "Block transaction immediately. Freeze UPI transfer, alert account holder and flag for manual risk audit."
        
    return {
        'risk_score': final_score,
        'risk_category': category,
        'prediction_status': status,
        'color': color,
        'badge_style': badge_style,
        'suggested_action': action,
        'components': {
            'supervised_prob': round(supervised_prob * 100, 1) if supervised_prob is not None else None,
            'anomaly_relative_score': round(anomaly_score, 1) if anomaly_score is not None else None,
            'semi_supervised_prob': round(semi_supervised_prob * 100, 1) if semi_supervised_prob is not None else None,
            'rule_violations_count': rule_flags_count
        }
    }
