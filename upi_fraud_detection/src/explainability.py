"""
Explainable AI (XAI) Module for UPI Scam and Fraud Detection
Extracts model-level global feature importances (MDI & Permutation)
and provides local rule-based + feature-attribution explanations for flagged transactions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.inspection import permutation_importance

def get_random_forest_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Retrieves Mean Decrease in Impurity (Gini importance) from Random Forest.
    """
    if not hasattr(model, 'feature_importances_'):
        return pd.DataFrame({'feature': feature_names, 'importance': [0.0] * len(feature_names)})
        
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    df_imp['importance_pct'] = (df_imp['importance'] / df_imp['importance'].sum()) * 100
    return df_imp

def compute_permutation_importance(
    model, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes permutation feature importance on validation dataset to prevent MDI bias.
    """
    result = permutation_importance(
        model, X_val, y_val, 
        n_repeats=n_repeats, 
        random_state=random_state, 
        scoring='roc_auc',
        n_jobs=-1
    )
    
    df_perm = pd.DataFrame({
        'feature': feature_names,
        'mean_importance': result.importances_mean,
        'std_importance': result.importances_std
    }).sort_values('mean_importance', ascending=False).reset_index(drop=True)
    
    return df_perm

def explain_transaction(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generates human-understandable, legally sound, and auditable reasons
    explaining why a specific transaction was flagged as suspicious or anomalous.
    """
    reasons = []
    
    # 1. Amount deviation
    amount = float(features.get('amount', 0))
    user_avg = float(features.get('user_avg_amount', 0))
    amount_ratio = float(features.get('amount_ratio_avg', 1.0))
    
    if user_avg > 0 and amount_ratio >= 3.0:
        reasons.append({
            'severity': 'HIGH',
            'factor': 'Unusual Transaction Amount',
            'explanation': f"Transaction amount (₹{amount:,.2f}) is {amount_ratio:.1f}x higher than sender's historical average (₹{user_avg:,.2f}).",
            'contribution': min(100, int(amount_ratio * 15))
        })
    elif user_avg > 0 and amount_ratio >= 1.8:
        reasons.append({
            'severity': 'MEDIUM',
            'factor': 'Elevated Transaction Amount',
            'explanation': f"Amount exceeds normal expenditure baseline by ₹{amount - user_avg:,.2f}.",
            'contribution': 35
        })
        
    # 2. Velocity spikes
    tx_5m = int(features.get('tx_count_5m', 0))
    tx_30m = int(features.get('tx_count_30m', 0))
    time_since = float(features.get('time_since_prev_tx_min', 999))
    
    if tx_5m >= 3:
        reasons.append({
            'severity': 'CRITICAL',
            'factor': 'Extreme High-Velocity Burst',
            'explanation': f"{tx_5m} transactions initiated in the last 5 minutes. Characteristic of automated bot attacks or frantic unauthorized draining.",
            'contribution': 85
        })
    elif tx_30m >= 5:
        reasons.append({
            'severity': 'HIGH',
            'factor': 'Unusual Transaction Frequency',
            'explanation': f"{tx_30m} transactions initiated within 30 minutes, deviating from standard user intervals.",
            'contribution': 65
        })
    elif time_since < 1.0 and user_avg > 0:
        reasons.append({
            'severity': 'MEDIUM',
            'factor': 'Rapid Successive Transfer',
            'explanation': f"Transfer initiated less than 60 seconds ({time_since * 60:.0f}s) after preceding transaction.",
            'contribution': 40
        })
        
    # 3. Unusual hours
    is_unusual_hour = int(features.get('is_unusual_hour', 0))
    hour = int(features.get('hour_of_day', 12))
    if is_unusual_hour == 1:
        reasons.append({
            'severity': 'MEDIUM',
            'factor': 'Unusual Transaction Time',
            'explanation': f"Transaction triggered at {hour:02d}:00 hours (midnight to dawn window), outside standard daytime activity profile.",
            'contribution': 30
        })
        
    # 4. New Recipient
    is_new_rcv = int(features.get('is_new_recipient', 0))
    if is_new_rcv == 1 and (amount_ratio > 2.0 or amount > 15000):
        reasons.append({
            'severity': 'HIGH',
            'factor': 'First-Time High-Value Beneficiary',
            'explanation': "High-value transfer directed to a newly added payee with zero previous transaction history with this user.",
            'contribution': 55
        })
        
    # 5. Device and Location Anomaly
    is_device = int(features.get('is_device_changed', 0))
    is_location = int(features.get('is_location_changed', 0))
    
    if is_device == 1 and is_location == 1:
        reasons.append({
            'severity': 'CRITICAL',
            'factor': 'Dual Device & Geographic Discrepancy',
            'explanation': "Simultaneous change in both hardware device fingerprint and geographic IP/city location.",
            'contribution': 80
        })
    elif is_device == 1:
        reasons.append({
            'severity': 'MEDIUM',
            'factor': 'Unfamiliar Device Fingerprint',
            'explanation': "Payment initiated from a device model not previously recognized as primary for this UPI ID.",
            'contribution': 45
        })
    elif is_location == 1:
        reasons.append({
            'severity': 'LOW',
            'factor': 'Unusual Geographic Pattern',
            'explanation': "Originating location differs from home/frequent spending zone.",
            'contribution': 25
        })
        
    # 6. Failed Attempts
    failed = int(features.get('failed_attempts', 0))
    if failed >= 2:
        reasons.append({
            'severity': 'HIGH',
            'factor': 'Multiple Prior Failed Attempts',
            'explanation': f"{failed} consecutive PIN/payment attempts failed prior to this transaction. Possible brute-force or social engineering coerced retry.",
            'contribution': 60
        })
        
    # 7. Account Drain
    drain_ratio = float(features.get('account_drain_ratio', 0))
    if drain_ratio >= 0.85 and amount > 5000:
        reasons.append({
            'severity': 'CRITICAL',
            'factor': 'Substantial Account Balance Drain',
            'explanation': f"Transaction depletes {drain_ratio * 100:.1f}% of the remaining account balance in a single transfer.",
            'contribution': 75
        })
        
    if not reasons:
        reasons.append({
            'severity': 'INFO',
            'factor': 'Consistent with Baseline Profile',
            'explanation': "Transaction matches normal historical expenditure volume, verified device, typical hours, and known beneficiaries.",
            'contribution': 0
        })
        
    return reasons
