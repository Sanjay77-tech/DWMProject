"""
Dashboard Router for UPI Sentinel AI.
Computes genuine aggregated KPI metrics, risk distributions, and alerts
from actual loaded dataset records and persistent SQLite live predictions.
Zero fake transaction counts or fabricated data.
"""

from fastapi import APIRouter
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from backend.state import state
from src.database import get_prediction_summary_kpis, get_connection

router = APIRouter()

@router.get("/stats")
async def get_dashboard_stats():
    # 1. Fetch live prediction stats from SQLite
    pred_stats = get_prediction_summary_kpis()
    live_pred_count = pred_stats.get('total_predictions', 0)
    
    has_dataset = state.is_dataset_loaded()
    
    if not has_dataset and live_pred_count == 0:
        # Return clean empty state
        return {
            "is_empty": True,
            "message": "No active dataset ingested and no live predictions made yet.",
            "kpis": {
                "total_transactions": 0,
                "legitimate_transactions": 0,
                "suspicious_transactions": 0,
                "potential_frauds": 0,
                "fraud_rate": 0.0,
                "average_amount": 0.0,
                "max_amount": 0.0,
                "unique_senders": 0,
                "features_count": 0,
                "live_predictions_count": 0
            },
            "classification_breakdown": [
                {"name": "Legitimate", "value": 0, "color": "#10b981", "pct": 0.0},
                {"name": "Suspicious", "value": 0, "color": "#f59e0b", "pct": 0.0},
                {"name": "Fraud", "value": 0, "color": "#ef4444", "pct": 0.0}
            ],
            "risk_distribution": [
                {"tier": "Low Risk", "count": 0, "color": "#10b981", "pct": 0.0},
                {"tier": "Medium Risk", "count": 0, "color": "#f59e0b", "pct": 0.0},
                {"tier": "High Risk", "count": 0, "color": "#f97316", "pct": 0.0},
                {"tier": "Very High Risk", "count": 0, "color": "#ef4444", "pct": 0.0}
            ],
            "volume_by_type": [],
            "alert_stream": [],
            "active_source_label": "No Data Active"
        }

    # 2. Extract dataset transactions if present
    df = state.current_df if has_dataset else pd.DataFrame()
    ds_total = len(df)
    has_labels = ('is_fraud' in df.columns) if has_dataset else False
    has_scores = ('risk_score' in df.columns) if has_dataset else False

    # Base counts from dataset
    if has_labels:
        ds_fraud = int(df['is_fraud'].sum())
        ds_normal = ds_total - ds_fraud
    else:
        ds_fraud = 0
        ds_normal = ds_total

    # Incorporate live predictions from SQLite
    live_fraud = int(pred_stats.get('fraud_or_high_risk', 0))
    live_normal = int(pred_stats.get('legitimate_count', 0))
    live_suspicious = int(pred_stats.get('medium_risk_count', 0))

    total_tx = ds_total + live_pred_count
    total_fraud = ds_fraud + live_fraud
    total_normal = ds_normal + live_normal

    # Suspicious count
    if has_scores:
        scores = df['risk_score']
        ds_suspicious = int(((scores > 30.0) & (scores <= 60.0)).sum())
    else:
        ds_suspicious = 0

    total_suspicious = ds_suspicious + live_suspicious
    fraud_rate = float((total_fraud / total_tx) * 100.0) if total_tx > 0 else 0.0

    # Amounts
    amounts = []
    if has_dataset and 'amount' in df.columns:
        amounts.extend(df['amount'].dropna().tolist())
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT amount FROM predictions")
        for r in cursor.fetchall():
            amounts.append(float(r['amount']))

    avg_amt = float(np.mean(amounts)) if amounts else 0.0
    max_amt = float(np.max(amounts)) if amounts else 0.0

    # Classification breakdown for Donut Chart
    classification_breakdown = [
        {"name": "Legitimate", "value": total_normal, "color": "#10b981", "pct": round((total_normal / total_tx * 100.0), 1) if total_tx > 0 else 0.0},
        {"name": "Suspicious", "value": total_suspicious, "color": "#f59e0b", "pct": round((total_suspicious / total_tx * 100.0), 1) if total_tx > 0 else 0.0},
        {"name": "Fraud", "value": total_fraud, "color": "#ef4444", "pct": round((total_fraud / total_tx * 100.0), 1) if total_tx > 0 else 0.0}
    ]

    # Risk Tier Breakdown
    low_c = int(pred_stats.get('legitimate_count', 0))
    med_c = int(pred_stats.get('medium_risk_count', 0))
    high_c = int(pred_stats.get('high_risk_count', 0))
    vhigh_c = int(pred_stats.get('critical_risk_count', 0))

    if has_scores:
        scores = df['risk_score']
        low_c += int((scores <= 30.0).sum())
        med_c += int(((scores > 30.0) & (scores <= 60.0)).sum())
        high_c += int(((scores > 60.0) & (scores <= 80.0)).sum())
        vhigh_c += int((scores > 80.0).sum())
    elif has_labels:
        # Categorize by ground truth
        low_c += ds_normal
        vhigh_c += ds_fraud

    risk_distribution = [
        {"tier": "Low Risk", "count": low_c, "color": "#10b981", "pct": round(low_c / total_tx * 100.0, 1) if total_tx > 0 else 0.0},
        {"tier": "Medium Risk", "count": med_c, "color": "#f59e0b", "pct": round(med_c / total_tx * 100.0, 1) if total_tx > 0 else 0.0},
        {"tier": "High Risk", "count": high_c, "color": "#f97316", "pct": round(high_c / total_tx * 100.0, 1) if total_tx > 0 else 0.0},
        {"tier": "Very High Risk", "count": vhigh_c, "color": "#ef4444", "pct": round(vhigh_c / total_tx * 100.0, 1) if total_tx > 0 else 0.0}
    ]

    # Volume by Type
    volume_by_type = []
    if has_dataset and 'transaction_type' in df.columns:
        vc = df['transaction_type'].value_counts()
        for txn_t, cnt in vc.items():
            volume_by_type.append({"type": str(txn_t), "count": int(cnt)})

    # Recent Alerts from live predictions and flagged dataset rows
    alert_stream = []
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT transaction_id, sender_id, receiver_id, amount, transaction_type, risk_score, risk_category, top_reason
        FROM predictions
        WHERE risk_score > 60.0
        ORDER BY id DESC
        LIMIT 6
        """)
        for r in cursor.fetchall():
            alert_stream.append(dict(r))

    if len(alert_stream) < 6 and has_dataset and has_labels:
        flagged_sample = df[df['is_fraud'] == 1].tail(6 - len(alert_stream))
        cols = [c for c in ['transaction_id', 'sender_id', 'receiver_id', 'amount', 'transaction_type', 'location', 'device_id', 'risk_score'] if c in df.columns]
        for row_dict in flagged_sample[cols].to_dict(orient='records'):
            row_dict['risk_category'] = 'High Risk'
            alert_stream.append(row_dict)

    unique_senders = int(df['sender_id'].nunique()) if (has_dataset and 'sender_id' in df.columns) else 0

    return {
        "kpis": {
            "total_transactions": total_tx,
            "legitimate_transactions": total_normal,
            "suspicious_transactions": total_suspicious,
            "potential_frauds": total_fraud,
            "fraud_rate": round(fraud_rate, 2),
            "average_amount": round(avg_amt, 2),
            "max_amount": round(max_amt, 2),
            "unique_senders": unique_senders,
            "features_count": len(df.columns) if has_dataset else 18,
            "live_predictions_count": live_pred_count
        },
        "classification_breakdown": classification_breakdown,
        "risk_distribution": risk_distribution,
        "volume_by_type": volume_by_type,
        "alert_stream": alert_stream,
        "active_source_label": state.active_source_label if has_dataset else "Persistent SQLite Ledger"
    }
