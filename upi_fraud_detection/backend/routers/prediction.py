"""
Prediction Router: Real-time single transaction inference, scenario presets, and XAI attribution.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import datetime

from backend.state import state
from src.pipeline_manager import predict_single_transaction

router = APIRouter()

class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0)
    transaction_type: str = "TRANSFER"
    hour_of_day: int = Field(14, ge=0, le=23)
    sender_id: str = "user_0042@okaxis"
    receiver_id: str = "merchant_8890@paytm"
    sender_avg_amt: float = 1200.0
    tx_count_5m: int = 0
    tx_count_30m: int = 0
    tx_count_24h: int = 1
    time_since_prev_min: float = 60.0
    is_new_recipient: int = 0
    is_device_changed: int = 0
    is_location_changed: int = 0
    failed_attempts: int = 0
    old_balance: float = 25000.0
    new_balance: float = 22500.0

@router.post("/single")
async def analyze_single_transaction(txn: TransactionInput):
    payload = txn.dict()
    try:
        report = predict_single_transaction(payload)
        
        # Save to evaluated history log
        history_entry = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'transaction_id': f"TXN{datetime.datetime.now().strftime('%M%S%f')[:8]}",
            'sender_id': txn.sender_id,
            'receiver_id': txn.receiver_id,
            'amount': txn.amount,
            'risk_score': report['risk_score'],
            'risk_category': report['risk_category'],
            'prediction_status': report['prediction_status'],
            'top_reason': report['explanations'][0]['explanation'] if report['explanations'] else 'Normal Transaction'
        }
        state.evaluated_history.insert(0, history_entry)
        
        return {
            "status": "success",
            "report": report,
            "transaction_record": history_entry
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.get("/presets")
async def get_scenario_presets():
    presets = [
        {
            "id": "normal_merchant",
            "title": "Normal Grocery Payment",
            "category": "legitimate",
            "badge": "🟢 Normal (₹350)",
            "data": {
                "amount": 350.0,
                "transaction_type": "PAYMENT",
                "hour_of_day": 14,
                "sender_id": "ananya.sharma@okaxis",
                "receiver_id": "fresh_groceries@paytm",
                "sender_avg_amt": 420.0,
                "tx_count_5m": 0, "tx_count_30m": 0, "tx_count_24h": 1,
                "time_since_prev_min": 180.0,
                "is_new_recipient": 0, "is_device_changed": 0, "is_location_changed": 0,
                "failed_attempts": 0, "old_balance": 18500.0, "new_balance": 18150.0
            }
        },
        {
            "id": "midnight_drain",
            "title": "Midnight Account Drain",
            "category": "fraud",
            "badge": "🔴 Midnight Drain (₹85,000)",
            "data": {
                "amount": 85000.0,
                "transaction_type": "TRANSFER",
                "hour_of_day": 3,
                "sender_id": "rahul.verma@oksbi",
                "receiver_id": "mule_account_99@ybl",
                "sender_avg_amt": 1200.0,
                "tx_count_5m": 3, "tx_count_30m": 4, "tx_count_24h": 5,
                "time_since_prev_min": 1.5,
                "is_new_recipient": 1, "is_device_changed": 1, "is_location_changed": 1,
                "failed_attempts": 2, "old_balance": 87000.0, "new_balance": 2000.0
            }
        },
        {
            "id": "velocity_burst",
            "title": "High-Velocity Micro-Burst",
            "category": "suspicious",
            "badge": "🟡 Velocity Burst (₹30,000)",
            "data": {
                "amount": 30000.0,
                "transaction_type": "TRANSFER",
                "hour_of_day": 22,
                "sender_id": "vikram.singh@icici",
                "receiver_id": "crypto_exchange_p2p@axis",
                "sender_avg_amt": 2500.0,
                "tx_count_5m": 4, "tx_count_30m": 6, "tx_count_24h": 8,
                "time_since_prev_min": 0.8,
                "is_new_recipient": 1, "is_device_changed": 0, "is_location_changed": 0,
                "failed_attempts": 1, "old_balance": 52000.0, "new_balance": 22000.0
            }
        },
        {
            "id": "phishing_qr",
            "title": "Phishing QR Scam Cash-Out",
            "category": "fraud",
            "badge": "🟠 Phishing QR (₹45,000)",
            "data": {
                "amount": 45000.0,
                "transaction_type": "CASH_OUT",
                "hour_of_day": 18,
                "sender_id": "priya.nair@hdfc",
                "receiver_id": "fake_reward_refund@paytm",
                "sender_avg_amt": 1800.0,
                "tx_count_5m": 1, "tx_count_30m": 2, "tx_count_24h": 3,
                "time_since_prev_min": 12.0,
                "is_new_recipient": 1, "is_device_changed": 1, "is_location_changed": 0,
                "failed_attempts": 3, "old_balance": 46200.0, "new_balance": 1200.0
            }
        }
    ]
    return {"presets": presets}
