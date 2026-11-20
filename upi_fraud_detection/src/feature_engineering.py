"""
Feature Engineering Module for UPI Scam and Fraud Detection
Extracts behavioral, temporal, velocity, and relational features
while strictly preventing data leakage (only historical transactions prior to current time).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def extract_transaction_features(df: pd.DataFrame, user_profiles: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Computes time-aware, behavioral, and anomaly indicators for each transaction.
    Guarantees no future look-ahead data leakage by strictly sorting chronologically
    and updating user state incrementally.
    """
    df = df.copy()
    
    # 1. Parse Timestamps
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'time_step' in df.columns:
        # Step in hours from epoch 2026-01-01
        base = pd.to_datetime('2026-01-01 00:00:00')
        df['datetime'] = base + pd.to_timedelta(df['time_step'], unit='h')
    else:
        df['datetime'] = pd.date_range(start='2026-01-01', periods=len(df), freq='5min')

    # Fill NaT if any
    if df['datetime'].isnull().any():
        df['datetime'] = df['datetime'].fillna(pd.to_datetime('2026-01-01 00:00:00'))

    # Sort strictly chronologically to prevent temporal data leakage
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 2. Time-Based Features
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_of_week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    # High-risk hours: typically 00:00 to 05:00 AM (often associated with unauthorized account drains)
    df['is_unusual_hour'] = df['hour_of_day'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    
    # 3. User Historical Behavior & Velocity Features
    # We maintain user-level rolling statistics strictly using prior transactions.
    
    has_sender = 'sender_id' in df.columns
    has_receiver = 'receiver_id' in df.columns
    has_device = 'device_id' in df.columns
    has_location = 'location' in df.columns
    has_balances = 'old_balance_sender' in df.columns and 'new_balance_sender' in df.columns
    has_failed = 'failed_attempts' in df.columns
    
    n = len(df)
    
    # Pre-allocate feature arrays for maximum speed
    user_avg_amount = np.zeros(n, dtype=np.float32)
    amount_deviation = np.zeros(n, dtype=np.float32)
    amount_ratio_avg = np.zeros(n, dtype=np.float32)
    time_since_prev_tx = np.zeros(n, dtype=np.float32)
    tx_count_5m = np.zeros(n, dtype=np.int32)
    tx_count_30m = np.zeros(n, dtype=np.int32)
    tx_count_24h = np.zeros(n, dtype=np.int32)
    is_new_recipient = np.zeros(n, dtype=np.int32)
    is_device_changed = np.zeros(n, dtype=np.int32)
    is_location_changed = np.zeros(n, dtype=np.int32)
    balance_error = np.zeros(n, dtype=np.float32)
    account_drain_ratio = np.zeros(n, dtype=np.float32)
    
    # Tracking dictionaries for chronological state
    user_tx_history = {}      # sender -> list of (timestamp, amount)
    user_recipients = {}      # sender -> set of known receivers
    user_devices = {}         # sender -> last seen device
    user_locations = {}       # sender -> last seen location
    recipient_counts = {}     # receiver -> count
    
    amounts = df['amount'].values
    timestamps = df['datetime'].values
    senders = df['sender_id'].values if has_sender else ['user_default'] * n
    receivers = df['receiver_id'].values if has_receiver else ['rec_default'] * n
    devices = df['device_id'].values if has_device else [None] * n
    locations = df['location'].values if has_location else [None] * n
    
    if has_balances:
        old_bals = df['old_balance_sender'].values
        new_bals = df['new_balance_sender'].values
    else:
        old_bals = np.zeros(n)
        new_bals = np.zeros(n)
        
    for i in range(n):
        curr_time = timestamps[i]
        curr_amt = float(amounts[i])
        snd = senders[i]
        rcv = receivers[i]
        dev = devices[i]
        loc = locations[i]
        
        # User Prior History
        if snd in user_tx_history and len(user_tx_history[snd]) > 0:
            history = user_tx_history[snd]
            prior_amts = [h[1] for h in history]
            prior_times = [h[0] for h in history]
            
            avg_amt = float(np.mean(prior_amts))
            user_avg_amount[i] = avg_amt
            amount_deviation[i] = curr_amt - avg_amt
            amount_ratio_avg[i] = curr_amt / (avg_amt + 1.0)
            
            # Time delta in seconds from last transaction
            last_time = prior_times[-1]
            dt_sec = (curr_time - last_time) / np.timedelta64(1, 's')
            time_since_prev_tx[i] = max(0.0, float(dt_sec))
            
            # Velocity features: transactions in last 5m, 30m, 24h
            t_curr = curr_time
            t_5m = t_curr - np.timedelta64(5, 'm')
            t_30m = t_curr - np.timedelta64(30, 'm')
            t_24h = t_curr - np.timedelta64(24, 'h')
            
            # Count historical timestamps in intervals
            tx_count_5m[i] = sum(1 for t in prior_times if t >= t_5m)
            tx_count_30m[i] = sum(1 for t in prior_times if t >= t_30m)
            tx_count_24h[i] = sum(1 for t in prior_times if t >= t_24h)
            
            # Recipient novelty
            if snd in user_recipients:
                is_new_recipient[i] = 1 if rcv not in user_recipients[snd] else 0
            else:
                is_new_recipient[i] = 1
                
            # Device change
            if dev is not None and snd in user_devices:
                is_device_changed[i] = 1 if user_devices[snd] != dev else 0
            else:
                is_device_changed[i] = 0
                
            # Location change
            if loc is not None and snd in user_locations:
                is_location_changed[i] = 1 if user_locations[snd] != loc else 0
            else:
                is_location_changed[i] = 0
                
        else:
            # Cold start / First transaction observed for this sender
            user_avg_amount[i] = curr_amt
            amount_deviation[i] = 0.0
            amount_ratio_avg[i] = 1.0
            time_since_prev_tx[i] = 86400.0  # Default 24h
            tx_count_5m[i] = 0
            tx_count_30m[i] = 0
            tx_count_24h[i] = 0
            is_new_recipient[i] = 1
            is_device_changed[i] = 0
            is_location_changed[i] = 0
            
        # Balance Discrepancy & Drain Analysis
        if has_balances:
            ob = old_bals[i]
            nb = new_bals[i]
            # Mathematical balance delta: oldbalance - amount - newbalance
            balance_error[i] = float(ob - curr_amt - nb)
            account_drain_ratio[i] = float(curr_amt / (ob + 1.0)) if ob > 0 else 0.0
        else:
            balance_error[i] = 0.0
            account_drain_ratio[i] = 0.0
            
        # Update state with current transaction
        if snd not in user_tx_history:
            user_tx_history[snd] = []
            user_recipients[snd] = set()
        user_tx_history[snd].append((curr_time, curr_amt))
        user_recipients[snd].add(rcv)
        user_devices[snd] = dev
        user_locations[snd] = loc
        recipient_counts[rcv] = recipient_counts.get(rcv, 0) + 1

    # Attach computed features to dataframe
    df['user_avg_amount'] = user_avg_amount
    df['amount_deviation'] = amount_deviation
    df['amount_ratio_avg'] = amount_ratio_avg
    df['time_since_prev_tx_min'] = time_since_prev_tx / 60.0
    df['tx_count_5m'] = tx_count_5m
    df['tx_count_30m'] = tx_count_30m
    df['tx_count_24h'] = tx_count_24h
    df['is_new_recipient'] = is_new_recipient
    df['is_device_changed'] = is_device_changed
    df['is_location_changed'] = is_location_changed
    df['balance_error'] = balance_error
    df['account_drain_ratio'] = account_drain_ratio
    
    # Failed attempts handling
    if has_failed:
        df['failed_attempts'] = df['failed_attempts'].fillna(0).astype(int)
    else:
        df['failed_attempts'] = 0
        
    # Transaction type encoding flags
    if 'transaction_type' in df.columns:
        df['is_transfer'] = (df['transaction_type'] == 'TRANSFER').astype(int)
        df['is_cash_out'] = (df['transaction_type'] == 'CASH_OUT').astype(int)
    else:
        df['is_transfer'] = 0
        df['is_cash_out'] = 0
        
    return df

def extract_single_transaction_features(
    amount: float,
    txn_type: str,
    hour: int,
    sender_avg_amt: float,
    tx_count_5m: int,
    tx_count_30m: int,
    tx_count_24h: int,
    time_since_prev_min: float,
    is_new_recipient: int,
    is_device_changed: int,
    is_location_changed: int,
    failed_attempts: int,
    old_balance: float = 0.0,
    new_balance: float = 0.0
) -> Dict[str, float]:
    """
    Transforms user inputs from the prediction form into the exact feature vector
    required by trained models.
    """
    user_avg = sender_avg_amt if sender_avg_amt > 0 else amount
    amount_dev = amount - user_avg
    amount_ratio = amount / (user_avg + 1.0)
    is_unusual_hour = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
    
    drain_ratio = (amount / (old_balance + 1.0)) if old_balance > 0 else 0.0
    bal_error = (old_balance - amount - new_balance) if old_balance > 0 else 0.0
    
    return {
        'amount': float(amount),
        'user_avg_amount': float(user_avg),
        'amount_deviation': float(amount_dev),
        'amount_ratio_avg': float(amount_ratio),
        'hour_of_day': int(hour),
        'is_unusual_hour': int(is_unusual_hour),
        'tx_count_5m': int(tx_count_5m),
        'tx_count_30m': int(tx_count_30m),
        'tx_count_24h': int(tx_count_24h),
        'time_since_prev_tx_min': float(time_since_prev_min),
        'is_new_recipient': int(is_new_recipient),
        'is_device_changed': int(is_device_changed),
        'is_location_changed': int(is_location_changed),
        'failed_attempts': int(failed_attempts),
        'balance_error': float(bal_error),
        'account_drain_ratio': float(drain_ratio),
        'is_transfer': 1 if txn_type == 'TRANSFER' else 0,
        'is_cash_out': 1 if txn_type == 'CASH_OUT' else 0
    }
