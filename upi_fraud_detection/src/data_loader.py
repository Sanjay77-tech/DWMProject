"""
Data Loader Module for UPI Scam and Fraud Detection
Handles dataset loading, column mapping, schema detection, and benchmark dataset generation.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional

# Standard internal column names used across the pipeline
STANDARD_COLUMNS = {
    'trans_id': 'transaction_id',
    'step': 'time_step',
    'type': 'transaction_type',
    'amount': 'amount',
    'nameOrig': 'sender_id',
    'oldbalanceOrg': 'old_balance_sender',
    'newbalanceOrig': 'new_balance_sender',
    'nameDest': 'receiver_id',
    'oldbalanceDest': 'old_balance_receiver',
    'newbalanceDest': 'new_balance_receiver',
    'isFraud': 'is_fraud',
    'isFlaggedFraud': 'is_flagged_fraud',
    'timestamp': 'timestamp',
    'device_id': 'device_id',
    'location': 'location',
    'status': 'status',
    'failed_attempts': 'failed_attempts'
}

# Fuzzy mapping keywords to detect uploaded custom CSV schemas
SYNONYMS = {
    'transaction_id': ['transaction_id', 'trans_id', 'tx_id', 'id', 'txid', 'reference_id'],
    'sender_id': ['sender_id', 'nameorig', 'sender', 'from_account', 'source_upi_id', 'payer_id', 'user_id'],
    'receiver_id': ['receiver_id', 'namedest', 'receiver', 'to_account', 'target_upi_id', 'payee_id', 'beneficiary'],
    'amount': ['amount', 'trans_amount', 'transaction_amount', 'amt', 'value'],
    'transaction_type': ['transaction_type', 'type', 'trans_type', 'payment_mode', 'txn_type'],
    'timestamp': ['timestamp', 'trans_date', 'date', 'datetime', 'time', 'tx_time'],
    'time_step': ['step', 'time_step', 'hour_step'],
    'old_balance_sender': ['oldbalanceorg', 'old_balance_sender', 'sender_old_balance', 'prev_balance_orig', 'balance_before'],
    'new_balance_sender': ['newbalanceorig', 'new_balance_sender', 'sender_new_balance', 'curr_balance_orig', 'balance_after'],
    'old_balance_receiver': ['oldbalancedest', 'old_balance_receiver', 'dest_old_balance'],
    'new_balance_receiver': ['newbalancedest', 'new_balance_receiver', 'dest_new_balance'],
    'is_fraud': ['isfraud', 'is_fraud', 'fraud', 'fraud_label', 'class', 'target', 'label'],
    'device_id': ['device_id', 'device', 'device_model', 'imei', 'device_fingerprint'],
    'location': ['location', 'city', 'geo_location', 'state', 'ip_location', 'trans_location'],
    'status': ['status', 'trans_status', 'payment_status', 'result'],
    'failed_attempts': ['failed_attempts', 'failed_count', 'fail_count', 'failed_tx_count']
}

def detect_column_mapping(columns: list) -> Dict[str, str]:
    """
    Intelligently maps raw dataset column names to standardized schema names.
    """
    mapping = {}
    lower_cols = {col.lower().replace(" ", "_").replace("-", "_"): col for col in columns}
    
    for std_name, possible_names in SYNONYMS.items():
        for candidate in possible_names:
            if candidate in lower_cols:
                raw_col = lower_cols[candidate]
                mapping[raw_col] = std_name
                break
                
    return mapping

def generate_synthetic_upi_dataset(num_records: int = 10000, fraud_ratio: float = 0.04, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates an authentic, academically rigorous synthetic UPI transaction dataset.
    Features realistic normal transaction behaviours (P2P, P2M, BILL_PAYMENT, SCAN_AND_PAY)
    alongside sophisticated simulated fraud patterns (burst velocity, midnight cashout, 
    account drain, QR hijacking, phishing spoofing).
    
    Returns:
        pd.DataFrame with both base attributes and real-world behavioral metadata.
    """
    np.random.seed(random_seed)
    
    # 1. Base User and Merchant Pool
    num_users = max(50, int(num_records / 15))
    num_merchants = max(30, int(num_records / 20))
    
    user_ids = [f"user_{i:04d}@okaxis" for i in range(1, num_users + 1)]
    merchant_ids = [f"merchant_{j:04d}@paytm" for j in range(1, num_merchants + 1)]
    all_recipients = user_ids + merchant_ids
    
    # User profiles: baseline spending, device, home location
    cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Pune', 'Chennai', 'Kolkata', 'Ahmedabad']
    devices = ['Samsung_Galaxy_M32', 'iPhone_13', 'Redmi_Note_11', 'OnePlus_Nord_CE', 'Vivo_V23', 'Realme_9']
    
    user_profiles = {}
    for uid in user_ids:
        user_profiles[uid] = {
            'avg_amount': np.random.uniform(300, 4500),
            'home_location': np.random.choice(cities),
            'primary_device': np.random.choice(devices),
            'balance': np.random.uniform(5000, 150000)
        }
        
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    data = []
    
    # We will generate sequential timestamps over 30 days
    current_time = start_time
    
    # Pre-determine fraud indices
    num_frauds = int(num_records * fraud_ratio)
    fraud_indices = set(np.random.choice(range(50, num_records), size=num_frauds, replace=False))
    
    txn_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'BILL_PAY']
    
    for i in range(num_records):
        txn_id = f"UPI{202601000000 + i + 1}"
        is_fraud_case = 1 if i in fraud_indices else 0
        
        # Advance time by random intervals (0.5 to 10 minutes)
        time_delta_sec = np.random.exponential(scale=180)
        current_time += timedelta(seconds=time_delta_sec)
        
        sender = np.random.choice(user_ids)
        u_profile = user_profiles[sender]
        
        if is_fraud_case:
            # Inject realistic fraud patterns
            fraud_pattern_type = np.random.choice(['midnight_burst', 'account_drain', 'unfamiliar_takeover', 'qr_phish'])
            
            if fraud_pattern_type == 'midnight_burst':
                # Large burst, midnight hour (1 AM - 4 AM)
                fraud_hour = np.random.choice([1, 2, 3, 4])
                tx_timestamp = current_time.replace(hour=fraud_hour, minute=np.random.randint(0, 59))
                amount = float(np.random.uniform(35000, 98000))
                txn_type = 'TRANSFER'
                receiver = f"mule_{np.random.randint(100, 999)}@ybl"
                device = np.random.choice(['Unknown_Android_Device', 'Linux_Emulator_VM', 'Rooted_Phone'])
                loc = np.random.choice([c for c in cities if c != u_profile['home_location']])
                failed = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
                old_bal = float(u_profile['balance'])
                new_bal = max(0.0, old_bal - amount)
                
            elif fraud_pattern_type == 'account_drain':
                # Emptying out account balance entirely
                tx_timestamp = current_time
                old_bal = float(u_profile['balance'])
                amount = round(old_bal * np.random.uniform(0.92, 1.0), 2)
                new_bal = max(0.0, old_bal - amount)
                txn_type = 'CASH_OUT'
                receiver = f"crypto_p2p_{np.random.randint(10, 99)}@icici"
                device = u_profile['primary_device']
                loc = u_profile['home_location']
                failed = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                
            elif fraud_pattern_type == 'unfamiliar_takeover':
                # New device + foreign location + high amount
                tx_timestamp = current_time
                amount = float(np.random.uniform(25000, 75000))
                txn_type = 'TRANSFER'
                receiver = f"foreign_payee_{np.random.randint(100, 999)}@oksbi"
                device = 'Foreign_iPhone_15_Jailbroken'
                loc = 'Jammu & Kashmir (Proxy IP)'
                failed = 2
                old_bal = float(u_profile['balance'])
                new_bal = max(0.0, old_bal - amount)
                
            else: # qr_phish
                tx_timestamp = current_time
                amount = float(np.random.uniform(15000, 48000))
                txn_type = 'PAYMENT'
                receiver = f"lottery_claim_{np.random.randint(100, 999)}@paytm"
                device = u_profile['primary_device']
                loc = u_profile['home_location']
                failed = np.random.choice([0, 1], p=[0.7, 0.3])
                old_bal = float(u_profile['balance'])
                new_bal = max(0.0, old_bal - amount)
                
        else:
            # Legitimate transactions
            tx_timestamp = current_time
            # Regular normal expenditure around user average
            amount = float(np.clip(np.random.lognormal(mean=np.log(u_profile['avg_amount']), sigma=0.65), 10.0, 45000.0))
            txn_type = np.random.choice(['PAYMENT', 'TRANSFER', 'DEBIT', 'BILL_PAY'], p=[0.50, 0.30, 0.12, 0.08])
            
            # Legitimate receiver: frequently seen merchant or friend
            if txn_type in ['PAYMENT', 'BILL_PAY']:
                receiver = np.random.choice(merchant_ids)
            else:
                receiver = np.random.choice([uid for uid in user_ids if uid != sender])
                
            # Mostly primary device and home location
            device = u_profile['primary_device'] if np.random.rand() > 0.05 else np.random.choice(devices)
            loc = u_profile['home_location'] if np.random.rand() > 0.06 else np.random.choice(cities)
            failed = np.random.choice([0, 1], p=[0.94, 0.06])
            
            old_bal = float(u_profile['balance'])
            new_bal = max(0.0, old_bal - amount)
            # Update user profile balance
            user_profiles[sender]['balance'] = new_bal + np.random.uniform(50, 500) # top up occasionally
            
        data.append({
            'transaction_id': txn_id,
            'timestamp': tx_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'time_step': int((tx_timestamp - start_time).total_seconds() // 3600) + 1,
            'sender_id': sender,
            'receiver_id': receiver,
            'transaction_type': txn_type,
            'amount': round(amount, 2),
            'old_balance_sender': round(old_bal, 2),
            'new_balance_sender': round(new_bal, 2),
            'device_id': device,
            'location': loc,
            'status': 'SUCCESS' if failed == 0 else ('FAILED_RETRY' if failed < 3 else 'BLOCKED'),
            'failed_attempts': failed,
            'is_fraud': is_fraud_case,
            'is_synthetic': 1
        })
        
    df = pd.DataFrame(data)
    # Chronological sort
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp_dt').reset_index(drop=True)
    df = df.drop(columns=['timestamp_dt'])
    return df

def load_dataset(file_path: Optional[str] = None, uploaded_file=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universal dataset loader that accepts a file path, an uploaded file stream from Streamlit,
    or falls back to generating/loading the synthetic benchmark dataset.
    
    Returns:
        df: pd.DataFrame with standardized columns
        metadata: Dict with dataset stats and detected schema details
    """
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {str(e)}")
    elif file_path and os.path.exists(file_path):
        df_raw = pd.read_csv(file_path)
    else:
        # Default benchmark path
        raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        default_file = os.path.join(raw_dir, 'upi_benchmark_transactions.csv')
        
        if os.path.exists(default_file):
            df_raw = pd.read_csv(default_file)
        else:
            print("Generating initial synthetic UPI transaction dataset...")
            df_raw = generate_synthetic_upi_dataset(num_records=7500, fraud_ratio=0.045)
            df_raw.to_csv(default_file, index=False)
            
    # Detect and standardize columns
    mapping = detect_column_mapping(df_raw.columns.tolist())
    df_std = df_raw.rename(columns=mapping).copy()
    
    # Sanitize string and numeric types to ensure PyArrow compatibility
    string_cols = ['transaction_id', 'sender_id', 'receiver_id', 'device_id', 'location', 'status', 'transaction_type']
    for sc in string_cols:
        if sc in df_std.columns:
            df_std[sc] = df_std[sc].fillna('Unknown').astype(str)
            
    numeric_cols = ['amount', 'old_balance_sender', 'new_balance_sender', 'failed_attempts', 'time_step', 'is_fraud']
    for nc in numeric_cols:
        if nc in df_std.columns:
            df_std[nc] = pd.to_numeric(df_std[nc], errors='coerce').fillna(0)
            
    # Identify which columns were mapped
    std_detected = list(mapping.values())
    has_labels = 'is_fraud' in df_std.columns
    
    fname = getattr(uploaded_file, 'name', os.path.basename(file_path) if file_path else 'upi_benchmark_transactions.csv')
    
    metadata = {
        'filename': fname,
        'total_rows': len(df_std),
        'total_columns': len(df_raw.columns),
        'raw_columns': list(df_raw.columns),
        'mapped_columns': mapping,
        'dtypes': {col: str(dtype) for col, dtype in df_raw.dtypes.items()},
        'missing_values_total': int(df_raw.isnull().sum().sum()),
        'missing_per_column': df_raw.isnull().sum().to_dict(),
        'duplicate_rows': int(df_raw.duplicated().sum()),
        'has_labels': has_labels,
        'fraud_count': int(df_std['is_fraud'].sum()) if has_labels else 0,
        'fraud_rate': float(df_std['is_fraud'].mean() * 100) if has_labels and len(df_std) > 0 else 0.0,
        'has_timestamp': 'timestamp' in df_std.columns or 'time_step' in df_std.columns,
        'has_device': 'device_id' in df_std.columns,
        'has_location': 'location' in df_std.columns,
        'has_balance': 'old_balance_sender' in df_std.columns and 'new_balance_sender' in df_std.columns
    }
    
    return df_std, metadata
