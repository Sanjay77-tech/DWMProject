"""
Analytics Router: Temporal attack patterns, log-normal distributions,
transaction modality vulnerability, and NetworkX topological mule graphs.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from backend.state import state
from src.graph_analytics import build_transaction_network, analyze_network_nodes

router = APIRouter()

@router.get("/all")
async def get_all_analytics():
    if not state.is_dataset_loaded():
        raise HTTPException(status_code=400, detail="No dataset loaded. Ingest a dataset first.")
        
    df = state.current_df
    has_labels = 'is_fraud' in df.columns
    
    # 1. Hourly Temporal Pattern (0 - 23)
    if 'timestamp' in df.columns:
        hours = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour.fillna(12).astype(int)
    elif 'hour_of_day' in df.columns:
        hours = df['hour_of_day'].astype(int)
    elif 'time_step' in df.columns:
        hours = (df['time_step'] % 24).astype(int)
    else:
        hours = pd.Series(np.zeros(len(df), dtype=int))
        
    hourly_data = []
    for h in range(24):
        h_mask = (hours == h)
        if has_labels:
            n_cnt = int(((df['is_fraud'] == 0) & h_mask).sum())
            f_cnt = int(((df['is_fraud'] == 1) & h_mask).sum())
        else:
            n_cnt = int(h_mask.sum())
            f_cnt = 0
        hourly_data.append({
            "hour": h,
            "label": f"{h:02d}:00",
            "normal_count": n_cnt,
            "fraud_count": f_cnt,
            "total": n_cnt + f_cnt
        })

    # 2. Transaction Amount Distribution (Log10 bins)
    bins = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
    bin_labels = [
        "₹10–₹30",
        "₹30–₹100",
        "₹100–₹300",
        "₹300–₹1k",
        "₹1k–₹3k",
        "₹3k–₹10k",
        "₹10k–₹30k",
        "₹30k–₹100k",
        "₹100k–₹300k",
        "₹300k+"
    ]
    amounts = df['amount'].fillna(0).values
    amount_distribution = []
    for i in range(len(bins)):
        b_low = bins[i]
        b_high = bins[i+1] if i + 1 < len(bins) else 1e9
        lbl = bin_labels[i]
        mask = (amounts >= b_low) & (amounts < b_high) if i + 1 < len(bins) else (amounts >= b_low)
        if has_labels:
            n_amt = int(((df['is_fraud'] == 0) & mask).sum())
            f_amt = int(((df['is_fraud'] == 1) & mask).sum())
        else:
            n_amt = int(mask.sum())
            f_amt = 0
        tot = n_amt + f_amt
        amount_distribution.append({
            "range": lbl,
            "count": tot,
            "total": tot,
            "normal": n_amt,
            "fraud": f_amt
        })

    # 3. Transaction Type Vulnerability
    modality_vulnerability = []
    if 'transaction_type' in df.columns:
        types = df['transaction_type'].unique()
        for t in types:
            t_mask = (df['transaction_type'] == t)
            t_total = int(t_mask.sum())
            if t_total > 0:
                if has_labels:
                    f_cases = int(((df['is_fraud'] == 1) & t_mask).sum())
                    n_cases = t_total - f_cases
                    f_rate = round(f_cases / t_total * 100, 2)
                else:
                    f_cases = 0
                    n_cases = t_total
                    f_rate = 0.0
                modality_vulnerability.append({
                    "type": str(t),
                    "total": t_total,
                    "legitimate": n_cases,
                    "fraud": f_cases,
                    "fraud_rate": f_rate
                })

    # 4. Topological Network Graph (Mule Accounts)
    G = build_transaction_network(df, max_edges=80)
    top_nodes_df = analyze_network_nodes(G)
    top_nodes = top_nodes_df.head(10).to_dict(orient='records')
    
    # Format graph nodes & links for visual rendering
    nodes = []
    node_set = set()
    for n, data in G.nodes(data=True):
        nodes.append({
            "id": str(n),
            "label": str(n).split('@')[0] if '@' in str(n) else str(n)[:12],
            "in_degree": data.get('in_degree', 0),
            "out_degree": data.get('out_degree', 0),
            "is_hub": bool(data.get('is_hub', False)),
            "is_fraud_involved": bool(data.get('is_fraud_involved', False)),
            "total_volume": float(data.get('total_volume', 0.0))
        })
        node_set.add(str(n))
        
    links = []
    for u, v, d in G.edges(data=True):
        if str(u) in node_set and str(v) in node_set:
            links.append({
                "source": str(u),
                "target": str(v),
                "amount": float(d.get('amount', 0.0)),
                "is_fraud": bool(d.get('is_fraud', False))
            })

    # 5. Core Fraud Archetypes
    archetypes = [
        {
            "id": 1,
            "title": "Midnight Account Drain (TRANSFER)",
            "severity": "CRITICAL",
            "color": "#ef4444",
            "description": "High-value transfers (₹35,000 – ₹99,000) triggered between 1:00 AM and 4:30 AM draining 85%+ of account balance."
        },
        {
            "id": 2,
            "title": "High-Velocity Micro-Burst",
            "severity": "HIGH",
            "color": "#f59e0b",
            "description": "3 to 6 back-to-back transfers inside 5 minutes to multiple unverified UPI handles to bypass per-transaction limits."
        },
        {
            "id": 3,
            "title": "Phishing QR & Remote Access (CASH_OUT)",
            "severity": "HIGH",
            "color": "#38bdf8",
            "description": "Immediate liquidation after multiple failed PIN entries from unfamiliar device fingerprints."
        },
        {
            "id": 4,
            "title": "Money Mule Aggregation Hubs",
            "severity": "MEDIUM",
            "color": "#a78bfa",
            "description": "Accounts receiving sudden inbound waves of transfers from distinct victims, immediately followed by cash-out withdrawals."
        }
    ]

    return {
        "hourly_pattern": hourly_data,
        "amount_distribution": amount_distribution,
        "modality_vulnerability": modality_vulnerability,
        "network_graph": {
            "nodes": nodes,
            "links": links,
            "top_nodes": top_nodes
        },
        "archetypes": archetypes
    }
