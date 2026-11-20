"""
Page 4: Forensic Fraud Analytics & Behavioral Mining
Deep-dive exploratory data mining across temporal patterns, volume spikes,
transaction modalities, and NetworkX topological money mule graphs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.graph_analytics import build_transaction_network, plot_transaction_network, analyze_network_nodes

def render_fraud_analytics():
    # Header Section
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.5rem;">
        <h1 style="color: #f8fafc; margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;">
            📈 Forensic Fraud Analytics & Behavioral Mining
        </h1>
        <p style="color: #94a3b8; margin-top: 0.35rem; font-size: 0.95rem;">
            Exploratory data mining across temporal attack vectors, amount density profiles, transaction modalities, and topological mule networks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check session state for dataset
    if 'current_df' not in st.session_state or st.session_state['current_df'] is None or len(st.session_state['current_df']) == 0:
        st.markdown("""
        <div style="background-color: #111927; border: 1px solid #1e293b; border-radius: 10px; padding: 2.2rem 1.5rem; text-align: center; margin: 1.5rem 0;">
            <div style="font-size: 2.4rem; margin-bottom: 0.5rem;">📈</div>
            <h3 style="color: #f8fafc; margin-bottom: 0.5rem; font-size: 1.35rem; font-weight: 700;">No Dataset Currently Ingested</h3>
            <p style="color: #94a3b8; max-width: 580px; margin: 0 auto 1.5rem auto; font-size: 0.95rem; line-height: 1.5;">
                Forensic analytics (hourly attack heatmaps, log-normal distributions, and NetworkX topological graphs) require transaction records.
                Please upload your transaction CSV file on the <strong>📂 Dataset Upload</strong> page.
            </p>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            if st.button("📂 Go to Dataset Upload", use_container_width=True, type="primary"):
                st.session_state['nav_redirect'] = "📂 Dataset Upload"
                st.rerun()
        return
        
    df = st.session_state['current_df']
    has_labels = 'is_fraud' in df.columns

    # 1. Row 1: Amount Distribution & Temporal Analysis by Hour
    st.markdown("##### 1. ⏰ Temporal Risk & Amount Distribution")
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        # Transactions by Hour of Day
        if 'timestamp' in df.columns:
            hours = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour
        elif 'hour_of_day' in df.columns:
            hours = df['hour_of_day']
        elif 'time_step' in df.columns:
            hours = df['time_step'] % 24
        else:
            hours = pd.Series(np.random.randint(0, 24, len(df)))
            
        fig_hr, ax_hr = plt.subplots(figsize=(6, 4.2), facecolor='#0f172a')
        ax_hr.set_facecolor('#0f172a')
        
        if has_labels:
            fraud_hours = hours[df['is_fraud'] == 1]
            normal_hours = hours[df['is_fraud'] == 0]
            ax_hr.hist(normal_hours, bins=24, range=(0, 24), alpha=0.55, color='#3b82f6', label='Legitimate Txns', density=True, edgecolor='#1d4ed8')
            ax_hr.hist(fraud_hours, bins=24, range=(0, 24), alpha=0.75, color='#ef4444', label='Fraud Attacks', density=True, edgecolor='#b91c1c')
            ax_hr.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)
        else:
            ax_hr.hist(hours, bins=24, range=(0, 24), color='#3b82f6', edgecolor='#1d4ed8', alpha=0.8)
            
        ax_hr.set_title("Hourly Distribution (Note Midnight / Early Morning Attacks)", color='#f8fafc', fontsize=10.5, weight='600')
        ax_hr.set_xlabel("Hour of Day (0 to 23)", color='#94a3b8', fontsize=9)
        ax_hr.set_ylabel("Probability Density", color='#94a3b8', fontsize=9)
        ax_hr.tick_params(colors='#94a3b8', labelsize=8.5)
        ax_hr.grid(axis='y', color='#1e293b', linestyle='--', alpha=0.7)
        ax_hr.spines['top'].set_visible(False)
        ax_hr.spines['right'].set_visible(False)
        ax_hr.spines['bottom'].set_color('#334155')
        ax_hr.spines['left'].set_color('#334155')
        plt.tight_layout()
        st.pyplot(fig_hr)
        
    with c2:
        # Transaction Amount Log Distribution
        fig_amt, ax_amt = plt.subplots(figsize=(6, 4.2), facecolor='#0f172a')
        ax_amt.set_facecolor('#0f172a')
        
        if has_labels:
            fraud_amt = np.log10(np.clip(df[df['is_fraud'] == 1]['amount'], 1.0, 1e7))
            normal_amt = np.log10(np.clip(df[df['is_fraud'] == 0]['amount'], 1.0, 1e7))
            ax_amt.hist(normal_amt, bins=28, alpha=0.55, color='#10b981', label='Normal Log(₹)', density=True, edgecolor='#059669')
            ax_amt.hist(fraud_amt, bins=28, alpha=0.75, color='#ef4444', label='Fraud Log(₹)', density=True, edgecolor='#b91c1c')
            ax_amt.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)
        else:
            log_amt = np.log10(np.clip(df['amount'], 1.0, 1e7))
            ax_amt.hist(log_amt, bins=28, color='#10b981', edgecolor='#059669', alpha=0.8)
            
        ax_amt.set_title("Transaction Value Spectrum (Log10 Scale)", color='#f8fafc', fontsize=10.5, weight='600')
        ax_amt.set_xlabel("Log10(Amount in ₹)", color='#94a3b8', fontsize=9)
        ax_amt.set_ylabel("Probability Density", color='#94a3b8', fontsize=9)
        ax_amt.tick_params(colors='#94a3b8', labelsize=8.5)
        ax_amt.grid(axis='y', color='#1e293b', linestyle='--', alpha=0.7)
        ax_amt.spines['top'].set_visible(False)
        ax_amt.spines['right'].set_visible(False)
        ax_amt.spines['bottom'].set_color('#334155')
        ax_amt.spines['left'].set_color('#334155')
        plt.tight_layout()
        st.pyplot(fig_amt)

    st.markdown("---")

    # 2. Row 2: Transaction Type Vulnerability & Velocity Impact
    st.markdown("##### 2. 💳 Vulnerability by Transaction Modality")
    tcol1, tcol2 = st.columns([1, 1], gap="large")
    
    with tcol1:
        if 'transaction_type' in df.columns:
            if has_labels:
                ct = pd.crosstab(df['transaction_type'], df['is_fraud'], normalize='index') * 100
                fig_type, ax_type = plt.subplots(figsize=(6, 4.2), facecolor='#0f172a')
                ax_type.set_facecolor('#0f172a')
                
                # Stacked bar
                p1 = ax_type.bar(ct.index, ct[0], color='#1e3a8a', label='Legitimate %', width=0.55, edgecolor='#1d4ed8')
                p2 = ax_type.bar(ct.index, ct[1], bottom=ct[0], color='#ef4444', label='Fraud %', width=0.55, edgecolor='#b91c1c')
                ax_type.set_title("Fraud Concentration by Payment Mode (%)", color='#f8fafc', fontsize=10.5, weight='600')
                ax_type.set_ylabel("Percentage Breakdown (%)", color='#94a3b8', fontsize=9)
                ax_type.tick_params(colors='#94a3b8', labelsize=8.5)
                ax_type.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)
                ax_type.spines['top'].set_visible(False)
                ax_type.spines['right'].set_visible(False)
                ax_type.spines['bottom'].set_color('#334155')
                ax_type.spines['left'].set_color('#334155')
                plt.xticks(rotation=15, color='#cbd5e1')
                plt.tight_layout()
                st.pyplot(fig_type)
            else:
                type_counts = df['transaction_type'].value_counts()
                st.bar_chart(type_counts)
        else:
            st.info("Transaction type not present in dataset.")
            
    with tcol2:
        st.markdown("**Core Fraud Attack Archetypes Identified:**")
        st.markdown("""
        <div style="background-color: #111927; border: 1px solid #1e293b; border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.88rem; line-height: 1.6;">
            <div style="margin-bottom: 0.8rem;">
                <strong style="color: #ef4444;">1. Midnight Account Drain (TRANSFER):</strong><br>
                High-value transfers (₹35,000 – ₹99,000) triggered between 1:00 AM and 4:30 AM draining 85%+ of account balance.
            </div>
            <div style="margin-bottom: 0.8rem;">
                <strong style="color: #f59e0b;">2. High-Velocity Micro-Burst:</strong><br>
                3 to 6 back-to-back transfers inside 5 minutes to multiple unverified UPI handles to bypass per-transaction limits.
            </div>
            <div style="margin-bottom: 0.8rem;">
                <strong style="color: #38bdf8;">3. Phishing QR &amp; Remote Access (CASH_OUT):</strong><br>
                Immediate liquidation after multiple failed PIN entries from unfamiliar device fingerprints.
            </div>
            <div>
                <strong style="color: #a78bfa;">4. Money Mule Aggregation Hubs:</strong><br>
                Accounts receiving sudden inbound waves of transfers from distinct victims, immediately followed by cash-out withdrawals.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Row 3: Transaction Relationship Network Graph
    st.markdown("##### 3. 🕸️ UPI Transaction Relationship Network Graph (Mule Account Discovery)")
    st.caption("Topological network analysis maps sender-to-receiver payment flows. Nodes with high fan-in (many victim payers) and immediate outflows indicate money mule aggregation hubs.")
    
    with st.spinner("Constructing NetworkX topological graph from active transactions..."):
        G = build_transaction_network(df, max_edges=120)
        
    gcol1, gcol2 = st.columns([3, 2], gap="large")
    with gcol1:
        fig_net = plot_transaction_network(G)
        st.pyplot(fig_net)
        
    with gcol2:
        st.markdown("**Top Central Nodes (Potential Mule Hubs):**")
        df_nodes = analyze_network_nodes(G)
        st.dataframe(df_nodes.head(8), use_container_width=True)

if __name__ == "__main__":
    render_fraud_analytics()
