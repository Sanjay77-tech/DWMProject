"""
Page 3: Live Transaction Risk Prediction & Explainable AI
Interactive behavioral simulation, demonstration presets, ensemble scoring,
Isolation Forest anomaly detection, and plain-language XAI forensic reasons.
"""

import streamlit as st
import datetime
from src.pipeline_manager import predict_single_transaction

def render_transaction_prediction():
    # Header Section
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.5rem;">
        <h1 style="color: #f8fafc; margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;">
            🔍 Real-Time Transaction Risk Assessment
        </h1>
        <p style="color: #94a3b8; margin-top: 0.35rem; font-size: 0.95rem;">
            Evaluate single transactions against behavioral baseline models, unsupervised anomaly detectors, and supervised fraud classifiers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Test Scenarios for demonstration
    st.markdown("##### ⚡ Instant Scenario Presets")
    st.caption("Click any preset to simulate representative fraud attack archetypes or legitimate baseline payments.")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    
    preset_data = None
    with pcol1:
        if st.button("🟢 Normal Grocery (₹350)", use_container_width=True):
            preset_data = {
                'amount': 350.0,
                'type': 'PAYMENT',
                'hour': 14,
                'sender': 'ananya.sharma@okaxis',
                'receiver': 'fresh_groceries@paytm',
                'user_avg': 420.0,
                'tx_5m': 0, 'tx_30m': 0, 'tx_24h': 1,
                'time_since': 180.0,
                'new_rcv': 0, 'dev_chg': 0, 'loc_chg': 0,
                'failed': 0, 'old_bal': 18500.0, 'new_bal': 18150.0
            }
    with pcol2:
        if st.button("🔴 Midnight Drain (₹85,000)", use_container_width=True):
            preset_data = {
                'amount': 85000.0,
                'type': 'TRANSFER',
                'hour': 3,
                'sender': 'rahul.verma@oksbi',
                'receiver': 'mule_account_99@ybl',
                'user_avg': 1200.0,
                'tx_5m': 3, 'tx_30m': 4, 'tx_24h': 5,
                'time_since': 1.5,
                'new_rcv': 1, 'dev_chg': 1, 'loc_chg': 1,
                'failed': 2, 'old_bal': 87000.0, 'new_bal': 2000.0
            }
    with pcol3:
        if st.button("🟡 High-Velocity Burst (₹30,000)", use_container_width=True):
            preset_data = {
                'amount': 30000.0,
                'type': 'TRANSFER',
                'hour': 22,
                'sender': 'vikram.singh@icici',
                'receiver': 'crypto_exchange_p2p@axis',
                'user_avg': 2500.0,
                'tx_5m': 4, 'tx_30m': 6, 'tx_24h': 8,
                'time_since': 0.8,
                'new_rcv': 1, 'dev_chg': 0, 'loc_chg': 0,
                'failed': 1, 'old_bal': 52000.0, 'new_bal': 22000.0
            }
    with pcol4:
        if st.button("🟠 Phishing QR Scam (₹45,000)", use_container_width=True):
            preset_data = {
                'amount': 45000.0,
                'type': 'CASH_OUT',
                'hour': 18,
                'sender': 'priya.nair@hdfc',
                'receiver': 'fake_reward_refund@paytm',
                'user_avg': 1800.0,
                'tx_5m': 1, 'tx_30m': 2, 'tx_24h': 3,
                'time_since': 12.0,
                'new_rcv': 1, 'dev_chg': 1, 'loc_chg': 0,
                'failed': 3, 'old_bal': 46200.0, 'new_bal': 1200.0
            }
            
    if preset_data is not None:
        st.session_state['pred_input'] = preset_data

    stored = st.session_state.get('pred_input', {})

    st.markdown("---")
    
    # Grouped Input Form
    with st.form("prediction_form"):
        st.markdown("##### 📝 Transaction Parameters")
        
        # Section A: Financial & Identity
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input(
                "Transaction Amount (₹)", 
                min_value=1.0, 
                max_value=1000000.0, 
                value=float(stored.get('amount', 2500.0)),
                step=100.0
            )
            txn_type = st.selectbox(
                "Transaction Modality", 
                ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'BILL_PAY'],
                index=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'BILL_PAY'].index(stored.get('type', 'TRANSFER'))
            )
            hour_val = st.slider(
                "Hour of Day (0 to 23)", 
                min_value=0, max_value=23, 
                value=int(stored.get('hour', 14))
            )
            
        with c2:
            sender_id = st.text_input(
                "Sender UPI ID", 
                value=str(stored.get('sender', 'user_0042@okaxis'))
            )
            receiver_id = st.text_input(
                "Receiver / Payee UPI ID", 
                value=str(stored.get('receiver', 'merchant_8890@paytm'))
            )
            sender_avg = st.number_input(
                "Sender Historical Avg Amount (₹)", 
                min_value=1.0, 
                max_value=500000.0, 
                value=float(stored.get('user_avg', 1200.0)),
                step=100.0
            )
            
        with c3:
            old_bal = st.number_input(
                "Sender Account Balance Before (₹)", 
                min_value=0.0, 
                value=float(stored.get('old_bal', 25000.0)),
                step=1000.0
            )
            new_bal = st.number_input(
                "Sender Account Balance After (₹)", 
                min_value=0.0, 
                value=float(stored.get('new_bal', max(0.0, stored.get('old_bal', 25000.0) - amount))),
                step=1000.0
            )
            failed_attempts = st.selectbox(
                "Preceding Failed PIN Attempts", 
                [0, 1, 2, 3, 4], 
                index=[0, 1, 2, 3, 4].index(min(4, stored.get('failed', 0)))
            )

        # Section B: Velocity & Environmental Context
        st.markdown("##### ⏱️ Velocity & Environmental Context")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            tx_5m = st.number_input("Txns in Last 5 Mins", min_value=0, max_value=50, value=int(stored.get('tx_5m', 0)))
            tx_30m = st.number_input("Txns in Last 30 Mins", min_value=0, max_value=100, value=int(stored.get('tx_30m', 0)))
        with v2:
            tx_24h = st.number_input("Txns in Last 24 Hours", min_value=0, max_value=500, value=int(stored.get('tx_24h', 1)))
            time_since = st.number_input("Mins Since Previous Txn", min_value=0.1, max_value=10000.0, value=float(stored.get('time_since', 60.0)))
        with v3:
            is_new_rcv = st.selectbox(
                "Beneficiary Novelty", 
                [0, 1], 
                format_func=lambda x: "Known Frequent Payee" if x == 0 else "Brand New Recipient",
                index=int(stored.get('new_rcv', 0))
            )
            is_dev_chg = st.selectbox(
                "Device Fingerprint", 
                [0, 1], 
                format_func=lambda x: "Trusted Primary Device" if x == 0 else "Unrecognized Device / IP",
                index=int(stored.get('dev_chg', 0))
            )
        with v4:
            is_loc_chg = st.selectbox(
                "Geographic Location", 
                [0, 1], 
                format_func=lambda x: "Home / Familiar City" if x == 0 else "Unusual / New City",
                index=int(stored.get('loc_chg', 0))
            )
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🛡️ Analyse Transaction Risk", use_container_width=True, type="primary")

    if submit_btn:
        input_payload = {
            'amount': amount,
            'transaction_type': txn_type,
            'hour_of_day': hour_val,
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'sender_avg_amt': sender_avg,
            'tx_count_5m': tx_5m,
            'tx_count_30m': tx_30m,
            'tx_count_24h': tx_24h,
            'time_since_prev_min': time_since,
            'is_new_recipient': is_new_rcv,
            'is_device_changed': is_dev_chg,
            'is_location_changed': is_loc_chg,
            'failed_attempts': failed_attempts,
            'old_balance': old_bal,
            'new_balance': new_bal
        }
        
        with st.spinner("Executing Anomaly Detection & Fraud Classification Pipeline..."):
            report = predict_single_transaction(input_payload)
            
        # Store in session history
        if 'evaluated_history' not in st.session_state:
            st.session_state['evaluated_history'] = []
            
        history_entry = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'transaction_id': f"TXN{datetime.datetime.now().strftime('%M%S%f')[:8]}",
            'sender_id': sender_id,
            'receiver_id': receiver_id,
            'amount': amount,
            'risk_score': report['risk_score'],
            'risk_category': report['risk_category'],
            'prediction_status': report['prediction_status'],
            'top_reason': report['explanations'][0]['explanation'] if report['explanations'] else 'Normal Transaction'
        }
        st.session_state['evaluated_history'].insert(0, history_entry)
        
        # Display Results Section
        st.markdown("---")
        st.subheader("🎯 Risk Assessment & AI Decision")
        
        rcol1, rcol2 = st.columns([1, 1], gap="large")
        
        # Color mapping based on risk
        risk_color = report.get('color', '#10b981')
        if report['risk_category'] == 'Low Risk':
            risk_color = '#10b981'
        elif report['risk_category'] == 'Medium Risk':
            risk_color = '#f59e0b'
        elif report['risk_category'] == 'High Risk':
            risk_color = '#ef4444'
        else:
            risk_color = '#dc2626'
            
        with rcol1:
            st.markdown(f"""
            <div style="background-color: #111927; border: 1.5px solid {risk_color}; border-radius: 10px; padding: 1.6rem; text-align: center;">
                <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                    Composite Fraud Risk Score
                </span>
                <div style="color: {risk_color}; font-size: 3.2rem; font-weight: 800; margin: 0.4rem 0; letter-spacing: -0.03em;">
                    {report['risk_score']} <span style="font-size: 1.2rem; color: #64748b; font-weight: 500;">/ 100</span>
                </div>
                <div style="display: inline-block; background-color: {risk_color}22; color: {risk_color}; border: 1px solid {risk_color}; border-radius: 20px; padding: 0.3rem 1.1rem; font-weight: 600; font-size: 0.95rem;">
                    {report['risk_category']} • {report['prediction_status']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📌 Suggested Verification Protocol")
            st.info(f"**Action Protocol:** {report['suggested_action']}")
            
        with rcol2:
            st.markdown("##### 🔬 Multi-Model Confidence Breakdown")
            comps = report['components']
            
            rf_val = comps['supervised_prob'] if comps['supervised_prob'] is not None else 0.0
            st.write(f"**Random Forest Supervised Fraud Probability:** `{rf_val:.1f}%`")
            st.progress(min(1.0, rf_val / 100.0))
            
            iso_val = comps['anomaly_relative_score'] if comps['anomaly_relative_score'] is not None else 0.0
            st.write(f"**Isolation Forest Anomaly Deviation Index:** `{iso_val:.1f} / 100`")
            st.progress(min(1.0, iso_val / 100.0))
            
            if comps['semi_supervised_prob'] is not None:
                semi_val = comps['semi_supervised_prob']
                st.write(f"**Semi-Supervised Self-Training Confidence:** `{semi_val:.1f}%`")
                st.progress(min(1.0, semi_val / 100.0))
                
            st.caption("ℹ️ *Note: Anomaly scores evaluate deviation from learned baseline behavior, while supervised scores reflect similarity to known fraud attacks.*")

        # Explainable AI (XAI) Factors
        st.markdown("---")
        st.subheader("💡 Explainable AI: Why was this transaction flagged?")
        
        for exp in report['explanations']:
            sev = exp['severity']
            badge_color = "#ef4444" if sev == 'CRITICAL' else ("#f59e0b" if sev == 'HIGH' else ("#38bdf8" if sev == 'MEDIUM' else "#10b981"))
            st.markdown(f"""
            <div style="border-left: 3.5px solid {badge_color}; background-color: #111927; padding: 0.8rem 1.1rem; margin-bottom: 0.6rem; border-radius: 4px; border-top: 1px solid #1e293b; border-right: 1px solid #1e293b; border-bottom: 1px solid #1e293b;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{exp['factor']}</span>
                    <span style="background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}; padding: 0.15rem 0.6rem; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">
                        {sev}
                    </span>
                </div>
                <p style="color: #94a3b8; margin: 0.35rem 0 0 0; font-size: 0.9rem; line-height: 1.4;">{exp['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_transaction_prediction()
