"""
Main Streamlit Application Entrypoint for
AI-Based UPI Scam and Fraud Detection System
Using Semi-Supervised Learning, Anomaly Detection and Transaction Behaviour Analysis.
"""

import os
import sys
import streamlit as st

# Add project root to sys.path to enable clean module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from views.dataset_upload import render_dataset_upload
from views.dashboard import render_dashboard
from views.transaction_prediction import render_transaction_prediction
from views.fraud_analytics import render_fraud_analytics
from views.model_training import render_model_training
from views.transaction_history import render_transaction_history

def setup_page_styling():
    st.set_page_config(
        page_title="UPI Sentinel AI | Fraud & Risk Engine",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for high-end fintech cybersecurity aesthetic
    st.markdown("""
    <style>
        /* Base application canvas */
        .stApp {
            background-color: #0a0f1d;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        
        /* Metric cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #131c31 0%, #0d1527 100%);
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #334155;
            transform: translateY(-1px);
        }
        div[data-testid="stMetricLabel"] {
            color: #94a3b8;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.02em;
            padding: 0.5rem 1.1rem;
            transition: all 0.15s ease-in-out;
            background-color: #162032;
            color: #e2e8f0;
            border: 1px solid #26354a;
        }
        .stButton>button:hover {
            border-color: #10b981;
            color: #10b981;
            background-color: #1a273e;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
        }
        .stButton>button[kind="primary"] {
            background-color: #059669;
            color: #ffffff;
            border: 1px solid #10b981;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #047857;
            border-color: #34d399;
            color: #ffffff;
            box-shadow: 0 0 14px rgba(16, 185, 129, 0.3);
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #f8fafc;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        h4, h5, h6 {
            color: #e2e8f0;
            font-weight: 600;
        }
        p, span, label {
            color: #cbd5e1;
        }
        
        /* File uploader styling */
        div[data-testid="stFileUploader"] {
            background-color: #111927;
            border: 1.5px dashed #334155;
            border-radius: 10px;
            padding: 1rem;
            transition: border-color 0.2s ease;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: #10b981;
        }
        
        /* Tabs styling */
        button[data-baseweb="tab"] {
            color: #94a3b8;
            font-weight: 600;
            padding-top: 0.6rem;
            padding-bottom: 0.6rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #10b981 !important;
            border-bottom-color: #10b981 !important;
        }
        
        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1e293b;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Radio selection in sidebar */
        div[data-testid="stRadio"] > div {
            gap: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    setup_page_styling()
    
    # Sidebar Header Branding
    st.sidebar.markdown("""
    <div style="padding: 0.8rem 0 1.2rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.2rem;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.3rem;">
            <div style="font-size: 1.8rem; line-height: 1;">🛡️</div>
            <div>
                <h3 style="color: #f8fafc; margin: 0; font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em;">
                    UPI Sentinel AI
                </h3>
                <span style="color: #10b981; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                    Fintech Fraud Defense
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Radio - Exactly the 6 options with Dataset Upload first
    nav_options = [
        "📂 Dataset Upload",
        "🏠 Dashboard",
        "🔍 Transaction Prediction",
        "📈 Fraud Analytics",
        "⚙️ Model Training",
        "📜 Transaction History"
    ]
    
    # Allow programmatic tab switching if requested via session state
    nav_default_index = 0
    if 'nav_redirect' in st.session_state and st.session_state['nav_redirect'] in nav_options:
        nav_default_index = nav_options.index(st.session_state['nav_redirect'])
        del st.session_state['nav_redirect']
        
    selected_page = st.sidebar.radio(
        "NAVIGATION MENU",
        nav_options,
        index=nav_default_index
    )
    
    st.sidebar.markdown("---")
    
    # System Monitor Panel in Sidebar
    st.sidebar.markdown("#### ⚡ System Monitor")
    
    has_dataset = 'current_df' in st.session_state and st.session_state['current_df'] is not None and len(st.session_state['current_df']) > 0
    if has_dataset:
        dataset_rows = len(st.session_state['current_df'])
        active_ds_name = st.session_state.get('dataset_source_label', f"Custom Data ({dataset_rows:,} records)")
        st.sidebar.markdown(f"- **Data in Memory:** `{active_ds_name}`")
        st.sidebar.markdown("- **Engine Status:** `🟢 Active Operational`")
    else:
        st.sidebar.markdown("- **Data in Memory:** `🟡 Awaiting Ingestion`")
        st.sidebar.markdown("- **Engine Status:** `⚪ Idle (Upload Required)`")
        
    st.sidebar.markdown("- **Primary Detector:** `Hybrid Ensemble`")
    st.sidebar.markdown("- **Algorithm Suite:** `RF + IF + STC + K-Means`")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("🎓 **Academic Final-Year Project**\nCollege Data Mining & Machine Learning\nDesigned for Ethical Security Analytics")

    # Route Page Execution
    if selected_page == "📂 Dataset Upload":
        render_dataset_upload()
    elif selected_page == "🏠 Dashboard":
        render_dashboard()
    elif selected_page == "🔍 Transaction Prediction":
        render_transaction_prediction()
    elif selected_page == "📈 Fraud Analytics":
        render_fraud_analytics()
    elif selected_page == "⚙️ Model Training":
        render_model_training()
    elif selected_page == "📜 Transaction History":
        render_transaction_history()

if __name__ == "__main__":
    main()
