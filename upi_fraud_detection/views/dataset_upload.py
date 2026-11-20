"""
Page 1: Dataset Upload & Automated Forensic Inspection
Initial Application Entrypoint: Drag-and-drop CSV ingestion, automatic structural inspection,
schema synonym detection, missing value auditing, and pipeline activation.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_dataset, detect_column_mapping
from src.preprocessing import clean_transaction_data, build_preprocessing_pipeline, prepare_feature_matrices
from src.feature_engineering import extract_transaction_features
from src.anomaly_detection import predict_anomaly, load_anomaly_model
from src.classification import predict_supervised, load_classifier
from src.risk_scoring import compute_risk_score
from src.pipeline_manager import get_model_paths, ensure_baseline_models_exist

def process_and_activate_dataframe(df_loaded, meta, source_label, file_id=None):
    """
    Cleans, extracts behavioral features, scores transactions with ensemble models,
    and stores the active dataset in Streamlit session state.
    """
    ensure_baseline_models_exist()
    paths = get_model_paths()
    
    df_clean = clean_transaction_data(df_loaded)
    df_feat = extract_transaction_features(df_clean)
    
    iso = load_anomaly_model(paths['isolation_forest'])
    rf = load_classifier(paths['random_forest'])
    from src.preprocessing import load_preprocessor
    preprocessor = load_preprocessor(paths['preprocessor'])
    
    X_scaled, _, _, _ = prepare_feature_matrices(df_feat, preprocessor, fit=False)
    is_ano, ano_scores = predict_anomaly(iso, X_scaled, 'isolation_forest')
    _, rf_probs = predict_supervised(rf, X_scaled)
    
    df_feat['anomaly_flag'] = is_ano
    df_feat['anomaly_score'] = ano_scores
    df_feat['supervised_fraud_prob'] = np.round(rf_probs * 100, 2)
    risk_scores = np.round((rf_probs * 50.0) + (ano_scores * 0.50), 1)
    df_feat['risk_score'] = risk_scores
    
    st.session_state['current_df'] = df_feat
    st.session_state['meta'] = meta
    st.session_state['dataset_source_label'] = source_label
    if file_id:
        st.session_state['active_uploaded_file'] = file_id

def render_dataset_upload():
    # Header Section
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.5rem;">
        <h1 style="color: #f8fafc; margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;">
            📂 Transaction Dataset Ingestion
        </h1>
        <p style="color: #94a3b8; margin-top: 0.35rem; font-size: 0.95rem;">
            Upload transaction logs to inspect data quality, extract behavioral velocity features, and activate the security monitoring pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    has_dataset = 'current_df' in st.session_state and st.session_state['current_df'] is not None and len(st.session_state['current_df']) > 0
    
    # Active Dataset Notification Banner
    if has_dataset:
        df_active = st.session_state['current_df']
        source_lbl = st.session_state.get('dataset_source_label', 'Active Ingested Dataset')
        meta = st.session_state.get('meta', {})
        fraud_c = meta.get('fraud_count', int(df_active['is_fraud'].sum()) if 'is_fraud' in df_active.columns else 0)
        fraud_p = meta.get('fraud_rate', (fraud_c / len(df_active) * 100) if len(df_active) > 0 and 'is_fraud' in df_active.columns else 0.0)
        
        st.markdown(f"""
        <div style="background-color: #0f231d; border: 1px solid #059669; border-radius: 8px; padding: 1rem 1.4rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #10b981; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em;">
                        ● Dataset Active &amp; Operational
                    </span>
                    <span style="color: #64748b;">|</span>
                    <span style="color: #f8fafc; font-weight: 600; font-size: 0.95rem;">{source_lbl}</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">
                    Records: <strong>{len(df_active):,}</strong> &nbsp;•&nbsp; 
                    Attributes: <strong>{len(df_active.columns)} features</strong> &nbsp;•&nbsp; 
                    Fraud Cases: <strong>{fraud_c:,} ({fraud_p:.2f}%)</strong>
                </div>
            </div>
            <div style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; padding: 0.4rem 0.9rem; border-radius: 6px; font-size: 0.82rem; font-weight: 600;">
                All Dashboard Pages Unlocked
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #111927; border: 1px dashed #334155; border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 1.5rem;">
            <div style="color: #f59e0b; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.2rem;">
                ⚠️ Ingestion Required to Activate Analytics
            </div>
            <div style="color: #94a3b8; font-size: 0.88rem; line-height: 1.4;">
                No transaction dataset is currently loaded in memory. Upload your custom payment logs or download one of the sample templates below to activate the Dashboard, Forensic Analytics, and Model Training modules.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Ingestion Controls
    col_uploader, col_help = st.columns([1.8, 1.2], gap="large")
    
    with col_uploader:
        st.markdown("##### 📤 Drag & Drop Transaction CSV")
        st.caption("Upload raw UPI logs, PaySim extracts, or bank payment records in CSV format.")
        
        uploaded_file = st.file_uploader(
            "Select CSV transaction file", 
            type=['csv'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Check if this file is newly uploaded
            if st.session_state.get('active_uploaded_file') != uploaded_file.name:
                try:
                    with st.spinner("Parsing CSV structure, verifying column schema, and calculating baseline statistics..."):
                        df_loaded, meta = load_dataset(uploaded_file=uploaded_file)
                        
                        if len(df_loaded) == 0:
                            st.error("Uploaded CSV file is empty. Please upload a valid dataset with transaction rows.")
                        elif 'amount' not in df_loaded.columns and 'step' not in df_loaded.columns:
                            st.error("Uploaded dataset is missing essential financial columns (such as amount or step). Please review schema requirements.")
                        else:
                            source_label = f"Custom: {uploaded_file.name} ({len(df_loaded):,} records)"
                            process_and_activate_dataframe(df_loaded, meta, source_label, file_id=uploaded_file.name)
                            st.success(f"Successfully ingested and processed '{uploaded_file.name}' ({len(df_loaded):,} records)!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error processing dataset: {str(e)}")

    with col_help:
        st.markdown("##### 📋 Supported Dataset Format")
        st.markdown("""
        The engine automatically detects column synonyms using fuzzy schema mapping:
        - **Transaction Amount:** `amount`, `amt`, `value`
        - **Sender Identity:** `nameOrig`, `sender_id`, `from_account`
        - **Receiver Identity:** `nameDest`, `receiver_id`, `to_account`
        - **Account Balances:** `oldbalanceOrg`, `newbalanceOrig`
        - **Temporal Attribute:** `step`, `timestamp`, `datetime`
        - **Fraud Target Label (Optional):** `isFraud`, `is_fraud`, `class`
        """)

    # Direct Download buttons for Sample Datasets
    st.markdown("---")
    st.markdown("##### 📥 Sample Datasets for Testing & Demonstration")
    dcol1, dcol2, dcol3 = st.columns([1, 1, 1])
    
    with dcol1:
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'custom_sample_upload.csv')
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                sample_bytes = f.read()
            st.download_button(
                label="📥 Download Custom Sample (30 rows)",
                data=sample_bytes,
                file_name="custom_sample_upload.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with dcol2:
        raw_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'upi_benchmark_transactions.csv')
        if os.path.exists(raw_csv_path):
            with open(raw_csv_path, 'rb') as f:
                raw_bytes = f.read()
            st.download_button(
                label="📥 Download Benchmark CSV (7,500 rows)",
                data=raw_bytes,
                file_name="upi_benchmark_transactions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with dcol3:
        if st.button("⚡ Quick-Load Curated Benchmark (7,500 rows)", use_container_width=True):
            with st.spinner("Ingesting curated 7,500-record benchmark dataset..."):
                df_b, meta_b = load_dataset()
                source_label = f"Curated UPI Benchmark ({len(df_b):,} records)"
                process_and_activate_dataframe(df_b, meta_b, source_label, file_id=None)
            st.success("Benchmark dataset loaded successfully!")
            st.rerun()

    # Detailed Dataset Inspection Card
    if has_dataset:
        st.markdown("---")
        st.subheader("🔍 Automated Dataset Inspection & Quality Audit")
        
        df_active = st.session_state['current_df']
        meta = st.session_state.get('meta', {})
        
        # 8 Inspection Metrics Required by Spec:
        # 1. Filename, 2. Total Records, 3. Total Columns, 4. Duplicate Records,
        # 5. Missing Values, 6. Fraud Label Availability, 7. Data Types, 8. Column Names
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(
                label="Dataset Filename",
                value=meta.get('filename', 'uploaded_dataset.csv')[:22]
            )
        with kpi2:
            st.metric(
                label="Total Records",
                value=f"{meta.get('total_rows', len(df_active)):,}"
            )
        with kpi3:
            st.metric(
                label="Total Columns",
                value=f"{meta.get('total_columns', len(df_active.columns))}"
            )
        with kpi4:
            dup_count = meta.get('duplicate_rows', 0)
            st.metric(
                label="Duplicate Records",
                value=f"{dup_count:,}",
                delta="Clean (0 duplicates)" if dup_count == 0 else f"{dup_count} duplicates found",
                delta_color="normal" if dup_count == 0 else "inverse"
            )
            
        kpi5, kpi6, kpi7, kpi8 = st.columns(4)
        with kpi5:
            missing_total = meta.get('missing_values_total', int(df_active.isnull().sum().sum()))
            st.metric(
                label="Total Missing Values",
                value=f"{missing_total:,}",
                delta="100% Complete" if missing_total == 0 else f"{missing_total} null entries",
                delta_color="normal" if missing_total == 0 else "inverse"
            )
        with kpi6:
            has_labels = meta.get('has_labels', 'is_fraud' in df_active.columns)
            st.metric(
                label="Fraud Label Available",
                value="Yes (Supervised)" if has_labels else "No (Anomaly Only)",
                delta=f"{meta.get('fraud_count', 0):,} fraud incidents" if has_labels else "Unsupervised",
                delta_color="normal" if has_labels else "off"
            )
        with kpi7:
            num_numeric = sum(1 for dt in meta.get('dtypes', {}).values() if 'int' in dt.lower() or 'float' in dt.lower())
            st.metric(
                label="Numeric Features",
                value=f"{num_numeric} columns"
            )
        with kpi8:
            num_cat = len(meta.get('dtypes', {})) - num_numeric
            st.metric(
                label="Categorical / String",
                value=f"{max(0, num_cat)} columns"
            )

        # Tabbed Inspection Details
        st.markdown("<br>", unsafe_allow_html=True)
        tab_schema, tab_audit, tab_preview, tab_export = st.tabs([
            "📋 Column Names & Schema Mapping",
            "🧹 Data Types & Missing Values Audit",
            "📄 Raw & Processed Preview",
            "📥 Export Scored Dataset"
        ])
        
        with tab_schema:
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Column Schema Mapping:**")
                mapped = meta.get('mapped_columns', {})
                if mapped:
                    mapping_df = pd.DataFrame(
                        list(mapped.items()),
                        columns=['Uploaded Column Name', 'Internal Mapped Attribute']
                    )
                    st.dataframe(mapping_df, use_container_width=True)
                else:
                    st.info("Dataset adheres directly to standardized schema.")
            with sc2:
                st.markdown("**Engineered Feature Inventory:**")
                st.markdown("""
                The forensic pipeline automatically extracts 18 behavioral features:
                - **Velocity Metrics:** `tx_count_5m`, `tx_count_30m`, `tx_count_24h`
                - **Inter-Transaction Spacing:** `time_since_prev_tx_min`
                - **Spending Deviation:** `amount_ratio_avg`, `amount_to_balance_ratio`
                - **Temporal Anomaly:** `is_unusual_hour` (midnight to 5 AM)
                - **Account Drain:** `account_drain_ratio`
                - **Beneficiary Novelty:** `is_new_recipient`
                """)

        with tab_audit:
            st.markdown("**Comprehensive Column Quality Breakdown:**")
            dtypes_dict = meta.get('dtypes', {c: str(df_active[c].dtype) for c in df_active.columns})
            null_dict = meta.get('missing_per_column', df_active.isnull().sum().to_dict())
            
            audit_records = []
            for col, dt in dtypes_dict.items():
                m_cnt = null_dict.get(col, 0)
                tot = len(df_active) if len(df_active) > 0 else 1
                audit_records.append({
                    'Column Name': col,
                    'Data Type': dt,
                    'Missing Count': m_cnt,
                    'Missing Percentage': f"{(m_cnt / tot * 100):.2f}%",
                    'Status': '✅ Valid' if m_cnt == 0 else '⚠️ Contains Nulls'
                })
            st.dataframe(pd.DataFrame(audit_records), use_container_width=True)

        with tab_preview:
            st.markdown(f"**First 10 Rows with Forensic Risk Scores:**")
            display_cols = [c for c in ['transaction_id', 'timestamp', 'sender_id', 'receiver_id', 'amount', 'transaction_type', 'risk_score', 'supervised_fraud_prob', 'anomaly_score', 'is_fraud'] if c in df_active.columns]
            st.dataframe(df_active[display_cols].head(10), use_container_width=True)

        with tab_export:
            st.markdown("**Download Fully Enriched Dataset:**")
            st.caption("Exports the transaction log augmented with all 18 engineered features, Isolation Forest anomaly index, and Random Forest risk probabilities.")
            csv_bytes = df_active.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Dataset (CSV)",
                data=csv_bytes,
                file_name="upi_fraud_scored_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("""
        <div style="background-color: #0f172a; border-left: 4px solid #10b981; padding: 1rem 1.4rem; border-radius: 4px;">
            <strong style="color: #10b981;">🚀 Next Step:</strong>
            <span style="color: #cbd5e1; margin-left: 0.5rem;">
                Your dataset is active. Open the sidebar navigation to explore <strong>🏠 Dashboard</strong>, test single transactions in <strong>🔍 Transaction Prediction</strong>, or examine money mule networks in <strong>📈 Fraud Analytics</strong>.
            </span>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_dataset_upload()
