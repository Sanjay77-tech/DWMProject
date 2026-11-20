"""
Page 5: Interactive Model Training & Hyperparameter Tuning
Allows users to train and tune Supervised, Semi-Supervised, Anomaly Detection,
and Clustering models directly from the UI with real-time performance feedback.
"""

import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_dataset
from src.preprocessing import clean_transaction_data, build_preprocessing_pipeline, split_chronological, prepare_feature_matrices, save_preprocessor
from src.feature_engineering import extract_transaction_features
from src.classification import train_random_forest, train_logistic_regression, train_gradient_boosting, save_classifier
from src.anomaly_detection import train_isolation_forest, train_one_class_svm, save_anomaly_model
from src.semi_supervised import create_semi_supervised_labels, train_self_training_model, train_label_spreading_model, train_label_propagation_model, save_semi_supervised_model
from src.clustering import perform_kmeans_clustering, compute_pca_projection
from src.evaluation import evaluate_classification_model, evaluate_multiple_models
from src.pipeline_manager import get_model_paths

def render_model_training():
    # Header Section
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.5rem;">
        <h1 style="color: #f8fafc; margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;">
            ⚙️ Interactive Model Training & Optimization
        </h1>
        <p style="color: #94a3b8; margin-top: 0.35rem; font-size: 0.95rem;">
            Train, tune hyperparameters, and benchmark Supervised, Semi-Supervised, Anomaly Detection, and Clustering algorithms.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check session state for dataset
    if 'current_df' not in st.session_state or st.session_state['current_df'] is None or len(st.session_state['current_df']) == 0:
        st.markdown("""
        <div style="background-color: #111927; border: 1px solid #1e293b; border-radius: 10px; padding: 2.2rem 1.5rem; text-align: center; margin: 1.5rem 0;">
            <div style="font-size: 2.4rem; margin-bottom: 0.5rem;">⚙️</div>
            <h3 style="color: #f8fafc; margin-bottom: 0.5rem; font-size: 1.35rem; font-weight: 700;">No Dataset Currently Ingested</h3>
            <p style="color: #94a3b8; max-width: 580px; margin: 0 auto 1.5rem auto; font-size: 0.95rem; line-height: 1.5;">
                Model training and validation diagnostics require training transactions.
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
    paths = get_model_paths()
    
    # 1. Select Model Family & Algorithm
    st.markdown("##### 1. 🧠 Machine Learning Paradigm & Algorithm Selection")
    
    family = st.selectbox(
        "Select Machine Learning Paradigm",
        [
            "Supervised Classification (Labelled)",
            "Semi-Supervised Learning (Partial Labels + Pseudo-Labelling)",
            "One-Class Anomaly Detection (Trained on Normal Baselines)",
            "Unsupervised Clustering (Behavior Grouping)"
        ]
    )
    
    col_algo, col_params = st.columns([1.1, 1.9], gap="large")
    
    with col_algo:
        if family == "Supervised Classification (Labelled)":
            algo = st.radio("Choose Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        elif family == "Semi-Supervised Learning (Partial Labels + Pseudo-Labelling)":
            algo = st.radio("Choose Model", ["Self-Training (Random Forest base)", "Label Spreading", "Label Propagation"])
        elif family == "One-Class Anomaly Detection (Trained on Normal Baselines)":
            algo = st.radio("Choose Model", ["Isolation Forest", "One-Class SVM"])
        else:
            algo = st.radio("Choose Model", ["K-Means Clustering"])
            
    with col_params:
        st.markdown(f"**Hyperparameters for {algo}:**")
        
        if algo == "Random Forest":
            n_est = st.slider("Number of Trees (n_estimators)", 50, 300, 120, 10)
            max_d = st.slider("Maximum Depth", 3, 25, 12)
            min_split = st.slider("Min Samples Split", 2, 10, 4)
            
        elif algo == "Gradient Boosting":
            n_est = st.slider("Number of Estimators", 30, 200, 80, 10)
            lr = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            max_d = st.slider("Max Depth", 2, 8, 4)
            
        elif algo == "Logistic Regression":
            c_val = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.1)
            max_iter = st.slider("Max Iterations", 200, 2000, 800, 100)
            
        elif algo == "Isolation Forest":
            contamination = st.slider("Expected Contamination Rate", 0.01, 0.15, 0.045, 0.005)
            n_est = st.slider("Number of Estimators", 50, 250, 120, 10)
            
        elif algo == "One-Class SVM":
            nu = st.slider("Upper Bound on Training Errors (nu)", 0.01, 0.20, 0.05, 0.01)
            kernel = st.selectbox("Kernel Function", ["rbf", "linear", "poly"])
            
        elif algo == "Self-Training (Random Forest base)":
            st_thresh = st.slider("Pseudo-Label Confidence Threshold", 0.55, 0.95, 0.75, 0.05)
            lab_ratio = st.slider("Initial Labelled Ratio (%)", 5, 50, 15, 5) / 100.0
            st_iter = st.slider("Maximum Iterations", 2, 15, 6)
            
        elif algo in ["Label Spreading", "Label Propagation"]:
            lab_ratio = st.slider("Initial Labelled Ratio (%)", 5, 50, 15, 5) / 100.0
            gamma = st.slider("RBF Kernel Gamma", 1.0, 50.0, 20.0, 5.0)
            if algo == "Label Spreading":
                alpha = st.slider("Clamping Parameter (Alpha)", 0.05, 0.95, 0.20, 0.05)
                
        elif algo == "K-Means Clustering":
            k_val = st.slider("Number of Clusters (K)", 2, 8, 4)

    st.markdown("---")
    
    # 2. Training Trigger
    if st.button("🚀 Train & Optimize Selected Model", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Preparing chronological train-val-test split & feature pipeline...")
        progress_bar.progress(20)
        start_time = time.time()
        
        # Pipeline execution
        df_clean = clean_transaction_data(df)
        df_feat = extract_transaction_features(df_clean)
        df_train, df_val, df_test = split_chronological(df_feat, 0.70, 0.15, 0.15)
        
        preprocessor = build_preprocessing_pipeline()
        X_train, y_train, preprocessor, fcols = prepare_feature_matrices(df_train, preprocessor, fit=True)
        X_test, y_test, _, _ = prepare_feature_matrices(df_test, preprocessor, fit=False)
        
        save_preprocessor(preprocessor, paths['preprocessor'])
        
        progress_bar.progress(50)
        status_text.text(f"Fitting {algo} algorithm...")
        
        # Model Fitting
        if algo == "Random Forest":
            model = train_random_forest(X_train, y_train, n_estimators=n_est, max_depth=max_d, min_samples_split=min_split)
            save_classifier(model, paths['random_forest'])
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        elif algo == "Gradient Boosting":
            model = train_gradient_boosting(X_train, y_train, n_estimators=n_est, learning_rate=lr, max_depth=max_d)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        elif algo == "Logistic Regression":
            model = train_logistic_regression(X_train, y_train, C=c_val, max_iter=max_iter)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        elif algo == "Isolation Forest":
            normal_mask = (y_train == 0) if y_train is not None else np.ones(len(X_train), dtype=bool)
            model = train_isolation_forest(X_train[normal_mask], contamination=contamination, n_estimators=n_est)
            save_anomaly_model(model, paths['isolation_forest'])
            preds = model.predict(X_test)
            y_pred = np.where(preds == -1, 1, 0)
            y_prob = 1.0 / (1.0 + np.exp(8.0 * (model.decision_function(X_test) + 0.05)))
            
        elif algo == "One-Class SVM":
            normal_mask = (y_train == 0) if y_train is not None else np.ones(len(X_train), dtype=bool)
            model = train_one_class_svm(X_train[normal_mask], nu=nu, kernel=kernel)
            save_anomaly_model(model, paths['one_class_svm'])
            preds = model.predict(X_test)
            y_pred = np.where(preds == -1, 1, 0)
            y_prob = 1.0 / (1.0 + np.exp(1.5 * model.decision_function(X_test)))
            
        elif algo == "Self-Training (Random Forest base)":
            y_semi = create_semi_supervised_labels(y_train, labeled_ratio=lab_ratio)
            model, st_stats = train_self_training_model(X_train, y_semi, threshold=st_thresh, max_iter=st_iter)
            save_semi_supervised_model(model, paths['self_training'])
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        elif algo == "Label Spreading":
            y_semi = create_semi_supervised_labels(y_train, labeled_ratio=lab_ratio)
            model, st_stats = train_label_spreading_model(X_train, y_semi, gamma=gamma, alpha=alpha)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        elif algo == "Label Propagation":
            y_semi = create_semi_supervised_labels(y_train, labeled_ratio=lab_ratio)
            model, st_stats = train_label_propagation_model(X_train, y_semi, gamma=gamma)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        elif algo == "K-Means Clustering":
            kmeans, clusters, summary_df = perform_kmeans_clustering(X_train, n_clusters=k_val)
            y_pred = None
            y_prob = None
            
        progress_bar.progress(100)
        elapsed = time.time() - start_time
        status_text.text(f"Training completed in {elapsed:.2f} seconds.")
        
        # Display Results
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Training & Validation Diagnostics")
        
        if y_pred is not None and y_test is not None:
            metrics = evaluate_classification_model(y_test, y_pred, y_prob, model_name=algo)
            
            # Metrics cards
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            mc2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
            mc3.metric("Recall (Sensitivity)", f"{metrics['recall'] * 100:.2f}%")
            mc4.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            mc5.metric("ROC-AUC", f"{metrics['roc_auc']}")
            
            # Confusion Matrix
            col_cm, col_rep = st.columns(2, gap="large")
            with col_cm:
                st.markdown("**Confusion Matrix on Holdout Test Partition:**")
                raw_cm = np.array(metrics['confusion_matrix'])
                if raw_cm.shape == (2, 2):
                    cm = raw_cm
                else:
                    cm = np.zeros((2, 2), dtype=int)
                    cm[:min(2, raw_cm.shape[0]), :min(2, raw_cm.shape[1])] = raw_cm[:min(2, raw_cm.shape[0]), :min(2, raw_cm.shape[1])]
                fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8), facecolor='#0f172a')
                ax_cm.set_facecolor('#0f172a')
                im = ax_cm.imshow(cm, cmap='Blues')
                ax_cm.set_xticks([0, 1])
                ax_cm.set_yticks([0, 1])
                ax_cm.set_xticklabels(['Normal (0)', 'Fraud (1)'], color='#cbd5e1', fontsize=9)
                ax_cm.set_yticklabels(['Normal (0)', 'Fraud (1)'], color='#cbd5e1', fontsize=9)
                ax_cm.set_xlabel("Predicted Label", color='#94a3b8', fontsize=9.5)
                ax_cm.set_ylabel("Actual Ground Truth", color='#94a3b8', fontsize=9.5)
                ax_cm.tick_params(colors='#94a3b8')
                for i in range(2):
                    for j in range(2):
                        ax_cm.text(j, i, f"{cm[i, j]:,}", ha='center', va='center', color='#f59e0b', fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_cm)
                
            with col_rep:
                st.markdown("**Detailed Risk Performance Metrics:**")
                st.write(f"- **False Positive Rate (Legitimate flagged):** `{metrics['false_positive_rate'] * 100:.2f}%`")
                st.write(f"- **False Negative Rate (Missed Frauds):** `{metrics['false_negative_rate'] * 100:.2f}%`")
                st.write(f"- **True Positives Caught:** `{metrics['true_positives']} fraud incidents`")
                st.write(f"- **False Negatives Escaped:** `{metrics['false_negatives']} fraud incidents`")
                
                if "Self-Training" in algo:
                    st.markdown("**Semi-Supervised Pseudo-Labeling Stats:**")
                    st.write(f"- Total Unlabelled Pool: `{st_stats['initial_unlabelled']:,}`")
                    st.write(f"- Pseudo-Labels Assigned Confidently: `{st_stats['pseudo_labels_assigned']:,}`")
                    st.write(f"- Pseudo-Fraud Discovered: `{st_stats['pseudo_fraud_count']}`")
                    st.write(f"- Pseudo-Normal Certified: `{st_stats['pseudo_normal_count']:,}`")
                    
        elif algo == "K-Means Clustering":
            st.markdown("**Cluster Summary Profiles:**")
            st.dataframe(summary_df, use_container_width=True)
            
            # PCA projection
            X_pca = compute_pca_projection(X_train)
            fig_pca, ax_pca = plt.subplots(figsize=(6, 4), facecolor='#0f172a')
            ax_pca.set_facecolor('#0f172a')
            scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=15)
            ax_pca.set_title("K-Means Behavioral Clusters (2D PCA Projection)", color='#f8fafc', fontsize=10.5, weight='600')
            ax_pca.tick_params(colors='#94a3b8')
            ax_pca.axis('off')
            plt.tight_layout()
            st.pyplot(fig_pca)
            
        st.success(f"🎉 Model artifact '{algo}' successfully trained and persisted to models/ directory!")

    # Optional Multi-Model Benchmark Comparison Section
    st.markdown("---")
    with st.expander("📊 Multi-Model Benchmark Suite (Compare All Algorithms Side-by-Side)", expanded=False):
        st.caption("Compares Random Forest, Gradient Boosting, Self-Training, Logistic Regression, Isolation Forest, and One-Class SVM on the unseen holdout test split.")
        if st.button("⚡ Run Multi-Model Comparison Table", key="btn_compare_all"):
            with st.spinner("Evaluating all algorithms on chronological holdout test partition..."):
                df_clean_bench = clean_transaction_data(df)
                df_feat_bench = extract_transaction_features(df_clean_bench)
                df_tr, df_v, df_te = split_chronological(df_feat_bench, 0.70, 0.15, 0.15)
                
                prep_b = build_preprocessing_pipeline()
                X_tr, y_tr, prep_b, _ = prepare_feature_matrices(df_tr, prep_b, fit=True)
                X_te, y_te, _, _ = prepare_feature_matrices(df_te, prep_b, fit=False)
                
                if y_tr is not None and y_te is not None:
                    norm_mask = (y_tr == 0)
                    X_tr_norm = X_tr[norm_mask]
                    
                    rf_b = train_random_forest(X_tr, y_tr, n_estimators=120, max_depth=12)
                    gb_b = train_gradient_boosting(X_tr, y_tr, n_estimators=80, learning_rate=0.1, max_depth=4)
                    lr_b = train_logistic_regression(X_tr, y_tr, C=1.0)
                    iso_b = train_isolation_forest(X_tr_norm, contamination=0.045, n_estimators=100)
                    oc_b = train_one_class_svm(X_tr_norm, nu=0.05)
                    
                    y_semi_b = create_semi_supervised_labels(y_tr, labeled_ratio=0.15)
                    st_b, _ = train_self_training_model(X_tr, y_semi_b, threshold=0.75)
                    
                    models_suite = {
                        'Random Forest (Supervised)': rf_b,
                        'Gradient Boosting (Supervised)': gb_b,
                        'Self-Training (Semi-Supervised 15% Labels)': st_b,
                        'Logistic Regression (Supervised)': lr_b,
                        'Isolation Forest (One-Class Anomaly)': iso_b,
                        'One-Class SVM (Anomaly)': oc_b
                    }
                    
                    comp_df = evaluate_multiple_models(models_suite, X_te, y_te)
                    st.dataframe(comp_df.style.highlight_max(subset=['F1-Score', 'Precision', 'Recall'], color='#1e3a8a'), use_container_width=True)
                else:
                    st.warning("Ground-truth labels required for multi-model benchmark evaluation.")

if __name__ == "__main__":
    render_model_training()
