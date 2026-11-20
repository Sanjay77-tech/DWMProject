# 🛡️ AI-Based UPI Scam and Fraud Detection Using Semi-Supervised Learning, Anomaly Detection and Transaction Behaviour Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Project Overview & Abstract

With over **14 billion transactions processed monthly**, the Unified Payments Interface (**UPI**) forms the backbone of India's digital payments ecosystem. However, its real-time, irreversible settlement mechanism has made it a primary target for sophisticated financial crimes, including:
- **Remote Access Account Takeovers**
- **High-Velocity Midnight Account Drains**
- **Phishing QR Code and Refund Scams**
- **Social Engineering Mule Account Funnels**

Traditional rule-based fraud detection systems (e.g., static amount limits or blacklisted VPA lists) suffer from high false-positive rates and fail to detect novel fraud patterns. Furthermore, in commercial banking, **less than 15% of transactions ever receive human forensic investigation**, making purely supervised learning models impractical.

This academic-grade Data Mining project develops an end-to-end intelligent payment security system that **learns normal user behavioral profiles** and flags anomalies using a hybrid ensemble of:
1. **Semi-Supervised Learning** (`SelfTrainingClassifier`, `LabelSpreading`, `LabelPropagation`)
2. **One-Class Anomaly Detection** (`Isolation Forest`, `One-Class SVM`)
3. **Supervised Classification** (`Random Forest`, `Gradient Boosting`, `Logistic Regression`)
4. **Unsupervised Behavioral Clustering** (`K-Means` + PCA Projection)
5. **Transaction Relationship Graph Analysis** (`NetworkX` Mule Account Hub Discovery)
6. **Explainable AI (XAI)** providing auditable, human-understandable reasoning for every alert.

---

## 🏗️ 2. System Architecture

```
                  ┌─────────────────────────────────────────┐
                  │    Raw Data / Custom CSV / PaySim       │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │   Automated Schema & Synonym Detection  │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │  Temporal Sorting & Anti-Leakage Split  │
                  │   (Train 70% | Val 15% | Holdout 15%)   │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ Behavioral Feature Engineering Pipeline │
                  │  - Velocity (5m, 30m, 24h Tx Counts)    │
                  │  - User Baseline Deviation (Log & Ratio)│
                  │  - Recipient Novelty & Unusual Hours    │
                  │  - Device & Geographic Discrepancies    │
                  └────────────────────┬────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
            ▼                          ▼                          ▼
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │   One-Class Anomaly  │   │   Semi-Supervised    │   │      Supervised      │
 │  - Isolation Forest  │   │  - Self-Training     │   │  - Random Forest     │
 │  - One-Class SVM     │   │  - Label Spreading   │   │  - Gradient Boosting │
 └──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │        Composite Risk Engine            │
                  │   Calibrated Score [0 - 100 Range]     │
                  │  Low (0-30) | Med (31-60) | High (61+) │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │   Explainable AI (XAI) & Graph Mining   │
                  │  - Top Behavioral Trigger Explanations  │
                  │  - NetworkX Mule Hub Node Detection     │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │   Interactive Streamlit Dashboard (7p)  │
                  └─────────────────────────────────────────┘
```

---

## 🧮 3. Data Mining & Machine Learning Methodologies

### A. Behavioral Feature Engineering (Temporal Anti-Leakage)
To prevent temporal data leakage, all rolling statistics (user historical averages, velocity bursts, recipient familiarity) are computed **strictly using transactions occurring prior to the current transaction timestamp**:

- **Velocity Bursts**:
  $$\text{Count}_{5m}(t) = \sum_{t_i < t} \mathbb{I}(t - t_i \le 5\text{ min})$$
- **Baseline Spending Deviation**:
  $$\Delta_{\text{amount}} = \text{Amount}_t - \mu_{\text{user}}(< t)$$
  $$\text{Ratio}_{\text{amount}} = \frac{\text{Amount}_t}{\mu_{\text{user}}(< t) + 1.0}$$
- **Account Drain Ratio**:
  $$\text{DrainRatio} = \frac{\text{Amount}_t}{\text{OldBalance}_t + 1.0}$$

### B. One-Class Anomaly Detection
- **Isolation Forest**: Isolates anomalous instances by recursively generating random feature partition trees. Because anomalies require fewer splits to isolate, their average path length $h(x)$ is shorter:
  $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
  Where $c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$ is the average path length of unsuccessful searches in a Binary Search Tree.
- **One-Class SVM**: Finds a maximum margin hyperplane separating the normal transaction density from the coordinate origin in an RBF kernel feature space.

### C. Semi-Supervised Learning
- **Self-Training Classifier**: In payment environments, acquiring ground truth labels is expensive. The `SelfTrainingClassifier` takes a partially labelled dataset ($y \in \{0, 1, -1\}$ with 85% marked unlabelled $-1$), trains an initial base estimator, and iteratively assigns pseudo-labels to unlabelled transactions whose predicted class probability exceeds confidence threshold $\tau = 0.75$:
  $$\hat{y}_i = \arg\max_{c} P(Y = c | x_i) \quad \text{if } \max_{c} P(Y = c | x_i) \ge \tau$$

### D. Composite Fraud Risk Score (0 - 100)
To avoid conflating an unusual transaction with confirmed fraud, scores are calibrated into an audited index:
$$\text{Risk Score} = 0.50 \cdot (\text{Supervised Prob} \times 100) + 0.35 \cdot (\text{Anomaly Relative Score}) + 0.15 \cdot (\text{Semi-Supervised Prob} \times 100) + \text{Rule Penalty}$$

| Score Range | Category | Default Action |
| :--- | :--- | :--- |
| **0 – 30** | Low Risk | Allow transaction without friction. |
| **31 – 60** | Medium Risk | Step-up 2FA (SMS OTP or Push Confirmation). |
| **61 – 80** | High Risk | 15-minute temporary security hold + Biometric approval. |
| **81 – 100** | Very High Risk | Freeze transfer immediately; flag for manual forensic audit. |

---

## 📂 4. Project Folder Structure

```
upi_fraud_detection/
├── app.py                     # Main Streamlit Dashboard application entrypoint
├── requirements.txt           # Production dependencies
├── README.md                  # Comprehensive academic documentation
│
├── data/
│   ├── raw/                   # Ingested raw datasets (PaySim & synthetic benchmark)
│   └── processed/             # Cleaned datasets with engineered behavioral features
│
├── models/
│   ├── preprocessing_pipeline.pkl  # Fitted Scikit-learn ColumnTransformer
│   ├── isolation_forest.pkl        # Fitted Isolation Forest anomaly detector
│   ├── one_class_svm.pkl           # Fitted One-Class Support Vector Machine
│   ├── random_forest.pkl           # Fitted Supervised Random Forest Classifier
│   └── self_training.pkl           # Fitted Semi-Supervised Classifier
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Schema inference, fuzzy mapping & synthetic generator
│   ├── preprocessing.py       # Duplicate filtering, scaling & chronological split
│   ├── feature_engineering.py # Temporal velocity & user profiling features
│   ├── anomaly_detection.py   # Isolation Forest & One-Class SVM implementations
│   ├── semi_supervised.py     # Self-Training, Label Spreading & Label Propagation
│   ├── classification.py      # Random Forest, Gradient Boosting & Logistic Regression
│   ├── clustering.py          # K-Means clustering and PCA dimensionality reduction
│   ├── risk_scoring.py        # Calibrated 0-100 risk scoring & action engine
│   ├── explainability.py      # XAI feature attribution & human-readable reasoning
│   ├── graph_analytics.py     # NetworkX transaction network & mule hub discovery
│   ├── pipeline_manager.py    # Pipeline orchestration & live transaction inference
│   └── evaluation.py          # Academic imbalanced metrics (F1, PR-AUC, ROC-AUC)
│
├── views/
│   ├── dataset_upload.py          # Page 1: Initial Ingestion & Forensic Inspection Hub
│   ├── dashboard.py               # Page 2: Executive KPI Security Dashboard
│   ├── transaction_prediction.py  # Page 3: Real-Time Transaction Prediction & XAI
│   ├── fraud_analytics.py         # Page 4: Interactive Forensic Visualizations & Mule Graphs
│   ├── model_training.py          # Page 5: Hyperparameter Tuning, Training & Diagnostics
│   └── transaction_history.py     # Page 6: Audit Ledger & Filterable Transaction History
│
├── reports/                   # Exported forensic audit summaries
└── tests/
    └── test_pipeline.py       # Pytest unit and integration test suite
```

---

## 🚀 5. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.15 installed on Windows, macOS, or Linux.

### Step 1: Clone / Navigate to Project Directory
```powershell
cd c:\Users\Admin\OneDrive\Desktop\DWMProject\upi_fraud_detection
```

### Step 2: Install Dependencies
```powershell
py -m pip install -r requirements.txt
```

### Step 3: Run Automated Test Suite
```powershell
py -m pytest tests/test_pipeline.py -v
```

### Step 4: Launch the Streamlit Dashboard
```powershell
py -m streamlit run app.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 🖥️ 6. Dashboard Interface Walkthrough

1. **Page 1: Home Dashboard**: Displays executive metrics (Total Analyzed, Safe, Suspicious, Fraud Rate, Average Amount), classification donut charts, and live security feeds.
2. **Page 2: Transaction Prediction**: Enter transaction parameters (Amount, UPI IDs, Velocity, Device/Location status) or click one-click scam presets (e.g. *Midnight Drain*, *Phishing QR Scam*) to inspect calibrated risk scores and XAI reasons.
3. **Page 3: Dataset Upload**: Ingest custom CSV logs or PaySim benchmarks. Auto-detects columns, displays data health, runs batch feature engineering, and provides downloadable scored CSVs.
4. **Page 4: Fraud Analytics**: Visualizes hourly vulnerability spikes, log amount distributions, transaction mode vulnerabilities, and the **NetworkX Mule Hub Graph**.
5. **Page 5: Model Training**: Interactively tune hyperparameters (e.g. contamination rate, tree depth, pseudo-label thresholds) and retrain algorithms with live confusion matrices.
6. **Page 6: Transaction History**: Searchable, filterable audit ledger of evaluated transactions with risk tiers and one-click CSV export.
7. **Page 7: Model Evaluation**: Side-by-side benchmark comparison (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR) with ROC and Precision-Recall diagnostic curves.

---

## ⚖️ 7. Security and Responsible Design Statement
- **Synthetic Benchmark Data**: All demonstration data, VPAs, and account numbers are synthetically generated.
- **Privacy & Safety**: The system never asks for or stores UPI PINs, OTPs, or banking credentials.
- **Academic Intent**: Designed strictly as an academic research and defensive data mining demonstration. It does not interface with or intercept live banking payment gateways.
