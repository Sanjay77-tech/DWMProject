"""
Clustering Module for UPI Scam and Fraud Detection
Implements K-Means clustering and PCA behavioral projection
to discover transaction behavior groups and isolate high-risk anomaly clusters.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def perform_kmeans_clustering(
    X: np.ndarray, 
    n_clusters: int = 4, 
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
    """
    Fits K-Means algorithm to partition transactions into distinct behavioral clusters.
    Calculates cluster summary profiles including size and distance to centroids.
    """
    kmeans = KMeans(
        n_clusters=n_clusters, 
        init='k-means++', 
        n_init=10, 
        random_state=random_state
    )
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate distance to assigned cluster centroid
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X - centroids[cluster_labels], axis=1)
    
    # Compute summary per cluster
    summary_list = []
    for c_id in range(n_clusters):
        c_mask = (cluster_labels == c_id)
        c_size = int(np.sum(c_mask))
        c_dist_mean = float(np.mean(distances[c_mask])) if c_size > 0 else 0.0
        c_dist_max = float(np.max(distances[c_mask])) if c_size > 0 else 0.0
        
        summary_list.append({
            'cluster_id': c_id,
            'size': c_size,
            'percentage': round((c_size / len(X)) * 100, 2),
            'mean_distance_to_center': round(c_dist_mean, 3),
            'max_distance': round(c_dist_max, 3)
        })
        
    cluster_summary_df = pd.DataFrame(summary_list)
    return kmeans, cluster_labels, cluster_summary_df

def compute_pca_projection(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduces feature dimensionality to 2D using Principal Component Analysis
    for visual inspection of normal versus suspicious clusters.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca

def identify_risk_clusters(
    cluster_labels: np.ndarray, 
    y_true_or_anomaly: np.ndarray
) -> pd.DataFrame:
    """
    Analyzes the composition of each cluster to determine which clusters
    have elevated rates of anomalies or fraud.
    """
    df_analysis = pd.DataFrame({
        'cluster': cluster_labels,
        'flagged': y_true_or_anomaly
    })
    
    grouped = df_analysis.groupby('cluster').agg(
        total_tx=('flagged', 'count'),
        flagged_count=('flagged', 'sum'),
        risk_rate=('flagged', lambda x: np.round(np.mean(x) * 100, 2))
    ).reset_index()
    
    grouped['cluster_category'] = grouped['risk_rate'].apply(
        lambda r: 'High-Risk Cluster' if r > 40 else ('Moderate-Risk Cluster' if r > 15 else 'Typical Normal Behavior')
    )
    
    return grouped
