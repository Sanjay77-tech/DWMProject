"""
Transaction Relationship Graph Analytics Module for UPI Fraud Detection
Constructs directed transaction graphs using NetworkX to identify mule account hubs,
unusual high fan-in / fan-out networks, and cyclical transaction flows.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt

def build_transaction_network(df: pd.DataFrame, max_edges: int = 150) -> nx.DiGraph:
    """
    Constructs a directed graph G=(V, E) where:
    - Nodes V represent User and Merchant/Mule UPI IDs
    - Edges E represent financial transfer flows with aggregated amount and count
    """
    G = nx.DiGraph()
    
    if 'sender_id' not in df.columns or 'receiver_id' not in df.columns:
        return G
        
    # Take most recent or high-risk transactions if df is large
    if 'is_fraud' in df.columns:
        fraud_df = df[df['is_fraud'] == 1]
        normal_sample = df[df['is_fraud'] == 0].tail(max(20, max_edges - len(fraud_df)))
        sub_df = pd.concat([fraud_df, normal_sample]).head(max_edges)
    else:
        sub_df = df.tail(max_edges)
        
    for _, row in sub_df.iterrows():
        u = str(row['sender_id'])
        v = str(row['receiver_id'])
        amt = float(row.get('amount', 0))
        is_fraud = int(row.get('is_fraud', 0))
        
        # Add nodes with attributes
        if not G.has_node(u):
            G.add_node(u, node_type='sender', fraud_involvement=is_fraud)
        else:
            if is_fraud:
                G.nodes[u]['fraud_involvement'] = 1
                
        if not G.has_node(v):
            G.add_node(v, node_type='receiver', fraud_involvement=is_fraud)
        else:
            if is_fraud:
                G.nodes[v]['fraud_involvement'] = 1
                
        # Add or update edge
        if G.has_edge(u, v):
            G[u][v]['weight'] += amt
            G[u][v]['tx_count'] += 1
            if is_fraud:
                G[u][v]['has_fraud'] = 1
        else:
            G.add_edge(u, v, weight=amt, tx_count=1, has_fraud=is_fraud)
            
    return G

def analyze_network_nodes(G: nx.DiGraph) -> pd.DataFrame:
    """
    Computes graph centrality and hub statistics:
    - In-Degree (Fan-in): Number of distinct senders depositing money (potential mule accounts)
    - Out-Degree (Fan-out): Number of distinct recipients
    - Degree Centrality: Topological prominence in the payment network
    """
    if len(G.nodes) == 0:
        return pd.DataFrame(columns=['upi_id', 'in_degree', 'out_degree', 'degree_centrality', 'role_flag'])
        
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg_centrality = nx.degree_centrality(G)
    
    rows = []
    for node in G.nodes:
        i_deg = in_deg.get(node, 0)
        o_deg = out_deg.get(node, 0)
        cent = deg_centrality.get(node, 0.0)
        fraud_inv = G.nodes[node].get('fraud_involvement', 0)
        
        # Identify suspicious structural roles
        if i_deg >= 4 and o_deg <= 1:
            role = 'Suspected Mule Hub (High Fan-In)'
        elif o_deg >= 5:
            role = 'High Fan-Out Distributor'
        elif fraud_inv == 1:
            role = 'Flagged Participant'
        else:
            role = 'Standard Node'
            
        rows.append({
            'upi_id': node,
            'in_degree': i_deg,
            'out_degree': o_deg,
            'degree_centrality': round(cent, 4),
            'fraud_involved': fraud_inv,
            'network_role': role
        })
        
    df_nodes = pd.DataFrame(rows).sort_values(by=['in_degree', 'degree_centrality'], ascending=False).reset_index(drop=True)
    return df_nodes

def plot_transaction_network(G: nx.DiGraph, figsize: Tuple[int, int] = (10, 6)):
    """
    Renders matplotlib visual of the transaction relationship graph.
    Red nodes indicate fraud-involved entities, blue indicate standard entities.
    """
    if len(G.nodes) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No network graph data available", ha='center', va='center')
        ax.axis('off')
        return fig
        
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    
    pos = nx.spring_layout(G, k=0.35, seed=42)
    
    # Node colors based on fraud involvement
    node_colors = []
    for node in G.nodes():
        if G.nodes[node].get('fraud_involvement', 0) == 1:
            node_colors.append('#dc3545') # Red
        else:
            node_colors.append('#00bcd4') # Cyan
            
    # Edge colors
    edge_colors = ['#dc3545' if G[u][v].get('has_fraud', 0) == 1 else '#555555' for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=280, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.2, alpha=0.6, arrows=True, arrowsize=10, ax=ax)
    
    # Only label top nodes by degree
    deg = dict(G.degree())
    top_nodes = sorted(deg, key=deg.get, reverse=True)[:15]
    labels = {node: node.split('@')[0] for node in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='#ffffff', ax=ax)
    
    ax.set_title("UPI Transaction Relationship Network (Red = Fraud / Suspicious Hub)", color='#ffffff', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    return fig
