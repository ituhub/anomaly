"""
Real-Time Anomaly Detection Platform
=====================================
A comprehensive platform for detecting anomalies in financial data streams,
model drift, and market regime changes. Built on advanced ML techniques from
the enhanced trading system.

Features:
- Real-time data stream monitoring
- Multi-model anomaly detection (Statistical, ML, Deep Learning)
- Model drift detection with PSI and KS tests
- Market regime detection using Gaussian Mixture Models
- Feature importance and explainability
- Interactive dashboards with alerts
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import deque, defaultdict
import time
import json
from io import BytesIO
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings("ignore")

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Real-Time Anomaly Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with vibrant accents and enhanced sidebar
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a25;
        --bg-sidebar: #0d0d14;
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --accent-red: #ff3366;
        --accent-green: #00ff88;
        --accent-blue: #00aaff;
        --accent-yellow: #ffaa00;
        --accent-purple: #aa66ff;
        --accent-cyan: #00e5ff;
        --gradient-start: #ff3366;
        --gradient-end: #00aaff;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    /* ============================================
       ENHANCED SIDEBAR STYLES
       ============================================ */
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d14 0%, #151520 50%, #0d0d14 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, rgba(255,51,102,0.15), rgba(0,170,255,0.15));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        text-align: center;
    }
    
    .sidebar-header h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-red), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    
    .sidebar-header p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-secondary);
        margin: 0;
    }
    
    .sidebar-section {
        background: linear-gradient(145deg, rgba(26,26,37,0.8), rgba(18,18,26,0.9));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1rem;
        color: var(--accent-blue);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    
    .sidebar-metric-card {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.6rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .sidebar-metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0,170,255,0.3);
    }
    
    .sidebar-metric-card .label {
        font-size: 0.6rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    
    .sidebar-metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 0.2rem;
    }
    
    .sidebar-status-bar {
        background: linear-gradient(90deg, rgba(0,255,136,0.1), rgba(0,170,255,0.1));
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 10px;
        padding: 0.75rem;
        margin-top: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .sidebar-status-bar .status-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse 2s infinite;
        box-shadow: 0 0 10px var(--accent-green);
    }
    
    .sidebar-status-bar .status-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent-green);
    }
    
    .sidebar-info-card {
        background: linear-gradient(135deg, rgba(170,102,255,0.1), rgba(0,170,255,0.1));
        border: 1px solid rgba(170,102,255,0.2);
        border-radius: 10px;
        padding: 0.75rem;
        margin-top: 0.5rem;
        transition: transform 0.2s ease;
    }
    
    .sidebar-info-card:hover {
        transform: translateY(-2px);
    }
    
    .sidebar-info-card .info-title {
        font-size: 0.65rem;
        color: var(--accent-purple);
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        margin-bottom: 0.3rem;
    }
    
    .sidebar-info-card .info-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-primary);
    }
    
    /* ============================================
       SIDEBAR TAB STYLES
       ============================================ */
    
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        gap: 0.25rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
        padding: 4px;
        display: flex;
        gap: 4px;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        flex: 1;
        text-align: center;
        padding: 8px 12px !important;
        border-radius: 8px;
        font-size: 0.75rem !important;
        font-weight: 600;
        transition: all 0.2s ease;
        cursor: pointer;
        background: transparent;
        border: 1px solid transparent;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(0,170,255,0.1);
        border-color: rgba(0,170,255,0.2);
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(0,170,255,0.2), rgba(170,102,255,0.2)) !important;
        border-color: rgba(0,170,255,0.3) !important;
        color: #00aaff !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label > div {
        font-size: 0.75rem !important;
    }
    
    /* Sidebar stepper button styling */
    [data-testid="stSidebar"] button {
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        padding: 0.4rem !important;
        min-height: 2rem !important;
    }
    
    /* ============================================
       MAIN CONTENT STYLES
       ============================================ */
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-red), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    
    .overview-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.2rem;
        margin: 2rem 0;
    }

    .overview-card {
        background: linear-gradient(160deg, rgba(0,170,255,0.12), rgba(255,51,102,0.08));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .overview-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 24px 50px rgba(0,0,0,0.45);
    }

    .overview-card h3 {
        margin: 0;
        font-size: 1.1rem;
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-primary);
    }

    .overview-card ul {
        padding-left: 1.2rem;
        margin: 0;
        color: var(--text-secondary);
    }

    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.2rem;
        margin: 1.5rem 0;
    }
    
    .card-grid-5 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .card-grid-6 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.8rem;
        margin: 1.5rem 0;
    }

    .info-card, .data-card, .result-card {
        background: linear-gradient(160deg, var(--bg-card), rgba(0,170,255,0.05));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.25rem;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 12px 28px rgba(0,0,0,0.32);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .info-card:hover, .data-card:hover, .result-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    }
    
    /* Enhanced Display Cards */
    .display-card {
        background: linear-gradient(160deg, rgba(26,26,37,0.95), rgba(0,170,255,0.08));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .display-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        border-radius: 20px 20px 0 0;
    }
    
    .display-card.critical::before {
        background: linear-gradient(90deg, var(--accent-red), var(--accent-yellow));
    }
    
    .display-card.success::before {
        background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan));
    }
    
    .display-card.warning::before {
        background: linear-gradient(90deg, var(--accent-yellow), var(--accent-red));
    }
    
    .display-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0,0,0,0.5);
        border-color: rgba(0,170,255,0.3);
    }
    
    .display-card .card-badge {
        align-self: flex-start;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    
    .card-badge.critical {
        background: linear-gradient(135deg, rgba(255,51,102,0.25), rgba(255,51,102,0.1));
        color: var(--accent-red);
        border: 1px solid rgba(255,51,102,0.3);
    }
    
    .card-badge.warning {
        background: linear-gradient(135deg, rgba(255,170,0,0.25), rgba(255,170,0,0.1));
        color: var(--accent-yellow);
        border: 1px solid rgba(255,170,0,0.3);
    }
    
    .card-badge.success {
        background: linear-gradient(135deg, rgba(0,255,136,0.25), rgba(0,255,136,0.1));
        color: var(--accent-green);
        border: 1px solid rgba(0,255,136,0.3);
    }
    
    .card-badge.info {
        background: linear-gradient(135deg, rgba(0,170,255,0.25), rgba(0,170,255,0.1));
        color: var(--accent-blue);
        border: 1px solid rgba(0,170,255,0.3);
    }
    
    .card-badge.purple {
        background: linear-gradient(135deg, rgba(170,102,255,0.25), rgba(170,102,255,0.1));
        color: var(--accent-purple);
        border: 1px solid rgba(170,102,255,0.3);
    }
    
    .display-card .card-timestamp {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .display-card .card-timestamp::before {
        content: '🕐';
        font-size: 0.9rem;
    }
    
    .display-card .card-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--text-primary), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .display-card.critical .card-value {
        background: linear-gradient(135deg, var(--accent-red), var(--accent-yellow));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .display-card.success .card-value {
        background: linear-gradient(135deg, var(--accent-green), var(--accent-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .display-card .card-score {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-top: auto;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    
    .display-card .score-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        color: var(--text-secondary);
    }
    
    .display-card .score-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent-blue);
        background: rgba(0,170,255,0.1);
        padding: 4px 10px;
        border-radius: 8px;
    }
    
    .display-card.critical .score-value {
        color: var(--accent-red);
        background: rgba(255,51,102,0.1);
    }
    
    .display-card .score-bar {
        flex: 1;
        height: 6px;
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .display-card .score-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        transition: width 0.5s ease;
    }
    
    .display-card.critical .score-bar-fill {
        background: linear-gradient(90deg, var(--accent-yellow), var(--accent-red));
    }
    
    /* Stats Display Card */
    .stats-card {
        background: linear-gradient(160deg, rgba(26,26,37,0.9), rgba(170,102,255,0.08));
        border: 1px solid rgba(170,102,255,0.15);
        border-radius: 18px;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    }
    
    .stats-card .stats-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .stats-card .stats-icon {
        font-size: 1.5rem;
    }
    
    .stats-card .stats-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06rem;
        color: var(--text-secondary);
    }
    
    .stats-card .stats-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .stats-card .stats-meta {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }
    
    .mini-card {
        background: linear-gradient(145deg, rgba(26,26,37,0.9), rgba(18,18,26,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.85rem;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    
    .mini-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0,170,255,0.3);
    }
    
    .mini-card .mini-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.06rem;
        color: var(--text-secondary);
        margin-bottom: 0.35rem;
    }
    
    .mini-card .mini-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .feature-card {
        background: linear-gradient(160deg, rgba(170,102,255,0.1), rgba(0,170,255,0.08));
        border: 1px solid rgba(170,102,255,0.2);
        border-radius: 16px;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
    }
    
    .feature-card .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.25rem;
    }
    
    .feature-card .feature-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .feature-card .feature-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }
    
    .health-card {
        background: linear-gradient(145deg, var(--bg-card), rgba(0,255,136,0.05));
        border: 1px solid rgba(0,255,136,0.15);
        border-radius: 14px;
        padding: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }
    
    .health-card:hover {
        transform: translateY(-3px);
    }
    
    .health-card .health-icon {
        font-size: 1.5rem;
    }
    
    .health-card .health-content {
        flex: 1;
    }
    
    .health-card .health-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        color: var(--text-secondary);
    }
    
    .health-card .health-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .health-status {
        font-size: 0.65rem;
        padding: 3px 8px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .health-status.good { background: rgba(0,255,136,0.15); color: var(--accent-green); }
    .health-status.warning { background: rgba(255,170,0,0.15); color: var(--accent-yellow); }
    .health-status.critical { background: rgba(255,51,102,0.15); color: var(--accent-red); }
    
    .summary-banner {
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,170,255,0.1));
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .summary-banner.warning {
        background: linear-gradient(135deg, rgba(255,170,0,0.1), rgba(255,51,102,0.1));
        border-color: rgba(255,170,0,0.2);
    }
    
    .summary-banner .banner-content {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .summary-banner .banner-icon { font-size: 2rem; }
    
    .summary-banner .banner-text h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.25rem 0;
    }
    
    .summary-banner .banner-text p {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0;
    }
    
    .summary-banner .banner-stats { display: flex; gap: 1.5rem; }
    .summary-banner .banner-stat { text-align: center; }
    
    .summary-banner .banner-stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .summary-banner .banner-stat-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        color: var(--text-secondary);
    }
    
    .insights-panel {
        background: linear-gradient(160deg, rgba(0,170,255,0.08), rgba(170,102,255,0.08));
        border: 1px solid rgba(0,170,255,0.15);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1.5rem 0;
    }
    
    .insights-panel .insights-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--accent-blue);
        margin-bottom: 0.75rem;
    }
    
    .insights-panel .insight-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .insights-panel .insight-item:last-child { border-bottom: none; }
    .insights-panel .insight-icon { font-size: 1rem; }
    
    .insights-panel .insight-text {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }

    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        color: var(--text-secondary);
    }

    .card-metric {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .card-meta {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    .card-tag {
        align-self: flex-start;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06rem;
        text-transform: uppercase;
    }

    .tag-active { background: rgba(0,255,136,0.15); color: var(--accent-green); }
    .tag-warning { background: rgba(255,170,0,0.15); color: var(--accent-yellow); }
    .tag-critical { background: rgba(255,51,102,0.18); color: var(--accent-red); }
    .tag-info { background: rgba(0,170,255,0.15); color: var(--accent-blue); }
    .tag-purple { background: rgba(170,102,255,0.15); color: var(--accent-purple); }

    .metric-card {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-secondary));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-normal { color: var(--accent-green); }
    .status-warning { color: var(--accent-yellow); }
    .status-critical { color: var(--accent-red); }
    
    .alert-box {
        background: linear-gradient(135deg, rgba(255,51,102,0.2), rgba(255,51,102,0.05));
        border-left: 4px solid var(--accent-red);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .alert-box.warning {
        background: linear-gradient(135deg, rgba(255,170,0,0.2), rgba(255,170,0,0.05));
        border-left-color: var(--accent-yellow);
    }
    
    .alert-box.info {
        background: linear-gradient(135deg, rgba(0,170,255,0.2), rgba(0,170,255,0.05));
        border-left-color: var(--accent-blue);
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-blue);
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
    }
    
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0,255,136,0.1);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        color: var(--accent-green);
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    .stSelectbox > div > div {
        background-color: var(--bg-card);
    }
    
    .stSlider > div > div {
        background-color: var(--accent-blue);
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
    
    .detection-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-critical { background: var(--accent-red); color: white; }
    .badge-warning { background: var(--accent-yellow); color: black; }
    .badge-normal { background: var(--accent-green); color: black; }
    
    /* ============================================
       ENHANCED SIDEBAR SLIDER STYLING
       ============================================ */
    
    /* Sidebar Slider Labels - High Visibility */
    [data-testid="stSidebar"] [data-testid="stSlider"] label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stSlider"] label p {
        color: #ffffff !important;
        font-size: 0.9rem !important;
    }
    
    /* Sidebar Slider Value */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: #00aaff !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Slider Track */
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
        background: linear-gradient(90deg, #00aaff, #aa66ff) !important;
    }
    
    /* ============================================
       TEXT VISIBILITY FIXES
       ============================================ */
    
    /* File Uploader */
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] label p {
        color: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: rgba(26, 26, 37, 0.9) !important;
        border: 2px dashed rgba(0, 170, 255, 0.5) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stFileUploader"] section span {
        color: #e0e0f0 !important;
    }
    
    [data-testid="stFileUploader"] section small {
        color: #b0b0c0 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #00aaff, #0088cc) !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    /* Selectbox */
    [data-testid="stSelectbox"] label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSelectbox"] label p {
        color: #ffffff !important;
    }
    
    [data-testid="stSelectbox"] > div > div {
        background: rgba(26, 26, 37, 0.95) !important;
        border: 1px solid rgba(0, 170, 255, 0.3) !important;
        color: #ffffff !important;
    }
    
    /* Expander */
    [data-testid="stExpander"] summary {
        color: #ffffff !important;
    }
    
    [data-testid="stExpander"] summary span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Caption */
    [data-testid="stCaption"], .stCaption {
        color: #b0b0c0 !important;
    }
    
    /* General Text */
    .stMarkdown p {
        color: #e8e8f0 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
    }
    
    /* Tab Labels */
    .stTabs [data-baseweb="tab-list"] button {
        color: #c0c0d0 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton button, .stDownloadButton button {
        color: #ffffff !important;
    }
    
    /* Info/Warning/Error Messages */
    .stAlert p {
        color: #e0e0f0 !important;
    }
    
    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        color: #ffffff !important;
        background: rgba(26, 26, 37, 0.9) !important;
    }
    
    /* Checkbox and Radio */
    .stCheckbox label span, .stRadio label, .stRadio label span {
        color: #e0e0f0 !important;
    }
    
    /* Pulse Animation for Status */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ANOMALY DETECTION CLASSES
# ============================================================================

class StatisticalAnomalyDetector:
    """Statistical methods for anomaly detection"""
    
    def __init__(self, window_size=100, z_threshold=3.0, iqr_multiplier=1.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.history = deque(maxlen=window_size)
        
    def detect(self, value):
        """Detect anomaly using multiple statistical methods"""
        self.history.append(value)
        
        if len(self.history) < 10:
            return False, 0.0, {}
        
        data = np.array(self.history)
        
        # Z-score method
        mean = np.mean(data)
        std = np.std(data)
        z_score = (value - mean) / (std + 1e-8)
        z_anomaly = abs(z_score) > self.z_threshold
        
        # IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        iqr_anomaly = value < lower_bound or value > upper_bound
        
        # Modified Z-score (more robust)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z = 0.6745 * (value - median) / (mad + 1e-8)
        mad_anomaly = abs(modified_z) > self.z_threshold
        
        # Grubbs test for outliers
        grubbs_stat = (max(abs(data - mean))) / std if std > 0 else 0
        n = len(data)
        t_crit = stats.t.ppf(1 - 0.05/(2*n), n-2) if n > 2 else 0
        grubbs_crit = ((n-1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        grubbs_anomaly = grubbs_stat > grubbs_crit
        
        # Combine methods
        anomaly_score = (
            0.3 * (abs(z_score) / self.z_threshold) +
            0.3 * (1 if iqr_anomaly else 0) +
            0.2 * (abs(modified_z) / self.z_threshold) +
            0.2 * (1 if grubbs_anomaly else 0)
        )
        
        is_anomaly = anomaly_score > 0.5
        
        details = {
            'z_score': z_score,
            'modified_z': modified_z,
            'iqr_bounds': (lower_bound, upper_bound),
            'grubbs_stat': grubbs_stat,
            'methods_triggered': sum([z_anomaly, iqr_anomaly, mad_anomaly, grubbs_anomaly])
        }
        
        return is_anomaly, min(anomaly_score, 1.0), details


class MLAnomalyDetector:
    """Machine Learning based anomaly detection"""
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.elliptic_envelope = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True
        )
        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=contamination
        )
        self.fitted = False
        self.scaler = StandardScaler()
        
    def fit(self, X):
        """Fit all models on training data"""
        if len(X) < 50:
            return False
            
        X_scaled = self.scaler.fit_transform(X.reshape(-1, 1) if len(X.shape) == 1 else X)
        
        try:
            self.isolation_forest.fit(X_scaled)
            self.elliptic_envelope.fit(X_scaled)
            self.lof.fit(X_scaled)
            self.one_class_svm.fit(X_scaled)
            self.fitted = True
            return True
        except Exception as e:
            return False
            
    def detect(self, X):
        """Detect anomalies using ensemble of ML methods"""
        if not self.fitted:
            return np.zeros(len(X)), np.zeros(len(X))
            
        X_scaled = self.scaler.transform(X.reshape(-1, 1) if len(X.shape) == 1 else X)
        
        # Get predictions from each model (-1 for anomaly, 1 for normal)
        if_pred = self.isolation_forest.predict(X_scaled)
        if_scores = -self.isolation_forest.score_samples(X_scaled)
        
        try:
            ee_pred = self.elliptic_envelope.predict(X_scaled)
        except Exception:
            ee_pred = np.ones(len(X))
            
        lof_pred = self.lof.predict(X_scaled)
        svm_pred = self.one_class_svm.predict(X_scaled)
        
        # Ensemble voting
        votes = np.vstack([if_pred, ee_pred, lof_pred, svm_pred])
        ensemble_pred = np.sum(votes == -1, axis=0) / 4  # Proportion voting anomaly
        
        # Binary prediction (majority vote)
        is_anomaly = (ensemble_pred >= 0.5).astype(int)
        
        return is_anomaly, ensemble_pred


class DeepAnomalyDetector:
    """Autoencoder-based anomaly detection (simulated for demo)"""
    
    def __init__(self, threshold_percentile=95):
        self.threshold_percentile = threshold_percentile
        self.reconstruction_threshold = None
        self.history = []
        
    def fit(self, X):
        """Fit autoencoder on normal data"""
        # Simulate reconstruction errors (in production, use actual autoencoder)
        self.reconstruction_errors = np.random.exponential(0.1, len(X))
        self.reconstruction_threshold = np.percentile(
            self.reconstruction_errors, 
            self.threshold_percentile
        )
        
    def detect(self, X):
        """Detect anomalies based on reconstruction error"""
        # Simulate reconstruction error (higher for anomalies)
        base_error = np.abs(np.diff(X, prepend=X[0])) * np.random.uniform(0.5, 1.5, len(X))
        
        if self.reconstruction_threshold is None:
            self.reconstruction_threshold = np.percentile(base_error, 95)
            
        anomaly_scores = base_error / (self.reconstruction_threshold + 1e-8)
        is_anomaly = (anomaly_scores > 1.0).astype(int)
        
        return is_anomaly, np.clip(anomaly_scores, 0, 2)


class ModelDriftDetector:
    """Detect drift in model inputs and predictions"""
    
    def __init__(self, reference_window=500, detection_window=50, drift_threshold=0.1):
        self.reference_window = reference_window
        self.detection_window = detection_window
        self.drift_threshold = drift_threshold
        self.reference_data = None
        self.drift_history = []
        
    def set_reference(self, data):
        """Set reference distribution"""
        self.reference_data = np.array(data[-self.reference_window:])
        self.reference_stats = {
            'mean': np.mean(self.reference_data),
            'std': np.std(self.reference_data),
            'median': np.median(self.reference_data),
            'q25': np.percentile(self.reference_data, 25),
            'q75': np.percentile(self.reference_data, 75)
        }
        
    def detect_drift(self, current_data):
        """Detect if current data has drifted from reference"""
        if self.reference_data is None or len(current_data) < 10:
            return False, 0.0, {}
            
        current = np.array(current_data[-self.detection_window:])
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(self.reference_data, current)
        
        # Population Stability Index
        psi = self._calculate_psi(self.reference_data, current)
        
        # Jensen-Shannon divergence (approximated)
        js_divergence = self._calculate_js_divergence(self.reference_data, current)
        
        # Mean shift detection
        mean_shift = abs(np.mean(current) - self.reference_stats['mean']) / (self.reference_stats['std'] + 1e-8)
        
        # Variance ratio test
        var_ratio = np.var(current) / (np.var(self.reference_data) + 1e-8)
        variance_drift = abs(np.log(var_ratio))
        
        # Combined drift score
        drift_score = (
            0.25 * ks_stat +
            0.25 * min(psi / 0.25, 1.0) +
            0.20 * min(js_divergence, 1.0) +
            0.15 * min(mean_shift / 3, 1.0) +
            0.15 * min(variance_drift / 2, 1.0)
        )
        
        drift_detected = drift_score > self.drift_threshold
        
        details = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'psi': psi,
            'js_divergence': js_divergence,
            'mean_shift': mean_shift,
            'variance_drift': variance_drift,
            'current_mean': np.mean(current),
            'reference_mean': self.reference_stats['mean']
        }
        
        self.drift_history.append({
            'timestamp': datetime.now(),
            'score': drift_score,
            'detected': drift_detected
        })
        
        return drift_detected, drift_score, details
        
    def _calculate_psi(self, reference, current, buckets=10):
        """Calculate Population Stability Index"""
        try:
            _, bin_edges = np.histogram(reference, bins=buckets)
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)
            
            ref_pct = ref_counts / len(reference)
            curr_pct = curr_counts / len(current)
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 1e-10, ref_pct)
            curr_pct = np.where(curr_pct == 0, 1e-10, curr_pct)
            
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            return abs(psi)
        except Exception:
            return 0.0
            
    def _calculate_js_divergence(self, p, q, buckets=10):
        """Calculate Jensen-Shannon divergence"""
        try:
            _, bin_edges = np.histogram(np.concatenate([p, q]), bins=buckets)
            p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
            q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
            
            p_hist = p_hist + 1e-10
            q_hist = q_hist + 1e-10
            
            m = 0.5 * (p_hist + q_hist)
            
            js = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)
            return js
        except Exception:
            return 0.0


class MarketRegimeDetector:
    """Detect market regimes using Gaussian Mixture Models"""
    
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False
        self.regime_names = {
            0: 'Bullish Momentum',
            1: 'Bearish Decline',
            2: 'Consolidation',
            3: 'High Volatility'
        }
        self.regime_colors = {
            0: '#00ff88',  # Green
            1: '#ff3366',  # Red
            2: '#00aaff',  # Blue
            3: '#ffaa00'   # Yellow
        }
        
    def extract_features(self, data, window=20):
        """Extract regime detection features"""
        if len(data) < window:
            return None
            
        returns = np.diff(data) / data[:-1]
        
        features = []
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            features.append([
                np.mean(window_returns),       # Mean return
                np.std(window_returns),        # Volatility
                stats.skew(window_returns),    # Skewness
                stats.kurtosis(window_returns) # Kurtosis
            ])
            
        return np.array(features)
        
    def fit(self, data):
        """Fit GMM on historical data"""
        features = self.extract_features(data)
        if features is None or len(features) < self.n_regimes * 10:
            return False
            
        features_scaled = self.scaler.fit_transform(features)
        self.gmm.fit(features_scaled)
        self.fitted = True
        return True
        
    def detect_regime(self, data, window=20):
        """Detect current market regime"""
        if not self.fitted or len(data) < window + 1:
            return None
            
        features = self.extract_features(data[-window-10:], window)
        if features is None or len(features) == 0:
            return None
            
        features_scaled = self.scaler.transform(features[-1:])
        
        regime = self.gmm.predict(features_scaled)[0]
        probabilities = self.gmm.predict_proba(features_scaled)[0]
        
        return {
            'regime': int(regime),
            'name': self.regime_names.get(regime, f'Regime {regime}'),
            'color': self.regime_colors.get(regime, '#ffffff'),
            'confidence': float(np.max(probabilities)),
            'probabilities': probabilities.tolist()
        }


class AnomalyAlertManager:
    """Manage and display anomaly alerts"""
    
    def __init__(self, max_alerts=100):
        self.alerts = deque(maxlen=max_alerts)
        self.alert_counts = defaultdict(int)
        
    def add_alert(self, alert_type, severity, message, details=None):
        """Add new alert"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        self.alerts.append(alert)
        self.alert_counts[alert_type] += 1
        
    def get_recent_alerts(self, n=10, severity_filter=None):
        """Get recent alerts, optionally filtered by severity"""
        alerts = list(self.alerts)[-n:]
        if severity_filter:
            alerts = [a for a in alerts if a['severity'] == severity_filter]
        return alerts[::-1]  # Most recent first
        
    def get_alert_summary(self):
        """Get summary of alerts"""
        return dict(self.alert_counts)


# ============================================================================
# DATA SIMULATION
# ============================================================================

def generate_synthetic_data(n_points=500, with_anomalies=True, anomaly_rate=0.05):
    """Generate synthetic time series data with injected anomalies"""
    np.random.seed(int(time.time()) % 1000)
    
    # Base signal: trending with seasonality
    t = np.linspace(0, 10, n_points)
    trend = 100 + 0.5 * t
    seasonality = 5 * np.sin(2 * np.pi * t / 2)
    noise = np.random.normal(0, 1, n_points)
    
    data = trend + seasonality + noise
    
    # Add autocorrelation
    for i in range(1, len(data)):
        data[i] = 0.3 * data[i-1] + 0.7 * data[i]
    
    anomaly_labels = np.zeros(n_points)
    
    if with_anomalies:
        n_anomalies = int(n_points * anomaly_rate)
        anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'dip', 'shift', 'variance'])
            
            if anomaly_type == 'spike':
                data[idx] += np.random.uniform(8, 15)
            elif anomaly_type == 'dip':
                data[idx] -= np.random.uniform(8, 15)
            elif anomaly_type == 'shift':
                shift_length = min(10, n_points - idx)
                data[idx:idx+shift_length] += np.random.uniform(3, 6)
            else:  # variance change
                var_length = min(15, n_points - idx)
                data[idx:idx+var_length] *= np.random.uniform(1.5, 2.5)
                
            anomaly_labels[idx] = 1
            
    timestamps = pd.date_range(end=datetime.now(), periods=n_points, freq='1min')
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'value': data,
        'is_anomaly_injected': anomaly_labels
    })


def generate_multivariate_data(n_points=500):
    """Generate multivariate time series for regime detection"""
    np.random.seed(int(time.time()) % 1000)
    
    t = np.linspace(0, 20, n_points)
    
    # Create regime switches
    regimes = np.zeros(n_points, dtype=int)
    regime_change_points = np.sort(np.random.choice(n_points, 5, replace=False))
    
    for i, cp in enumerate(regime_change_points):
        regimes[cp:] = (i + 1) % 4
    
    # Generate data based on regime
    prices = np.zeros(n_points)
    prices[0] = 100
    
    for i in range(1, n_points):
        regime = regimes[i]
        if regime == 0:  # Bull
            drift = 0.0005
            vol = 0.01
        elif regime == 1:  # Bear
            drift = -0.0003
            vol = 0.015
        elif regime == 2:  # Consolidation
            drift = 0.0001
            vol = 0.005
        else:  # High volatility
            drift = 0
            vol = 0.03
            
        prices[i] = prices[i-1] * (1 + drift + vol * np.random.randn())
    
    # Calculate features
    returns = np.diff(prices) / prices[:-1]
    returns = np.insert(returns, 0, 0)
    
    volatility = pd.Series(returns).rolling(20).std().fillna(0.01).values
    momentum = pd.Series(prices).pct_change(20).fillna(0).values
    
    timestamps = pd.date_range(end=datetime.now(), periods=n_points, freq='1min')
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'returns': returns,
        'volatility': volatility,
        'momentum': momentum,
        'regime': regimes
    })


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_time_series_chart(df, anomaly_scores, detected_anomalies, title="Time Series with Anomalies"):
    """Create interactive time series chart with anomaly highlighting"""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Time Series', 'Anomaly Score')
    )
    
    # Main time series
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines',
            name='Value',
            line=dict(color='#00aaff', width=1.5),
            hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Highlight anomalies
    anomaly_mask = detected_anomalies > 0
    if anomaly_mask.any():
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'][anomaly_mask],
                y=df['value'][anomaly_mask],
                mode='markers',
                name='Detected Anomaly',
                marker=dict(
                    color='#ff3366',
                    size=12,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                hovertemplate='%{x}<br>Anomaly Value: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Anomaly score
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=anomaly_scores,
            mode='lines',
            name='Anomaly Score',
            line=dict(color='#ffaa00', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255,170,0,0.2)',
            hovertemplate='%{x}<br>Score: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Threshold line
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="#ff3366",
        annotation_text="Threshold", row=2, col=1
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white')),
        height=500,
        template='plotly_dark',
        paper_bgcolor='rgba(10,10,15,0)',
        plot_bgcolor='rgba(26,26,37,1)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_drift_chart(drift_history, title="Model Drift Over Time"):
    """Create drift monitoring chart"""
    
    if not drift_history:
        fig = go.Figure()
        fig.add_annotation(
            text="No drift data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='gray')
        )
    else:
        timestamps = [d['timestamp'] for d in drift_history]
        scores = [d['score'] for d in drift_history]
        detected = [d['detected'] for d in drift_history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=scores,
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='#aa66ff', width=2),
            marker=dict(
                size=8,
                color=['#ff3366' if d else '#00ff88' for d in detected],
                line=dict(color='white', width=1)
            ),
            hovertemplate='%{x}<br>Score: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_hline(
            y=0.1, line_dash="dash", line_color="#ffaa00",
            annotation_text="Drift Threshold"
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white')),
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(10,10,15,0)',
        plot_bgcolor='rgba(26,26,37,1)',
        xaxis_title="Time",
        yaxis_title="Drift Score"
    )
    
    return fig


def create_regime_chart(df, regime_info, title="Market Regime Detection"):
    """Create market regime visualization"""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price with Regime', 'Regime Probabilities')
    )
    
    # Price colored by regime
    regimes = df['regime'].values
    unique_regimes = np.unique(regimes)
    
    colors = {0: '#00ff88', 1: '#ff3366', 2: '#00aaff', 3: '#ffaa00'}
    names = {0: 'Bullish', 1: 'Bearish', 2: 'Consolidation', 3: 'High Vol'}
    
    for regime in unique_regimes:
        mask = regimes == regime
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'][mask],
                y=df['price'][mask],
                mode='markers',
                name=names.get(regime, f'Regime {regime}'),
                marker=dict(
                    color=colors.get(regime, 'gray'),
                    size=4
                ),
                hovertemplate='%{x}<br>Price: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Regime probability bars (if available)
    if regime_info and 'probabilities' in regime_info:
        probs = regime_info['probabilities']
        fig.add_trace(
            go.Bar(
                x=list(names.values()),
                y=probs,
                marker_color=[colors[i] for i in range(len(probs))],
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white')),
        height=500,
        template='plotly_dark',
        paper_bgcolor='rgba(10,10,15,0)',
        plot_bgcolor='rgba(26,26,37,1)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig


def create_detection_summary_chart(methods_summary):
    """Create summary chart of detection methods"""
    
    fig = go.Figure()
    
    methods = list(methods_summary.keys())
    detected = [methods_summary[m]['detected'] for m in methods]
    scores = [methods_summary[m]['score'] for m in methods]
    
    colors = ['#ff3366' if d > 0 else '#00ff88' for d in detected]
    
    fig.add_trace(go.Bar(
        x=methods,
        y=scores,
        marker_color=colors,
        text=[f'{s:.2f}' for s in scores],
        textposition='outside',
        hovertemplate='%{x}<br>Score: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.5)
    
    fig.update_layout(
        title=dict(text="Detection Methods Comparison", font=dict(size=16, color='white')),
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(10,10,15,0)',
        plot_bgcolor='rgba(26,26,37,1)',
        xaxis_title="Method",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1])
    )
    
    return fig


def create_feature_importance_chart(importance_dict, title="Feature Importance"):
    """Create feature importance visualization"""
    
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1][:15]
    features = [features[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Importance')
        ),
        text=[f'{v:.3f}' for v in importance],
        textposition='outside',
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='white')),
        height=400,
        template='plotly_dark',
        paper_bgcolor='rgba(10,10,15,0)',
        plot_bgcolor='rgba(26,26,37,1)',
        xaxis_title="Importance Score",
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def format_timedelta(delta):
    """Create compact human-readable duration strings"""
    if delta is None:
        return "n/a"

    try:
        total_seconds = int(delta.total_seconds())
    except Exception:
        return "n/a"

    if total_seconds <= 0:
        return "n/a"

    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts[:2])


def reset_detectors(z_threshold, contamination, drift_threshold):
    """Recreate detector objects with updated configuration"""
    st.session_state.stat_detector = StatisticalAnomalyDetector(z_threshold=z_threshold)
    st.session_state.ml_detector = MLAnomalyDetector(contamination=contamination)
    st.session_state.deep_detector = DeepAnomalyDetector()
    st.session_state.drift_detector = ModelDriftDetector(drift_threshold=drift_threshold)
    st.session_state.regime_detector = MarketRegimeDetector()
    st.session_state.alert_manager = AnomalyAlertManager()


def build_multivariate_dataset(df):
    """Derive multivariate features from the active univariate series"""
    if df is None or df.empty or 'timestamp' not in df.columns or 'value' not in df.columns:
        return generate_multivariate_data()

    value_series = pd.to_numeric(df['value'], errors='coerce')
    value_series = value_series.ffill().bfill()
    timestamp_series = pd.to_datetime(df['timestamp'], errors='coerce')
    valid_mask = value_series.notna() & timestamp_series.notna()

    value_series = value_series[valid_mask].reset_index(drop=True)
    timestamp_series = timestamp_series[valid_mask].reset_index(drop=True)

    if value_series.empty:
        return generate_multivariate_data()

    price = value_series.values.astype(float)
    returns = pd.Series(price).pct_change().fillna(0).values

    rolling_vol = pd.Series(returns).rolling(20).std().replace(0, np.nan)
    fallback_vol = float(np.nanstd(returns)) if np.isfinite(np.nanstd(returns)) and np.nanstd(returns) > 0 else 0.01
    volatility = rolling_vol.fillna(fallback_vol).values

    momentum = pd.Series(price).pct_change(20).fillna(0).values

    if len(price) < 10:
        regimes = np.zeros(len(price), dtype=int)
    else:
        finite_returns = returns[np.isfinite(returns)]
        finite_volatility = volatility[np.isfinite(volatility)]

        ret_high = np.quantile(finite_returns, 0.65) if finite_returns.size else 0
        ret_low = np.quantile(finite_returns, 0.35) if finite_returns.size else 0
        vol_high = np.quantile(finite_volatility, 0.75) if finite_volatility.size else fallback_vol
        vol_low = np.quantile(finite_volatility, 0.25) if finite_volatility.size else fallback_vol / 2

        regimes = []
        for r, v in zip(returns, volatility):
            if v >= vol_high:
                regimes.append(3)
            elif r >= ret_high and v <= vol_high:
                regimes.append(0)
            elif r <= ret_low and v >= vol_low:
                regimes.append(1)
            else:
                regimes.append(2)
        regimes = np.array(regimes, dtype=int)

    return pd.DataFrame({
        'timestamp': timestamp_series.values,
        'price': price,
        'returns': returns,
        'volatility': volatility,
        'momentum': momentum,
        'regime': regimes
    })


def update_active_dataset(df, timestamp_col, value_col, dataset_label, anomaly_col=None):
    """Clean and register uploaded data as the active stream"""
    try:
        working_df = df.copy()
        working_df[timestamp_col] = pd.to_datetime(working_df[timestamp_col], errors='coerce')
        working_df[value_col] = pd.to_numeric(working_df[value_col], errors='coerce')
    except Exception as exc:
        st.error(f"Unable to parse selected columns: {exc}")
        return False

    working_df = working_df.dropna(subset=[timestamp_col, value_col])
    if working_df.empty:
        st.error("No valid rows remained after cleaning the uploaded dataset.")
        return False

    rename_map = {}
    if timestamp_col != 'timestamp':
        rename_map[timestamp_col] = 'timestamp'
    if value_col != 'value':
        rename_map[value_col] = 'value'
    working_df = working_df.rename(columns=rename_map)

    if 'is_anomaly_injected' not in working_df.columns:
        working_df['is_anomaly_injected'] = 0

    if anomaly_col and anomaly_col in working_df.columns:
        working_df['is_anomaly_injected'] = pd.to_numeric(working_df[anomaly_col], errors='coerce').fillna(0)

    working_df['is_anomaly_injected'] = (working_df['is_anomaly_injected'] > 0).astype(int)
    working_df = working_df.sort_values('timestamp').reset_index(drop=True)

    st.session_state.data = working_df
    st.session_state.active_dataset_label = dataset_label
    st.session_state.multivariate_data = build_multivariate_dataset(working_df)

    config = st.session_state.get('current_config', {})
    z_threshold = config.get('z_threshold', 3.0)
    contamination = config.get('contamination', 0.1)
    drift_threshold = config.get('drift_threshold', 0.1)

    reset_detectors(z_threshold, contamination, drift_threshold)
    st.session_state.last_update = datetime.now()

    run_full_analysis()
    return True


def render_overview_section():
    """Top-level overview describing purpose, workflow, and benefits"""
    st.markdown("""
    <div class="overview-grid">
        <div class="overview-card">
            <span class="card-tag tag-active">Purpose</span>
            <h3>Unified anomaly intelligence</h3>
            <p>This workspace monitors financial-style data streams in real time, surfacing anomalies, drift, and regime changes with explainable context.</p>
            <ul>
                <li>Blends statistical, ML, and deep detection strategies</li>
                <li>Tracks distribution drift and data health baselines</li>
                <li>Surfaces insights via interactive dashboards</li>
            </ul>
        </div>
        <div class="overview-card">
            <span class="card-tag tag-warning">How it works</span>
            <h3>Layered detection pipeline</h3>
            <ul>
                <li>Ingest data from CSV, JSON, Excel, or Parquet streams</li>
                <li>Standardize signals and compute multivariate features</li>
                <li>Run ensemble detectors, drift monitors, and regime classifiers</li>
            </ul>
        </div>
        <div class="overview-card">
            <span class="card-tag tag-active">Benefits</span>
            <h3>Actionable intelligence</h3>
            <ul>
                <li>Faster response to production issues and market shifts</li>
                <li>Automated alerting aligned to risk tolerances</li>
                <li>Exportable evidence for audit and reporting</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_data_hub_tab():
    """Interactive hub for managing input datasets with enhanced cards"""
    st.markdown('<h2 class="section-title">🗂 Data Management & Integration</h2>', unsafe_allow_html=True)
    
    # Feature cards for data hub capabilities
    st.markdown("""
    <div class="card-grid">
        <div class="feature-card">
            <div class="feature-icon">📤</div>
            <div class="feature-title">Upload Data</div>
            <div class="feature-desc">Import CSV, JSON, Excel, or Parquet files for analysis</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <div class="feature-title">Auto-Sync</div>
            <div class="feature-desc">Detectors recalibrate automatically on new data</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📥</div>
            <div class="feature-title">Export Results</div>
            <div class="feature-desc">Download processed data in multiple formats</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Schema Detection</div>
            <div class="feature-desc">Automatic column type inference and mapping</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Upload external datasets or review the active synthetic stream. The platform automatically harmonizes schema, recalibrates detectors, and refreshes analytics.")

    uploaded_file = st.file_uploader(
        "Upload time series dataset",
        type=["csv", "json", "xlsx", "xls", "parquet", "pq"],
        key="data_hub_uploader",
        help="Supported formats: CSV, JSON, Excel, and Parquet."
    )

    if uploaded_file is not None:
        extension = uploaded_file.name.lower().split('.')[-1]
        try:
            if extension == 'csv':
                uploaded_df = pd.read_csv(uploaded_file)
            elif extension == 'json':
                uploaded_df = pd.read_json(uploaded_file)
            elif extension in ('xlsx', 'xls'):
                uploaded_df = pd.read_excel(uploaded_file)
            elif extension in ('parquet', 'pq'):
                uploaded_df = pd.read_parquet(uploaded_file)
            else:
                st.error(f"Unsupported file type: {extension.upper()}")
                uploaded_df = None
        except Exception as exc:
            st.error(f"Unable to read uploaded file: {exc}")
            uploaded_df = None

        if uploaded_df is not None:
            if uploaded_df.empty:
                st.warning("Uploaded file contains no rows.")
            else:
                columns = uploaded_df.columns.tolist()

                default_time = next((c for c in columns if any(key in c.lower() for key in ['time', 'date'])), columns[0])
                default_value_candidates = [c for c in columns if c != default_time and any(key in c.lower() for key in ['value', 'price', 'metric', 'close', 'amount'])]
                default_value = default_value_candidates[0] if default_value_candidates else (columns[1] if len(columns) > 1 else columns[0])

                time_col = st.selectbox("Timestamp column", columns, index=columns.index(default_time) if default_time in columns else 0, key="data_hub_time_col")
                value_col = st.selectbox("Value column", columns, index=columns.index(default_value) if default_value in columns else 0, key="data_hub_value_col")

                anomaly_candidates = [c for c in columns if c not in {time_col, value_col} and any(tag in c.lower() for tag in ['anomaly', 'flag', 'label'])]
                anomaly_options = ["None"] + anomaly_candidates
                anomaly_selection = st.selectbox("Optional anomaly flag column", anomaly_options, key="data_hub_anomaly_col")
                anomaly_col = None if anomaly_selection == "None" else anomaly_selection

                with st.expander("Preview uploaded data", expanded=True):
                    st.dataframe(uploaded_df.head(20), use_container_width=True)
                    st.caption("Showing the first 20 rows for a quick schema check.")

                if st.button("Activate dataset", use_container_width=True, key="data_hub_activate"):
                    if update_active_dataset(uploaded_df, time_col, value_col, uploaded_file.name, anomaly_col=anomaly_col):
                        st.success("Dataset activated. Detectors recalibrated on the new stream.")
                        st.rerun()

    active_df = st.session_state.get('data')
    if active_df is None or active_df.empty:
        st.info("No active dataset is loaded yet.")
        return

    dataset_label = st.session_state.get('active_dataset_label', 'Synthetic Stream')
    data_points = len(active_df)
    time_min = pd.to_datetime(active_df['timestamp']).min() if data_points else None
    time_max = pd.to_datetime(active_df['timestamp']).max() if data_points else None
    coverage = format_timedelta(time_max - time_min if time_min is not None and time_max is not None else None)

    diffs = pd.to_datetime(active_df['timestamp']).sort_values().diff().dropna()
    cadence = format_timedelta(diffs.median() if not diffs.empty else None)

    anomaly_flags = int(active_df['is_anomaly_injected'].sum()) if 'is_anomaly_injected' in active_df.columns else 0
    last_refresh = st.session_state.get('last_update')
    refresh_text = last_refresh.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_refresh, datetime) else 'n/a'
    
    # Calculate additional stats
    values = active_df['value'].values
    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    st.markdown("### 📊 Active Dataset Overview")
    
    # Summary Banner using st.columns for reliable rendering
    ban_left, ban_right = st.columns([2, 1])
    with ban_left:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,170,255,0.1));
                    border: 1px solid rgba(0,255,136,0.2); border-radius: 16px;
                    padding: 1.25rem 1.5rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 2rem;">📁</span>
                <div>
                    <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600;
                               color: #ffffff; margin: 0 0 0.25rem 0;">{dataset_label}</h3>
                    <p style="font-size: 0.85rem; color: #a0a0b0; margin: 0;">Last refreshed: {refresh_text}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ban_right:
        st.markdown(f"""
        <div style="display: flex; gap: 1.5rem; justify-content: center; align-items: center; height: 100%;">
            <div style="text-align: center;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #ffffff;">{data_points:,}</div>
                <div style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0;">Points</div>
            </div>
            <div style="text-align: center;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #ffffff;">{coverage}</div>
                <div style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0;">Coverage</div>
            </div>
            <div style="text-align: center;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #ffffff;">{anomaly_flags}</div>
                <div style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0;">Flags</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Mini cards using st.columns for reliable rendering
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mini_card_style = """
        background: linear-gradient(145deg, rgba(26,26,37,0.9), rgba(18,18,26,0.95));
        border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 0.85rem; text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    """
    mini_label_style = "font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06rem; color: #a0a0b0; margin-bottom: 0.35rem;"
    mini_value_style = "font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #ffffff;"
    
    with mc1:
        st.markdown(f'<div style="{mini_card_style}"><div style="{mini_label_style}">Data Points</div><div style="{mini_value_style}">{data_points:,}</div></div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div style="{mini_card_style}"><div style="{mini_label_style}">Coverage</div><div style="{mini_value_style}">{coverage}</div></div>', unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div style="{mini_card_style}"><div style="{mini_label_style}">Cadence</div><div style="{mini_value_style}">{cadence}</div></div>', unsafe_allow_html=True)
    with mc4:
        st.markdown(f'<div style="{mini_card_style}"><div style="{mini_label_style}">Mean</div><div style="{mini_value_style}">{mean_val:.2f}</div></div>', unsafe_allow_html=True)
    with mc5:
        st.markdown(f'<div style="{mini_card_style}"><div style="{mini_label_style}">Std Dev</div><div style="{mini_value_style}">{std_val:.2f}</div></div>', unsafe_allow_html=True)
    with mc6:
        st.markdown(f'<div style="{mini_card_style}"><div style="{mini_label_style}">Range</div><div style="{mini_value_style}">{max_val - min_val:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Active dataset snapshot")
    st.dataframe(active_df.tail(50), use_container_width=True)
    st.caption("Latest 50 observations from the active data feed.")

    base_label = dataset_label.replace(' ', '_').lower()
    timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_bytes = active_df.to_csv(index=False).encode('utf-8')
    json_bytes = active_df.to_json(orient='records', date_format='iso').encode('utf-8')

    parquet_bytes = None
    parquet_error = None
    try:
        parquet_buffer = BytesIO()
        active_df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        parquet_bytes = parquet_buffer.getvalue()
    except Exception as exc:
        parquet_error = str(exc)

    excel_bytes = None
    excel_error = None
    try:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            active_df.to_excel(writer, index=False, sheet_name='data')
        excel_buffer.seek(0)
        excel_bytes = excel_buffer.getvalue()
    except Exception as exc:
        excel_error = str(exc)

    st.markdown("#### Export active dataset")
    download_cols = st.columns(4)

    with download_cols[0]:
        st.download_button(
            "⬇️ CSV",
            data=csv_bytes,
            file_name=f"{base_label}_{timestamp_suffix}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with download_cols[1]:
        st.download_button(
            "⬇️ JSON",
            data=json_bytes,
            file_name=f"{base_label}_{timestamp_suffix}.json",
            mime="application/json",
            use_container_width=True
        )

    with download_cols[2]:
        if parquet_bytes is not None:
            st.download_button(
                "⬇️ Parquet",
                data=parquet_bytes,
                file_name=f"{base_label}_{timestamp_suffix}.parquet",
                mime="application/octet-stream",
                use_container_width=True
            )
        else:
            message = "Install pyarrow or fastparquet for Parquet export."
            if parquet_error:
                message = f"{message} ({parquet_error})"
            st.caption(message)

    with download_cols[3]:
        if excel_bytes is not None:
            st.download_button(
                "⬇️ Excel",
                data=excel_bytes,
                file_name=f"{base_label}_{timestamp_suffix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            message = "Install openpyxl or xlsxwriter for Excel export."
            if excel_error:
                message = f"{message} ({excel_error})"
            st.caption(message)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Real-Time Anomaly Detection Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced ML-powered anomaly detection with drift monitoring and regime analysis</p>', unsafe_allow_html=True)
    render_overview_section()
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = generate_synthetic_data()
    if 'multivariate_data' not in st.session_state:
        st.session_state.multivariate_data = generate_multivariate_data()
    if 'stat_detector' not in st.session_state:
        st.session_state.stat_detector = StatisticalAnomalyDetector()
    if 'ml_detector' not in st.session_state:
        st.session_state.ml_detector = MLAnomalyDetector()
    if 'deep_detector' not in st.session_state:
        st.session_state.deep_detector = DeepAnomalyDetector()
    if 'drift_detector' not in st.session_state:
        st.session_state.drift_detector = ModelDriftDetector()
    if 'regime_detector' not in st.session_state:
        st.session_state.regime_detector = MarketRegimeDetector()
    if 'alert_manager' not in st.session_state:
        st.session_state.alert_manager = AnomalyAlertManager()
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    if 'active_dataset_label' not in st.session_state:
        st.session_state.active_dataset_label = 'Synthetic Stream'
    if 'current_config' not in st.session_state:
        st.session_state.current_config = {
            'data_points': 500,
            'anomaly_rate': 0.05,
            'z_threshold': 3.0,
            'contamination': 0.1,
            'drift_threshold': 0.1
        }
    
    # Enhanced Sidebar with Tabbed Layout
    with st.sidebar:
        # Compact Header
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(0,170,255,0.15), rgba(170,102,255,0.15));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            text-align: center;
        ">
            <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">🔍</span>
                <h2 style="
                    font-family: 'Space Grotesk', sans-serif;
                    font-size: 1rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #00aaff, #aa66ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                ">Anomaly Detection</h2>
            </div>
            <p style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.65rem;
                color: #888;
                margin: 0.5rem 0 0 0;
            ">Real-Time ML Platform v2.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Live Status Indicator (always visible)
        data = st.session_state.get('data')
        data_points_count = len(data) if data is not None and not data.empty else 0
        anomaly_count = int(data.get('is_anomaly_injected', pd.Series([0])).sum()) if data is not None and not data.empty else 0
        anomaly_pct = (anomaly_count / data_points_count * 100) if data_points_count > 0 else 0
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, rgba(0,255,136,0.08), rgba(0,170,255,0.08));
            border: 1px solid rgba(0,255,136,0.15);
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="
                    width: 6px; height: 6px;
                    background: #00ff88;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                    box-shadow: 0 0 8px #00ff88;
                "></div>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #00ff88; font-weight: 600;">LIVE</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #a0a0b0;">
                    <span style="color: #00aaff;">{data_points_count:,}</span> pts
                </span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #a0a0b0;">
                    <span style="color: {'#ff3366' if anomaly_count > 0 else '#00ff88'};">{anomaly_pct:.1f}%</span> anom
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabbed Navigation
        sidebar_tab = st.radio(
            "Navigation",
            ["⚙️ Settings", "⚡ Actions", "🤖 Models"],
            horizontal=True,
            label_visibility="collapsed",
            key="sidebar_tab_selector"
        )
        
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        
        # ============ SETTINGS TAB ============
        if sidebar_tab == "⚙️ Settings":
            # Data Generation Card
            st.markdown("""
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 10px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.75rem;
            ">
                <div style="
                    font-family: 'Space Grotesk', sans-serif;
                    font-size: 0.7rem;
                    font-weight: 600;
                    color: #888;
                    text-transform: uppercase;
                    letter-spacing: 0.05rem;
                ">
                    📁 Data Generation
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Data Points Stepper
            st.markdown("<div style='font-size: 0.8rem; color: #a0a0a0; margin-bottom: 0.25rem;'>Data Points</div>", unsafe_allow_html=True)
            dp_col1, dp_col2, dp_col3 = st.columns([1, 2, 1])
            with dp_col1:
                if st.button("➖", key="dp_minus", use_container_width=True):
                    current_dp = st.session_state.current_config.get('data_points', 500)
                    st.session_state.current_config['data_points'] = max(100, current_dp - 100)
                    st.rerun()
            with dp_col2:
                st.markdown(f"""
                <div style="
                    background: rgba(0,0,0,0.3);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    padding: 0.5rem;
                    text-align: center;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 1rem;
                    font-weight: 600;
                    color: #fff;
                ">{st.session_state.current_config.get('data_points', 500)}</div>
                """, unsafe_allow_html=True)
            with dp_col3:
                if st.button("➕", key="dp_plus", use_container_width=True):
                    current_dp = st.session_state.current_config.get('data_points', 500)
                    st.session_state.current_config['data_points'] = min(1000, current_dp + 100)
                    st.rerun()
            
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            
            # Anomaly Rate Stepper
            st.markdown("<div style='font-size: 0.8rem; color: #a0a0a0; margin-bottom: 0.25rem;'>Anomaly Rate</div>", unsafe_allow_html=True)
            ar_col1, ar_col2, ar_col3 = st.columns([1, 2, 1])
            with ar_col1:
                if st.button("➖", key="ar_minus", use_container_width=True):
                    current_ar = st.session_state.current_config.get('anomaly_rate', 0.05)
                    st.session_state.current_config['anomaly_rate'] = max(0.01, round(current_ar - 0.01, 2))
                    st.rerun()
            with ar_col2:
                st.markdown(f"""
                <div style="
                    background: rgba(0,0,0,0.3);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    padding: 0.5rem;
                    text-align: center;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 1rem;
                    font-weight: 600;
                    color: #fff;
                ">{st.session_state.current_config.get('anomaly_rate', 0.05):.0%}</div>
                """, unsafe_allow_html=True)
            with ar_col3:
                if st.button("➕", key="ar_plus", use_container_width=True):
                    current_ar = st.session_state.current_config.get('anomaly_rate', 0.05)
                    st.session_state.current_config['anomaly_rate'] = min(0.2, round(current_ar + 0.01, 2))
                    st.rerun()
            
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            
            # Detection Parameters Card
            st.markdown("""
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 10px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.75rem;
            ">
                <div style="
                    font-family: 'Space Grotesk', sans-serif;
                    font-size: 0.7rem;
                    font-weight: 600;
                    color: #888;
                    text-transform: uppercase;
                    letter-spacing: 0.05rem;
                ">
                    🎛️ Detection Parameters
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Z-Score Threshold Stepper
            st.markdown("<div style='font-size: 0.8rem; color: #a0a0a0; margin-bottom: 0.25rem;'>Z-Score Threshold</div>", unsafe_allow_html=True)
            z_col1, z_col2, z_col3 = st.columns([1, 2, 1])
            with z_col1:
                if st.button("➖", key="z_minus", use_container_width=True):
                    current_z = st.session_state.current_config.get('z_threshold', 3.0)
                    st.session_state.current_config['z_threshold'] = max(2.0, round(current_z - 0.1, 1))
                    st.rerun()
            with z_col2:
                st.markdown(f"""
                <div style="
                    background: rgba(0,0,0,0.3);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    padding: 0.5rem;
                    text-align: center;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 1rem;
                    font-weight: 600;
                    color: #fff;
                ">{st.session_state.current_config.get('z_threshold', 3.0):.1f}</div>
                """, unsafe_allow_html=True)
            with z_col3:
                if st.button("➕", key="z_plus", use_container_width=True):
                    current_z = st.session_state.current_config.get('z_threshold', 3.0)
                    st.session_state.current_config['z_threshold'] = min(4.0, round(current_z + 0.1, 1))
                    st.rerun()
            
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            
            # ML Contamination Stepper
            st.markdown("<div style='font-size: 0.8rem; color: #a0a0a0; margin-bottom: 0.25rem;'>ML Contamination</div>", unsafe_allow_html=True)
            ml_col1, ml_col2, ml_col3 = st.columns([1, 2, 1])
            with ml_col1:
                if st.button("➖", key="ml_minus", use_container_width=True):
                    current_ml = st.session_state.current_config.get('contamination', 0.1)
                    st.session_state.current_config['contamination'] = max(0.05, round(current_ml - 0.01, 2))
                    st.rerun()
            with ml_col2:
                st.markdown(f"""
                <div style="
                    background: rgba(0,0,0,0.3);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    padding: 0.5rem;
                    text-align: center;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 1rem;
                    font-weight: 600;
                    color: #fff;
                ">{st.session_state.current_config.get('contamination', 0.1):.0%}</div>
                """, unsafe_allow_html=True)
            with ml_col3:
                if st.button("➕", key="ml_plus", use_container_width=True):
                    current_ml = st.session_state.current_config.get('contamination', 0.1)
                    st.session_state.current_config['contamination'] = min(0.2, round(current_ml + 0.01, 2))
                    st.rerun()
            
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            
            # Drift Threshold Stepper
            st.markdown("<div style='font-size: 0.8rem; color: #a0a0a0; margin-bottom: 0.25rem;'>Drift Threshold</div>", unsafe_allow_html=True)
            dr_col1, dr_col2, dr_col3 = st.columns([1, 2, 1])
            with dr_col1:
                if st.button("➖", key="dr_minus", use_container_width=True):
                    current_dr = st.session_state.current_config.get('drift_threshold', 0.1)
                    st.session_state.current_config['drift_threshold'] = max(0.05, round(current_dr - 0.01, 2))
                    st.rerun()
            with dr_col2:
                st.markdown(f"""
                <div style="
                    background: rgba(0,0,0,0.3);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    padding: 0.5rem;
                    text-align: center;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 1rem;
                    font-weight: 600;
                    color: #fff;
                ">{st.session_state.current_config.get('drift_threshold', 0.1):.2f}</div>
                """, unsafe_allow_html=True)
            with dr_col3:
                if st.button("➕", key="dr_plus", use_container_width=True):
                    current_dr = st.session_state.current_config.get('drift_threshold', 0.1)
                    st.session_state.current_config['drift_threshold'] = min(0.3, round(current_dr + 0.01, 2))
                    st.rerun()
            
            # Set variables for detector sync
            data_points = st.session_state.current_config.get('data_points', 500)
            anomaly_rate = st.session_state.current_config.get('anomaly_rate', 0.05)
            z_threshold = st.session_state.current_config.get('z_threshold', 3.0)
            contamination = st.session_state.current_config.get('contamination', 0.1)
            drift_threshold = st.session_state.current_config.get('drift_threshold', 0.1)
        
        # ============ ACTIONS TAB ============
        elif sidebar_tab == "⚡ Actions":
            # Get current config values
            data_points = st.session_state.current_config.get('data_points', 500)
            anomaly_rate = st.session_state.current_config.get('anomaly_rate', 0.05)
            z_threshold = st.session_state.current_config.get('z_threshold', 3.0)
            contamination = st.session_state.current_config.get('contamination', 0.1)
            drift_threshold = st.session_state.current_config.get('drift_threshold', 0.1)
            
            # Quick Actions
            st.markdown("""
            <div style="
                font-family: 'Space Grotesk', sans-serif;
                font-size: 0.7rem;
                font-weight: 600;
                color: #aa66ff;
                text-transform: uppercase;
                letter-spacing: 0.08rem;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.4rem;
            ">
                <span>🚀</span> Quick Actions
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Regenerate Data", use_container_width=True, key="sidebar_regen"):
                st.session_state.data = generate_synthetic_data(
                    n_points=data_points,
                    anomaly_rate=anomaly_rate
                )
                st.session_state.multivariate_data = generate_multivariate_data(n_points=data_points)
                st.session_state.stat_detector = StatisticalAnomalyDetector(z_threshold=z_threshold)
                st.session_state.ml_detector = MLAnomalyDetector(contamination=contamination)
                st.session_state.drift_detector = ModelDriftDetector(drift_threshold=drift_threshold)
                st.session_state.active_dataset_label = 'Synthetic Stream'
                st.session_state.last_update = datetime.now()
                st.rerun()
            
            if st.button("📊 Run Full Analysis", use_container_width=True, key="sidebar_analyze"):
                with st.spinner("Analyzing..."):
                    run_full_analysis()
                st.session_state.last_update = datetime.now()
                st.success("✓ Analysis complete!")
            
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            
            # Current Configuration Display
            st.markdown("""
            <div style="
                font-family: 'Space Grotesk', sans-serif;
                font-size: 0.7rem;
                font-weight: 600;
                color: #00aaff;
                text-transform: uppercase;
                letter-spacing: 0.08rem;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.4rem;
            ">
                <span>📋</span> Current Config
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="
                background: rgba(0,0,0,0.2);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 8px;
                padding: 0.75rem;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
            ">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="color: #888;">Data Points</span>
                    <span style="color: #fff;">{data_points:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="color: #888;">Anomaly Rate</span>
                    <span style="color: #fff;">{anomaly_rate:.0%}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="color: #888;">Z-Threshold</span>
                    <span style="color: #fff;">{z_threshold:.1f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="color: #888;">Contamination</span>
                    <span style="color: #fff;">{contamination:.0%}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #888;">Drift Threshold</span>
                    <span style="color: #fff;">{drift_threshold:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            
            # Last Update
            st.markdown(f"""
            <div style="
                text-align: center;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.65rem;
                color: #666;
            ">
                Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
        
        # ============ MODELS TAB ============
        else:  # Models tab
            # Get current config values for consistency
            data_points = st.session_state.current_config.get('data_points', 500)
            anomaly_rate = st.session_state.current_config.get('anomaly_rate', 0.05)
            z_threshold = st.session_state.current_config.get('z_threshold', 3.0)
            contamination = st.session_state.current_config.get('contamination', 0.1)
            drift_threshold = st.session_state.current_config.get('drift_threshold', 0.1)
            
            st.markdown("""
            <div style="
                font-family: 'Space Grotesk', sans-serif;
                font-size: 0.7rem;
                font-weight: 600;
                color: #00e5ff;
                text-transform: uppercase;
                letter-spacing: 0.08rem;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.4rem;
            ">
                <span>🤖</span> Active Detection Models
            </div>
            """, unsafe_allow_html=True)
            
            # Statistical Models Card
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(0,170,255,0.1), rgba(0,170,255,0.03));
                border: 1px solid rgba(0,170,255,0.2);
                border-radius: 10px;
                padding: 0.75rem;
                margin-bottom: 0.6rem;
            ">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.7rem; color: #00aaff; text-transform: uppercase; font-weight: 600;">Statistical</span>
                    <span style="font-size: 0.6rem; padding: 2px 6px; background: rgba(0,255,136,0.15); color: #00ff88; border-radius: 4px;">ACTIVE</span>
                </div>
                <div style="font-size: 0.75rem; color: #c0c0d0; line-height: 1.4;">
                    Z-Score • IQR • MAD • Grubbs Test
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # ML Models Card
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(170,102,255,0.1), rgba(170,102,255,0.03));
                border: 1px solid rgba(170,102,255,0.2);
                border-radius: 10px;
                padding: 0.75rem;
                margin-bottom: 0.6rem;
            ">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.7rem; color: #aa66ff; text-transform: uppercase; font-weight: 600;">Machine Learning</span>
                    <span style="font-size: 0.6rem; padding: 2px 6px; background: rgba(0,255,136,0.15); color: #00ff88; border-radius: 4px;">ACTIVE</span>
                </div>
                <div style="font-size: 0.75rem; color: #c0c0d0; line-height: 1.4;">
                    Isolation Forest • LOF • One-Class SVM • Elliptic Envelope
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Deep Learning Card
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,255,136,0.03));
                border: 1px solid rgba(0,255,136,0.2);
                border-radius: 10px;
                padding: 0.75rem;
                margin-bottom: 0.6rem;
            ">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.7rem; color: #00ff88; text-transform: uppercase; font-weight: 600;">Deep Learning</span>
                    <span style="font-size: 0.6rem; padding: 2px 6px; background: rgba(0,255,136,0.15); color: #00ff88; border-radius: 4px;">ACTIVE</span>
                </div>
                <div style="font-size: 0.75rem; color: #c0c0d0; line-height: 1.4;">
                    Autoencoder (Reconstruction Error)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Drift Detection Card
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(255,170,0,0.1), rgba(255,170,0,0.03));
                border: 1px solid rgba(255,170,0,0.2);
                border-radius: 10px;
                padding: 0.75rem;
                margin-bottom: 0.6rem;
            ">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.7rem; color: #ffaa00; text-transform: uppercase; font-weight: 600;">Drift Detection</span>
                    <span style="font-size: 0.6rem; padding: 2px 6px; background: rgba(0,255,136,0.15); color: #00ff88; border-radius: 4px;">ACTIVE</span>
                </div>
                <div style="font-size: 0.75rem; color: #c0c0d0; line-height: 1.4;">
                    PSI • KS Test • Mean/Var Shift
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Regime Detection Card
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(255,51,102,0.1), rgba(255,51,102,0.03));
                border: 1px solid rgba(255,51,102,0.2);
                border-radius: 10px;
                padding: 0.75rem;
            ">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.7rem; color: #ff3366; text-transform: uppercase; font-weight: 600;">Regime Analysis</span>
                    <span style="font-size: 0.6rem; padding: 2px 6px; background: rgba(0,255,136,0.15); color: #00ff88; border-radius: 4px;">ACTIVE</span>
                </div>
                <div style="font-size: 0.75rem; color: #c0c0d0; line-height: 1.4;">
                    Gaussian Mixture Model • HMM
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Footer (always visible)
        st.markdown("""
        <div style="margin-top: 1.5rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.05); text-align: center;">
            <p style="font-size: 0.6rem; color: #555; margin: 0;">Built with Streamlit • Plotly • scikit-learn</p>
        </div>
        """, unsafe_allow_html=True)

    # Sync detector thresholds with sidebar selections
    st.session_state.stat_detector.z_threshold = z_threshold
    st.session_state.ml_detector.contamination = contamination
    st.session_state.drift_detector.drift_threshold = drift_threshold
    
    # Main content
    tab_data, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📂 Data Hub",
        "📈 Anomaly Detection",
        "🔄 Model Drift",
        "📊 Regime Analysis",
        "⚠️ Alerts",
        "📋 Reports"
    ])

    with tab_data:
        render_data_hub_tab()
    
    with tab1:
        render_anomaly_detection_tab()
    
    with tab2:
        render_drift_detection_tab()
    
    with tab3:
        render_regime_analysis_tab()
    
    with tab4:
        render_alerts_tab()
    
    with tab5:
        render_reports_tab()


def run_full_analysis():
    """Run complete anomaly detection analysis"""
    data = st.session_state.data
    values = data['value'].values
    
    # Fit ML models
    st.session_state.ml_detector.fit(values)
    st.session_state.deep_detector.fit(values)
    
    # Set drift reference
    st.session_state.drift_detector.set_reference(values[:len(values)//2])
    
    # Fit regime detector
    mv_data = st.session_state.multivariate_data
    st.session_state.regime_detector.fit(mv_data['price'].values)


def render_anomaly_detection_tab():
    """Render anomaly detection tab with enhanced cards"""
    st.markdown('<h2 class="section-title">🎯 Multi-Method Anomaly Detection</h2>', unsafe_allow_html=True)
    
    data = st.session_state.data
    values = data['value'].values
    
    # Run detection methods
    col1, col2, col3, col4 = st.columns(4)
    
    # Statistical detection
    stat_scores = []
    stat_anomalies = []
    for val in values:
        is_anomaly, score, _ = st.session_state.stat_detector.detect(val)
        stat_scores.append(score)
        stat_anomalies.append(1 if is_anomaly else 0)
    stat_scores = np.array(stat_scores)
    stat_anomalies = np.array(stat_anomalies)
    
    # ML detection
    if st.session_state.ml_detector.fitted:
        ml_anomalies, ml_scores = st.session_state.ml_detector.detect(values)
    else:
        st.session_state.ml_detector.fit(values)
        ml_anomalies, ml_scores = st.session_state.ml_detector.detect(values)
    
    # Deep detection
    deep_anomalies, deep_scores = st.session_state.deep_detector.detect(values)
    
    # Ensemble
    ensemble_scores = 0.35 * stat_scores + 0.35 * ml_scores + 0.3 * deep_scores
    ensemble_anomalies = (ensemble_scores > 0.5).astype(int)
    
    # Summary Banner
    total_ensemble = int(np.sum(ensemble_anomalies))
    avg_score = float(np.mean(ensemble_scores))
    max_score = float(np.max(ensemble_scores))
    
    status_icon = "⚠️" if total_ensemble > 10 else "✅"
    status_text = "Elevated Anomaly Activity" if total_ensemble > 10 else "System Operating Normally"
    banner_bg = "linear-gradient(135deg, rgba(255,170,0,0.1), rgba(255,51,102,0.1))" if total_ensemble > 10 else "linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,170,255,0.1))"
    banner_border = "rgba(255,170,0,0.2)" if total_ensemble > 10 else "rgba(0,255,136,0.2)"
    
    b_left, b_right = st.columns([2, 1])
    with b_left:
        st.markdown(f"""
        <div style="background: {banner_bg}; border: 1px solid {banner_border};
                    border-radius: 16px; padding: 1.25rem 1.5rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 2rem;">{status_icon}</span>
                <div>
                    <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600;
                               color: #ffffff; margin: 0 0 0.25rem 0;">{status_text}</h3>
                    <p style="font-size: 0.85rem; color: #a0a0b0; margin: 0;">Ensemble detection across {len(values):,} data points</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with b_right:
        st.markdown(f"""
        <div style="display: flex; gap: 1.5rem; justify-content: center; align-items: center; height: 100%;">
            <div style="text-align: center;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #ffffff;">{total_ensemble}</div>
                <div style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0;">Anomalies</div>
            </div>
            <div style="text-align: center;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #ffffff;">{avg_score:.3f}</div>
                <div style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0;">Avg Score</div>
            </div>
            <div style="text-align: center;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #ffffff;">{max_score:.3f}</div>
                <div style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0;">Max Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detection method cards using st.columns
    stat_count = int(np.sum(stat_anomalies))
    ml_count = int(np.sum(ml_anomalies))
    deep_count = int(np.sum(deep_anomalies))
    ensemble_count = int(np.sum(ensemble_anomalies))
    
    def get_color(count):
        if count > 10: return "#ff3366"
        elif count > 5: return "#ffaa00"
        return "#00ff88"
    
    card_bg = "background: linear-gradient(145deg, #1a1a25, #12121a); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);"
    label_s = "font-family: 'Space Grotesk', sans-serif; color: #a0a0b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;"
    value_s = "font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; margin: 0.5rem 0;"
    meta_s = "font-size: 0.8rem; color: #888;"
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f'<div style="{card_bg}"><div style="{label_s}">Statistical Anomalies</div><div style="{value_s} color: {get_color(stat_count)};">{stat_count}</div><div style="{meta_s}">Z-Score + IQR + MAD + Grubbs</div></div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div style="{card_bg}"><div style="{label_s}">ML Anomalies</div><div style="{value_s} color: {get_color(ml_count)};">{ml_count}</div><div style="{meta_s}">Isolation Forest + LOF + SVM</div></div>', unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div style="{card_bg}"><div style="{label_s}">Deep Learning Anomalies</div><div style="{value_s} color: {get_color(deep_count)};">{deep_count}</div><div style="{meta_s}">Autoencoder Reconstruction</div></div>', unsafe_allow_html=True)
    with mc4:
        st.markdown(f'<div style="{card_bg}"><div style="{label_s}">Ensemble Anomalies</div><div style="{value_s} color: {get_color(ensemble_count)};">{ensemble_count}</div><div style="{meta_s}">Weighted Combination</div></div>', unsafe_allow_html=True)
    
    # Health indicators using st.columns
    detection_rate = ensemble_count / len(values) * 100 if len(values) > 0 else 0
    coverage = len([s for s in ensemble_scores if s > 0.3]) / len(values) * 100
    agreement = 1 - abs(stat_count - ml_count) / max(stat_count, ml_count, 1)
    
    health_status_1 = "good" if detection_rate < 10 else "warning" if detection_rate < 20 else "critical"
    health_label_1 = "Normal" if detection_rate < 10 else "Elevated" if detection_rate < 20 else "High"
    health_status_2 = "good" if agreement > 0.7 else "warning"
    health_label_2 = "High" if agreement > 0.7 else "Moderate"
    
    status_colors = {"good": ("#00ff88", "rgba(0,255,136,0.15)"), "warning": ("#ffaa00", "rgba(255,170,0,0.15)"), "critical": ("#ff3366", "rgba(255,51,102,0.15)")}
    health_bg = "background: linear-gradient(145deg, #1a1a25, rgba(0,255,136,0.05)); border: 1px solid rgba(0,255,136,0.15); border-radius: 14px; padding: 1rem;"
    
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        sc, sb = status_colors[health_status_1]
        st.markdown(f"""
        <div style="{health_bg}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">🎯</span>
                <div style="flex: 1;">
                    <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05rem; color: #a0a0b0;">Detection Rate</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #ffffff;">{detection_rate:.1f}%</div>
                </div>
                <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: {sb}; color: {sc};">{health_label_1}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with hc2:
        st.markdown(f"""
        <div style="{health_bg}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">📊</span>
                <div style="flex: 1;">
                    <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05rem; color: #a0a0b0;">Score Coverage</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #ffffff;">{coverage:.1f}%</div>
                </div>
                <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: rgba(0,255,136,0.15); color: #00ff88;">Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with hc3:
        sc2, sb2 = status_colors[health_status_2]
        st.markdown(f"""
        <div style="{health_bg}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 1.5rem;">🤝</span>
                <div style="flex: 1;">
                    <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05rem; color: #a0a0b0;">Model Agreement</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #ffffff;">{agreement * 100:.1f}%</div>
                </div>
                <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: {sb2}; color: {sc2};">{health_label_2}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main visualization
    method = st.selectbox(
        "Select Detection Method",
        ["Ensemble", "Statistical", "Machine Learning", "Deep Learning"]
    )
    
    if method == "Ensemble":
        scores, anomalies = ensemble_scores, ensemble_anomalies
    elif method == "Statistical":
        scores, anomalies = stat_scores, stat_anomalies
    elif method == "Machine Learning":
        scores, anomalies = ml_scores, ml_anomalies
    else:
        scores, anomalies = deep_scores, deep_anomalies
    
    fig = create_time_series_chart(data, scores, anomalies, f"{method} Anomaly Detection")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detection methods comparison
    st.markdown('<h3 class="section-title">📊 Methods Comparison</h3>', unsafe_allow_html=True)
    
    methods_summary = {
        'Statistical': {'detected': np.sum(stat_anomalies), 'score': np.mean(stat_scores)},
        'ML Ensemble': {'detected': np.sum(ml_anomalies), 'score': np.mean(ml_scores)},
        'Deep Learning': {'detected': np.sum(deep_anomalies), 'score': np.mean(deep_scores)},
        'Combined': {'detected': np.sum(ensemble_anomalies), 'score': np.mean(ensemble_scores)}
    }
    
    max_score = float(np.max(ensemble_scores))
    
    fig_summary = create_detection_summary_chart(methods_summary)
    st.plotly_chart(fig_summary, use_container_width=True)
    
    # Insights Panel - rendered as individual items for reliability
    insights = []
    if stat_count > ml_count * 1.5:
        insights.append(("📈", "Statistical methods detecting more anomalies — may indicate distribution shifts"))
    if ml_count > stat_count * 1.5:
        insights.append(("🤖", "ML methods detecting more anomalies — may indicate complex patterns"))
    if ensemble_count < 5:
        insights.append(("✅", "Low anomaly count suggests stable data quality"))
    if max_score > 0.9:
        insights.append(("🚨", "High-confidence anomalies detected requiring attention"))
    if not insights:
        insights.append(("📊", "Detection patterns are within normal parameters"))
    
    st.markdown("""
    <div style="background: linear-gradient(160deg, rgba(0,170,255,0.08), rgba(170,102,255,0.08));
                border: 1px solid rgba(0,170,255,0.15); border-radius: 16px;
                padding: 1.25rem; margin: 1rem 0;">
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; font-weight: 600;
                    color: #00aaff; margin-bottom: 0.75rem;">💡 Detection Insights</div>
    """, unsafe_allow_html=True)
    
    for icon, text in insights:
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; gap: 0.75rem; padding: 0.5rem 0;
                    border-bottom: 1px solid rgba(255,255,255,0.05);">
            <span style="font-size: 1rem;">{icon}</span>
            <span style="font-size: 0.85rem; color: #a0a0b0; line-height: 1.4;">{text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed statistics
    with st.expander("📋 Detailed Detection Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Statistical Method Details**")
            st.json({
                "Total Anomalies": int(np.sum(stat_anomalies)),
                "Mean Score": float(np.mean(stat_scores)),
                "Max Score": float(np.max(stat_scores)),
                "Score Std": float(np.std(stat_scores))
            })
        
        with col2:
            st.markdown("**ML Method Details**")
            st.json({
                "Isolation Forest": int(np.sum(ml_anomalies)),
                "Ensemble Score": float(np.mean(ml_scores)),
                "Detection Rate": f"{100 * np.sum(ml_anomalies) / len(values):.2f}%"
            })

    # Highest Impact Signals with per-card rendering
    st.markdown('<h3 class="section-title">🚨 Highest Impact Signals</h3>', unsafe_allow_html=True)
    ranked_idx = np.argsort(scores)[-6:][::-1]
    ranked_idx = ranked_idx.tolist()
    highlighted_idx = [idx for idx in ranked_idx if anomalies[idx] > 0]
    if not highlighted_idx:
        highlighted_idx = ranked_idx[:4]
    highlighted_idx = highlighted_idx[:4]

    if highlighted_idx and len(scores) > 0 and np.max(scores) > 0:
        sig_cols = st.columns(len(highlighted_idx))
        for col_i, idx in enumerate(highlighted_idx):
            ts_value = data['timestamp'].iloc[idx]
            val = data['value'].iloc[idx]
            score_val = scores[idx]
            is_anom = anomalies[idx] > 0
            
            top_color = "linear-gradient(90deg, #ff3366, #ffaa00)" if is_anom else "linear-gradient(90deg, #ffaa00, #00aaff)"
            badge_bg = "linear-gradient(135deg, rgba(255,51,102,0.25), rgba(255,51,102,0.1))" if is_anom else "linear-gradient(135deg, rgba(255,170,0,0.25), rgba(255,170,0,0.1))"
            badge_color = "#ff3366" if is_anom else "#ffaa00"
            badge_border = "rgba(255,51,102,0.3)" if is_anom else "rgba(255,170,0,0.3)"
            badge_label = "Anomaly" if is_anom else "Candidate"
            val_gradient = "linear-gradient(135deg, #ff3366, #ffaa00)" if is_anom else "linear-gradient(135deg, #ffffff, #00aaff)"
            score_pct = min(score_val * 100, 100)
            score_color = "#ff3366" if is_anom else "#00aaff"
            bar_gradient = "linear-gradient(90deg, #ffaa00, #ff3366)" if is_anom else "linear-gradient(90deg, #00aaff, #aa66ff)"
            
            with sig_cols[col_i]:
                st.markdown(f"""
                <div style="background: linear-gradient(160deg, rgba(26,26,37,0.95), rgba(0,170,255,0.08));
                            border: 1px solid rgba(255,255,255,0.1); border-radius: 20px;
                            padding: 1.5rem; position: relative; overflow: hidden;
                            box-shadow: 0 15px 35px rgba(0,0,0,0.4); min-height: 200px;">
                    <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px;
                                background: {top_color}; border-radius: 20px 20px 0 0;"></div>
                    <span style="display: inline-block; padding: 6px 14px; border-radius: 20px;
                                 font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08rem;
                                 text-transform: uppercase; margin-bottom: 0.75rem;
                                 background: {badge_bg}; color: {badge_color};
                                 border: 1px solid {badge_border};">{badge_label}</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
                                color: #a0a0b0; margin-bottom: 0.5rem;">🕐 {ts_value.strftime('%Y-%m-%d %H:%M')}</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.2rem;
                                font-weight: 700; background: {val_gradient};
                                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                                margin: 0.5rem 0; line-height: 1.2;">{val:.2f}</div>
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: auto;
                                padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.08);">
                        <span style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05rem;
                                     color: #a0a0b0;">Score</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600;
                                     color: {score_color}; background: rgba(0,170,255,0.1);
                                     padding: 4px 10px; border-radius: 8px;">{score_val:.3f}</span>
                        <div style="flex: 1; height: 6px; background: rgba(255,255,255,0.1);
                                    border-radius: 3px; overflow: hidden;">
                            <div style="height: 100%; border-radius: 3px; background: {bar_gradient};
                                        width: {score_pct}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Detection scores are stable; no standout signals to highlight right now.")


def render_drift_detection_tab():
    """Render model drift detection tab"""
    st.markdown('<h2 class="section-title">🔄 Model Drift Detection</h2>', unsafe_allow_html=True)
    
    data = st.session_state.data
    values = data['value'].values
    
    # Set reference if not set
    if st.session_state.drift_detector.reference_data is None:
        st.session_state.drift_detector.set_reference(values[:len(values)//2])
    
    # Detect drift on recent data
    drift_detected, drift_score, details = st.session_state.drift_detector.detect_drift(values)
    
    # Metrics using st.columns
    col1, col2, col3, col4 = st.columns(4)
    
    drift_card = "background: linear-gradient(145deg, #1a1a25, #12121a); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);"
    drift_label = "font-family: 'Space Grotesk', sans-serif; color: #a0a0b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;"
    
    with col1:
        status = "DRIFT DETECTED" if drift_detected else "STABLE"
        scolor = "#ff3366" if drift_detected else "#00ff88"
        st.markdown(f'<div style="{drift_card}"><div style="{drift_label}">Drift Status</div><div style="font-family: \'JetBrains Mono\', monospace; font-size: 1.3rem; font-weight: 700; margin: 0.5rem 0; color: {scolor};">{status}</div></div>', unsafe_allow_html=True)
    
    with col2:
        scolor = "#ff3366" if drift_score > 0.1 else "#ffaa00" if drift_score > 0.05 else "#00ff88"
        st.markdown(f'<div style="{drift_card}"><div style="{drift_label}">Drift Score</div><div style="font-family: \'JetBrains Mono\', monospace; font-size: 2rem; font-weight: 700; margin: 0.5rem 0; color: {scolor};">{drift_score:.3f}</div></div>', unsafe_allow_html=True)
    
    with col3:
        ks_stat = details.get('ks_statistic', 0)
        scolor = "#ff3366" if ks_stat > 0.2 else "#ffaa00" if ks_stat > 0.1 else "#00ff88"
        st.markdown(f'<div style="{drift_card}"><div style="{drift_label}">KS Statistic</div><div style="font-family: \'JetBrains Mono\', monospace; font-size: 2rem; font-weight: 700; margin: 0.5rem 0; color: {scolor};">{ks_stat:.3f}</div></div>', unsafe_allow_html=True)
    
    with col4:
        psi = details.get('psi', 0)
        scolor = "#ff3366" if psi > 0.25 else "#ffaa00" if psi > 0.1 else "#00ff88"
        st.markdown(f'<div style="{drift_card}"><div style="{drift_label}">PSI Score</div><div style="font-family: \'JetBrains Mono\', monospace; font-size: 2rem; font-weight: 700; margin: 0.5rem 0; color: {scolor};">{psi:.3f}</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Drift history chart
    fig = create_drift_chart(st.session_state.drift_detector.drift_history)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution comparison
    st.markdown('<h3 class="section-title">📊 Distribution Comparison</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        ref_data = st.session_state.drift_detector.reference_data
        if ref_data is not None:
            fig_ref = go.Figure()
            fig_ref.add_trace(go.Histogram(
                x=ref_data,
                name='Reference',
                marker_color='#00aaff',
                opacity=0.7,
                nbinsx=30
            ))
            fig_ref.update_layout(
                title="Reference Distribution",
                template='plotly_dark',
                paper_bgcolor='rgba(10,10,15,0)',
                plot_bgcolor='rgba(26,26,37,1)',
                height=300
            )
            st.plotly_chart(fig_ref, use_container_width=True)
    
    with col2:
        current_data = values[-50:]
        fig_curr = go.Figure()
        fig_curr.add_trace(go.Histogram(
            x=current_data,
            name='Current',
            marker_color='#ff3366' if drift_detected else '#00ff88',
            opacity=0.7,
            nbinsx=30
        ))
        fig_curr.update_layout(
            title="Current Distribution",
            template='plotly_dark',
            paper_bgcolor='rgba(10,10,15,0)',
            plot_bgcolor='rgba(26,26,37,1)',
            height=300
        )
        st.plotly_chart(fig_curr, use_container_width=True)
    
    # Detailed drift metrics
    with st.expander("📋 Detailed Drift Metrics"):
        st.json({
            "KS Statistic": details.get('ks_statistic', 0),
            "KS P-Value": details.get('ks_pvalue', 0),
            "PSI Score": details.get('psi', 0),
            "JS Divergence": details.get('js_divergence', 0),
            "Mean Shift (σ)": details.get('mean_shift', 0),
            "Variance Drift": details.get('variance_drift', 0),
            "Current Mean": details.get('current_mean', 0),
            "Reference Mean": details.get('reference_mean', 0)
        })


def render_regime_analysis_tab():
    """Render market regime analysis tab"""
    st.markdown('<h2 class="section-title">📊 Market Regime Analysis</h2>', unsafe_allow_html=True)
    
    mv_data = st.session_state.multivariate_data
    
    # Fit regime detector if not fitted
    if not st.session_state.regime_detector.fitted:
        st.session_state.regime_detector.fit(mv_data['price'].values)
    
    # Detect current regime
    regime_info = st.session_state.regime_detector.detect_regime(mv_data['price'].values)
    
    # Metrics using st.columns
    col1, col2, col3, col4 = st.columns(4)
    
    regime_card = "background: linear-gradient(145deg, #1a1a25, #12121a); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);"
    regime_label = "font-family: 'Space Grotesk', sans-serif; color: #a0a0b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;"
    regime_val = "font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;"
    
    if regime_info:
        with col1:
            regime_name = regime_info['name']
            regime_color = regime_info['color']
            st.markdown(f'<div style="{regime_card}"><div style="{regime_label}">Current Regime</div><div style="{regime_val} color: {regime_color};">{regime_name}</div></div>', unsafe_allow_html=True)
        
        with col2:
            confidence = regime_info['confidence']
            ccolor = "#00ff88" if confidence > 0.7 else "#ffaa00"
            st.markdown(f'<div style="{regime_card}"><div style="{regime_label}">Confidence</div><div style="{regime_val} color: {ccolor};">{confidence:.1%}</div></div>', unsafe_allow_html=True)
        
        with col3:
            volatility = mv_data['volatility'].iloc[-1]
            vcolor = "#ff3366" if volatility > 0.02 else "#ffaa00" if volatility > 0.01 else "#00ff88"
            st.markdown(f'<div style="{regime_card}"><div style="{regime_label}">Current Volatility</div><div style="{regime_val} color: {vcolor};">{volatility:.2%}</div></div>', unsafe_allow_html=True)
        
        with col4:
            momentum = mv_data['momentum'].iloc[-1]
            mcolor = "#00ff88" if momentum > 0 else "#ff3366"
            st.markdown(f'<div style="{regime_card}"><div style="{regime_label}">Momentum</div><div style="{regime_val} color: {mcolor};">{momentum:.2%}</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Regime chart
    fig = create_regime_chart(mv_data, regime_info)
    st.plotly_chart(fig, use_container_width=True)
    
    # Regime probabilities
    if regime_info and 'probabilities' in regime_info:
        st.markdown('<h3 class="section-title">🎯 Regime Probabilities</h3>', unsafe_allow_html=True)
        
        probs = regime_info['probabilities']
        regime_names = ['Bullish', 'Bearish', 'Consolidation', 'High Volatility']
        colors = ['#00ff88', '#ff3366', '#00aaff', '#ffaa00']
        
        cols = st.columns(4)
        for i, (col, name, prob, color) in enumerate(zip(cols, regime_names, probs, colors)):
            with col:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
                    <div style="font-size: 0.8rem; color: {color}; text-transform: uppercase;">{name}</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Regime transition matrix
    with st.expander("📋 Regime Analysis Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Regime Statistics**")
            regime_counts = mv_data['regime'].value_counts().to_dict()
            regime_names = {0: 'Bullish', 1: 'Bearish', 2: 'Consolidation', 3: 'High Vol'}
            
            for regime, count in regime_counts.items():
                st.write(f"- {regime_names.get(regime, f'Regime {regime}')}: {count} periods ({100*count/len(mv_data):.1f}%)")
        
        with col2:
            st.markdown("**Recent Regime Transitions**")
            regimes = mv_data['regime'].values
            transitions = []
            for i in range(1, min(10, len(regimes))):
                if regimes[-i] != regimes[-i-1]:
                    transitions.append(f"{regime_names.get(regimes[-i-1], '?')} → {regime_names.get(regimes[-i], '?')}")
            
            if transitions:
                for t in transitions[:5]:
                    st.write(f"- {t}")
            else:
                st.write("No recent transitions")


def render_alerts_tab():
    """Render alerts tab"""
    st.markdown('<h2 class="section-title">⚠️ Anomaly Alerts</h2>', unsafe_allow_html=True)
    
    alert_manager = st.session_state.alert_manager
    
    # Generate some sample alerts based on current analysis
    data = st.session_state.data
    values = data['value'].values
    
    # Check for anomalies and add alerts
    for i in range(-5, 0):
        val = values[i]
        is_anomaly, score, details = st.session_state.stat_detector.detect(val)
        
        if is_anomaly:
            severity = 'critical' if score > 0.8 else 'warning'
            alert_manager.add_alert(
                alert_type='statistical_anomaly',
                severity=severity,
                message=f"Statistical anomaly detected: value={val:.2f}, score={score:.3f}",
                details=details
            )
    
    # Check drift
    drift_detected, drift_score, drift_details = st.session_state.drift_detector.detect_drift(values)
    if drift_detected:
        alert_manager.add_alert(
            alert_type='model_drift',
            severity='critical',
            message=f"Model drift detected: score={drift_score:.3f}",
            details=drift_details
        )
    
    # Summary metrics using st.columns
    col1, col2, col3 = st.columns(3)
    
    alert_summary = alert_manager.get_alert_summary()
    total_alerts = sum(alert_summary.values())
    alert_card = "background: linear-gradient(145deg, #1a1a25, #12121a); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);"
    alert_label = "font-family: 'Space Grotesk', sans-serif; color: #a0a0b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;"
    alert_val = "font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; margin: 0.5rem 0;"
    
    with col1:
        critical_count = len([a for a in alert_manager.get_recent_alerts(100) if a['severity'] == 'critical'])
        st.markdown(f'<div style="{alert_card}"><div style="{alert_label}">Critical Alerts</div><div style="{alert_val} color: #ff3366;">{critical_count}</div></div>', unsafe_allow_html=True)
    
    with col2:
        warning_count = len([a for a in alert_manager.get_recent_alerts(100) if a['severity'] == 'warning'])
        st.markdown(f'<div style="{alert_card}"><div style="{alert_label}">Warnings</div><div style="{alert_val} color: #ffaa00;">{warning_count}</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div style="{alert_card}"><div style="{alert_label}">Total Alerts</div><div style="{alert_val} color: #ffffff;">{total_alerts}</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recent alerts
    st.markdown('<h3 class="section-title">📋 Recent Alerts</h3>', unsafe_allow_html=True)
    
    alerts = alert_manager.get_recent_alerts(20)
    
    if alerts:
        for alert in alerts:
            severity_class = 'alert-box' if alert['severity'] == 'critical' else 'alert-box warning'
            icon = '🚨' if alert['severity'] == 'critical' else '⚠️'
            
            st.markdown(f"""
            <div class="{severity_class}">
                <strong>{icon} {alert['type'].upper()}</strong> - {alert['timestamp'].strftime('%H:%M:%S')}<br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent alerts")
    
    # Alert configuration
    with st.expander("⚙️ Alert Configuration"):
        st.markdown("**Threshold Settings**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Anomaly Score Threshold", 0.0, 1.0, 0.5, 0.1, key="alert_anomaly_thresh")
            st.number_input("Drift Score Threshold", 0.0, 0.5, 0.1, 0.05, key="alert_drift_thresh")
        
        with col2:
            st.selectbox("Alert Severity Filter", ["All", "Critical Only", "Warnings Only"], key="alert_severity_filter")
            st.number_input("Max Alerts to Display", 5, 100, 20, key="alert_max_display")


def render_reports_tab():
    """Render reports tab"""
    st.markdown('<h2 class="section-title">📋 Analysis Reports</h2>', unsafe_allow_html=True)
    
    data = st.session_state.data
    values = data['value'].values
    mv_data = st.session_state.multivariate_data
    
    # Generate comprehensive report
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Executive Summary")
        
        # Anomaly summary
        stat_scores = []
        stat_anomalies = []
        for val in values:
            is_anomaly, score, _ = st.session_state.stat_detector.detect(val)
            stat_scores.append(score)
            stat_anomalies.append(1 if is_anomaly else 0)
        
        total_anomalies = sum(stat_anomalies)
        anomaly_rate = total_anomalies / len(values)
        
        # Drift summary
        drift_detected, drift_score, _ = st.session_state.drift_detector.detect_drift(values)
        
        # Regime summary
        regime_info = st.session_state.regime_detector.detect_regime(mv_data['price'].values)
        
        analysis_window = f"{data['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} → {data['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}"
        drift_tag = 'tag-critical' if drift_detected else 'tag-active'
        drift_label = 'Drift detected' if drift_detected else 'Stable'
        regime_name = regime_info['name'] if regime_info else 'Unknown'
        regime_confidence = regime_info['confidence'] if regime_info else 0

        # Summary cards using st.columns for reliable rendering
        sc1, sc2, sc3, sc4 = st.columns(4)
        
        rpt_card = "background: linear-gradient(160deg, rgba(26,26,37,0.95), rgba(0,170,255,0.08)); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 1.5rem; position: relative; overflow: hidden; box-shadow: 0 15px 35px rgba(0,0,0,0.4); min-height: 180px;"
        rpt_ts = "font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #a0a0b0; margin-bottom: 0.5rem;"
        rpt_val = "font-family: 'JetBrains Mono', monospace; font-weight: 700; margin: 0.5rem 0; line-height: 1.2;"
        
        with sc1:
            st.markdown(f"""
            <div style="{rpt_card}">
                <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #00ff88, #00e5ff); border-radius: 20px 20px 0 0;"></div>
                <span style="display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08rem; text-transform: uppercase; margin-bottom: 0.75rem; background: linear-gradient(135deg, rgba(0,170,255,0.25), rgba(0,170,255,0.1)); color: #00aaff; border: 1px solid rgba(0,170,255,0.3);">Scope</span>
                <div style="{rpt_ts}">{data['timestamp'].iloc[0].strftime('%Y-%m-%d')} → {data['timestamp'].iloc[-1].strftime('%Y-%m-%d')}</div>
                <div style="{rpt_val} font-size: 1.6rem; background: linear-gradient(135deg, #00ff88, #00e5ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{len(values):,}</div>
                <div style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0; margin-top: auto; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.08);">Points Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with sc2:
            anom_color = "#ff3366" if total_anomalies > 10 else "#ffaa00" if total_anomalies > 0 else "#00ff88"
            anom_bar = "linear-gradient(90deg, #ff3366, #ffaa00)" if total_anomalies > 10 else "linear-gradient(90deg, #ffaa00, #00aaff)"
            st.markdown(f"""
            <div style="{rpt_card}">
                <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: {anom_bar}; border-radius: 20px 20px 0 0;"></div>
                <span style="display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08rem; text-transform: uppercase; margin-bottom: 0.75rem; background: rgba(255,170,0,0.15); color: {anom_color}; border: 1px solid rgba(255,170,0,0.3);">Anomalies</span>
                <div style="{rpt_ts}">Detection Summary</div>
                <div style="{rpt_val} font-size: 2.2rem; color: {anom_color};">{total_anomalies:,}</div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: auto; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.08);">
                    <span style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0;">Rate</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; font-weight: 600; color: #00aaff; background: rgba(0,170,255,0.1); padding: 2px 8px; border-radius: 6px;">{anomaly_rate:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with sc3:
            drift_color = "#ff3366" if drift_detected else "#00ff88"
            drift_bar_top = "linear-gradient(90deg, #ff3366, #ffaa00)" if drift_detected else "linear-gradient(90deg, #00ff88, #00e5ff)"
            st.markdown(f"""
            <div style="{rpt_card}">
                <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: {drift_bar_top}; border-radius: 20px 20px 0 0;"></div>
                <span style="display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08rem; text-transform: uppercase; margin-bottom: 0.75rem; background: rgba(0,255,136,0.15); color: {drift_color}; border: 1px solid rgba(0,255,136,0.3);">Drift</span>
                <div style="{rpt_ts}">Model Health</div>
                <div style="{rpt_val} font-size: 1.4rem; color: {drift_color};">{drift_label}</div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: auto; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.08);">
                    <span style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0;">Score</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; font-weight: 600; color: #00aaff; background: rgba(0,170,255,0.1); padding: 2px 8px; border-radius: 6px;">{drift_score:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with sc4:
            st.markdown(f"""
            <div style="{rpt_card}">
                <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #aa66ff, #00aaff); border-radius: 20px 20px 0 0;"></div>
                <span style="display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08rem; text-transform: uppercase; margin-bottom: 0.75rem; background: rgba(170,102,255,0.15); color: #aa66ff; border: 1px solid rgba(170,102,255,0.3);">Regime</span>
                <div style="{rpt_ts}">Current State</div>
                <div style="{rpt_val} font-size: 1.2rem; color: #aa66ff;">{regime_name}</div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: auto; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.08);">
                    <span style="font-size: 0.7rem; text-transform: uppercase; color: #a0a0b0;">Confidence</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; font-weight: 600; color: #aa66ff; background: rgba(170,102,255,0.1); padding: 2px 8px; border-radius: 6px;">{regime_confidence:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Data quality metrics using st.columns
        missing = data.isnull().sum().sum()
        duplicates = data.duplicated().sum()
        data_range = values.max() - values.min()

        st.markdown("### 📈 Data Quality Pulse")
        dq1, dq2, dq3 = st.columns(3)
        
        stat_card_s = "background: linear-gradient(160deg, rgba(26,26,37,0.9), rgba(170,102,255,0.08)); border: 1px solid rgba(170,102,255,0.15); border-radius: 18px; padding: 1.25rem; box-shadow: 0 12px 30px rgba(0,0,0,0.35);"
        stat_lbl = "font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06rem; color: #a0a0b0;"
        stat_val_s = "font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #ffffff; margin: 0.3rem 0;"
        stat_meta = "font-size: 0.8rem; color: #a0a0b0; line-height: 1.4;"
        
        with dq1:
            badge_c = "#00ff88" if missing == 0 else "#ffaa00"
            badge_l = "Clean" if missing == 0 else "Check"
            st.markdown(f"""
            <div style="{stat_card_s}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">📊</span>
                    <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: rgba(0,255,136,0.15); color: {badge_c};">{badge_l}</span>
                </div>
                <div style="{stat_lbl}">Missing Values</div>
                <div style="{stat_val_s}">{int(missing):,}</div>
                <div style="{stat_meta}">Total null entries across the dataset.</div>
            </div>
            """, unsafe_allow_html=True)
        with dq2:
            badge_c = "#00ff88" if duplicates == 0 else "#ffaa00"
            badge_l = "Unique" if duplicates == 0 else "Review"
            st.markdown(f"""
            <div style="{stat_card_s}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">🔄</span>
                    <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: rgba(0,255,136,0.15); color: {badge_c};">{badge_l}</span>
                </div>
                <div style="{stat_lbl}">Duplicate Rows</div>
                <div style="{stat_val_s}">{int(duplicates):,}</div>
                <div style="{stat_meta}">Potential repeated observations detected.</div>
            </div>
            """, unsafe_allow_html=True)
        with dq3:
            st.markdown(f"""
            <div style="{stat_card_s}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">📏</span>
                    <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: rgba(0,170,255,0.15); color: #00aaff;">Range</span>
                </div>
                <div style="{stat_lbl}">Value Spread</div>
                <div style="{stat_val_s}">{data_range:.2f}</div>
                <div style="{stat_meta}">Spread between min and max values.</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Quick Stats")
        mean_val = np.mean(values)
        std_val = np.std(values)
        skew_val = stats.skew(values)
        kurt_val = stats.kurtosis(values)

        qs1, qs2 = st.columns(2)
        with qs1:
            st.markdown(f"""
            <div style="{stat_card_s}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">📈</span>
                    <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: rgba(0,170,255,0.15); color: #00aaff;">Mean</span>
                </div>
                <div style="{stat_lbl}">Average Value</div>
                <div style="{stat_val_s}">{mean_val:.2f}</div>
                <div style="{stat_meta}">Central tendency over the analysis window.</div>
            </div>
            """, unsafe_allow_html=True)
        with qs2:
            st.markdown(f"""
            <div style="{stat_card_s}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">📊</span>
                    <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: rgba(170,102,255,0.15); color: #aa66ff;">Volatility</span>
                </div>
                <div style="{stat_lbl}">Std Deviation</div>
                <div style="{stat_val_s}">{std_val:.2f}</div>
                <div style="{stat_meta}">Signal volatility measure.</div>
            </div>
            """, unsafe_allow_html=True)
        
        qs3, qs4 = st.columns(2)
        with qs3:
            skew_badge_c = "#ffaa00" if abs(skew_val) > 1 else "#00ff88"
            skew_badge_l = "Skewed" if abs(skew_val) > 1 else "Normal"
            st.markdown(f"""
            <div style="{stat_card_s}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">↔️</span>
                    <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: rgba(0,255,136,0.15); color: {skew_badge_c};">{skew_badge_l}</span>
                </div>
                <div style="{stat_lbl}">Skewness</div>
                <div style="{stat_val_s}">{skew_val:.3f}</div>
                <div style="{stat_meta}">Distribution asymmetry indicator.</div>
            </div>
            """, unsafe_allow_html=True)
        with qs4:
            kurt_badge_c = "#ffaa00" if kurt_val > 3 else "#00ff88"
            kurt_badge_l = "Heavy" if kurt_val > 3 else "Normal"
            st.markdown(f"""
            <div style="{stat_card_s}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">🔔</span>
                    <span style="font-size: 0.65rem; padding: 3px 8px; border-radius: 8px; font-weight: 600; background: rgba(0,255,136,0.15); color: {kurt_badge_c};">{kurt_badge_l}</span>
                </div>
                <div style="{stat_lbl}">Kurtosis</div>
                <div style="{stat_val_s}">{kurt_val:.3f}</div>
                <div style="{stat_meta}">Tail heaviness indicator.</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed analysis sections
    st.markdown("### 🔍 Detailed Analysis")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "Statistical Summary",
        "Anomaly Breakdown",
        "Recommendations"
    ])
    
    with analysis_tab1:
        st.markdown("#### Distribution Analysis")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=50,
            marker_color='#00aaff',
            opacity=0.7,
            name='Distribution'
        ))
        
        # Add normal distribution overlay
        x_range = np.linspace(values.min(), values.max(), 100)
        normal_dist = stats.norm.pdf(x_range, np.mean(values), np.std(values))
        normal_dist = normal_dist * len(values) * (values.max() - values.min()) / 50
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#ff3366', width=2)
        ))
        
        fig.update_layout(
            title="Value Distribution with Normal Overlay",
            template='plotly_dark',
            paper_bgcolor='rgba(10,10,15,0)',
            plot_bgcolor='rgba(26,26,37,1)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical tests
        st.markdown("#### Normality Tests")
        shapiro_stat, shapiro_p = stats.shapiro(values[:min(5000, len(values))])
        ks_stat, ks_p = stats.kstest(values, 'norm', args=(np.mean(values), np.std(values)))
        
        test_results = pd.DataFrame({
            'Test': ['Shapiro-Wilk', 'Kolmogorov-Smirnov'],
            'Statistic': [shapiro_stat, ks_stat],
            'P-Value': [shapiro_p, ks_p],
            'Normal?': ['Yes' if shapiro_p > 0.05 else 'No', 'Yes' if ks_p > 0.05 else 'No']
        })
        st.dataframe(test_results, use_container_width=True)
    
    with analysis_tab2:
        st.markdown("#### Anomaly Distribution Over Time")
        
        anomaly_indices = np.where(np.array(stat_anomalies) == 1)[0]
        
        if len(anomaly_indices) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=anomaly_indices,
                nbinsx=20,
                marker_color='#ff3366',
                name='Anomalies'
            ))
            
            fig.update_layout(
                title="Temporal Distribution of Anomalies",
                xaxis_title="Time Index",
                yaxis_title="Count",
                template='plotly_dark',
                paper_bgcolor='rgba(10,10,15,0)',
                plot_bgcolor='rgba(26,26,37,1)',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly severity breakdown
            high_severity = sum(1 for s in stat_scores if s > 0.8)
            medium_severity = sum(1 for s in stat_scores if 0.5 < s <= 0.8)
            low_severity = sum(1 for s in stat_scores if 0.3 < s <= 0.5)
            
            severity_df = pd.DataFrame({
                'Severity': ['High (>0.8)', 'Medium (0.5-0.8)', 'Low (0.3-0.5)'],
                'Count': [high_severity, medium_severity, low_severity]
            })
            
            fig_sev = px.pie(
                severity_df,
                values='Count',
                names='Severity',
                color_discrete_sequence=['#ff3366', '#ffaa00', '#00aaff']
            )
            fig_sev.update_layout(
                title="Anomaly Severity Distribution",
                template='plotly_dark',
                paper_bgcolor='rgba(10,10,15,0)',
                height=300
            )
            st.plotly_chart(fig_sev, use_container_width=True)
        else:
            st.info("No anomalies detected in the current data.")
    
    with analysis_tab3:
        st.markdown("#### 💡 Recommendations")
        
        recommendations = []
        
        if anomaly_rate > 0.1:
            recommendations.append({
                'priority': 'High',
                'category': 'Anomaly Rate',
                'recommendation': 'High anomaly rate detected. Consider investigating data quality or adjusting detection thresholds.',
                'action': 'Review data sources and detection parameters'
            })
        
        if drift_detected:
            recommendations.append({
                'priority': 'High',
                'category': 'Model Drift',
                'recommendation': 'Model drift detected. Consider retraining models with recent data.',
                'action': 'Schedule model retraining'
            })
        
        if regime_info and regime_info['name'] == 'High Volatility':
            recommendations.append({
                'priority': 'Medium',
                'category': 'Market Regime',
                'recommendation': 'High volatility regime detected. Consider adjusting risk parameters.',
                'action': 'Review and adjust position sizing'
            })
        
        if stats.kurtosis(values) > 3:
            recommendations.append({
                'priority': 'Low',
                'category': 'Distribution',
                'recommendation': 'Heavy-tailed distribution detected. Standard statistical methods may underestimate extreme events.',
                'action': 'Consider using robust statistical methods'
            })
        
        if recommendations:
            for rec in recommendations:
                priority_color = {'High': '#ff3366', 'Medium': '#ffaa00', 'Low': '#00aaff'}[rec['priority']]
                st.markdown(f"""
                <div class="alert-box {'warning' if rec['priority'] == 'Medium' else 'info' if rec['priority'] == 'Low' else ''}">
                    <span style="color: {priority_color}; font-weight: bold;">[{rec['priority']}]</span> <strong>{rec['category']}</strong><br>
                    {rec['recommendation']}<br>
                    <em>Action: {rec['action']}</em>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No significant issues detected. System is operating normally.")
    
    # Export options
    st.markdown("---")
    st.markdown("### 📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = data.to_csv(index=False)
        st.download_button(
            label="📊 Export Data (CSV)",
            data=csv,
            file_name=f"anomaly_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(values),
            'anomalies_detected': int(total_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'drift_detected': bool(drift_detected),  # Convert numpy bool_ to Python bool
            'drift_score': float(drift_score),
            'current_regime': regime_info['name'] if regime_info else 'Unknown',
            'statistics': {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'skewness': float(stats.skew(values)),
                'kurtosis': float(stats.kurtosis(values))
            }
        }
        st.download_button(
            label="📋 Export Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # PDF Report Generation
        if REPORTLAB_AVAILABLE:
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, 
                                    rightMargin=72, leftMargin=72,
                                    topMargin=72, bottomMargin=72)
            
            # Create custom styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#1a1a2e')
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.HexColor('#0066cc')
            )
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=8
            )
            
            story = []
            
            # Title
            story.append(Paragraph("Anomaly Detection Report", title_style))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                   ParagraphStyle('Date', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.gray)))
            story.append(Spacer(1, 30))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            summary_text = f"""
            This report provides a comprehensive analysis of the anomaly detection results for the monitored data stream.
            The analysis covers {len(values):,} data points with an overall anomaly rate of {anomaly_rate:.2%}.
            """
            story.append(Paragraph(summary_text, normal_style))
            story.append(Spacer(1, 20))
            
            # Key Metrics Table
            story.append(Paragraph("Key Metrics", heading_style))
            metrics_data = [
                ['Metric', 'Value', 'Status'],
                ['Total Data Points', f'{len(values):,}', 'OK'],
                ['Anomalies Detected', f'{int(total_anomalies):,}', 'Warning' if total_anomalies > 0 else 'OK'],
                ['Anomaly Rate', f'{anomaly_rate:.2%}', 'Critical' if anomaly_rate > 0.1 else 'OK'],
                ['Drift Score', f'{drift_score:.4f}', 'Warning' if drift_detected else 'OK'],
                ['Drift Detected', 'Yes' if drift_detected else 'No', 'Critical' if drift_detected else 'OK'],
                ['Current Regime', regime_info['name'] if regime_info else 'Unknown', 'Info'],
            ]
            
            metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 30))
            
            # Statistical Summary
            story.append(Paragraph("Statistical Summary", heading_style))
            stats_data = [
                ['Statistic', 'Value'],
                ['Mean', f'{float(np.mean(values)):.4f}'],
                ['Standard Deviation', f'{float(np.std(values)):.4f}'],
                ['Minimum', f'{float(np.min(values)):.4f}'],
                ['Maximum', f'{float(np.max(values)):.4f}'],
                ['Skewness', f'{float(stats.skew(values)):.4f}'],
                ['Kurtosis', f'{float(stats.kurtosis(values)):.4f}'],
            ]
            
            stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fff0')]),
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 30))
            
            # Recommendations
            story.append(Paragraph("Recommendations", heading_style))
            
            recommendations = []
            if anomaly_rate > 0.1:
                recommendations.append("• HIGH PRIORITY: Anomaly rate exceeds 10%. Investigate data quality and review detection thresholds.")
            if drift_detected:
                recommendations.append("• HIGH PRIORITY: Model drift detected. Consider retraining models with recent data.")
            if regime_info and regime_info['name'] == 'High Volatility':
                recommendations.append("• MEDIUM PRIORITY: High volatility regime detected. Review risk parameters.")
            if stats.kurtosis(values) > 3:
                recommendations.append("• LOW PRIORITY: Heavy-tailed distribution detected. Consider robust statistical methods.")
            
            if not recommendations:
                recommendations.append("• System is operating normally. No immediate action required.")
            
            for rec in recommendations:
                story.append(Paragraph(rec, normal_style))
            
            story.append(Spacer(1, 30))
            
            # Footer
            story.append(Paragraph("---", normal_style))
            footer_text = f"Report generated by Real-Time Anomaly Detection Platform v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], 
                                                               fontSize=8, textColor=colors.gray, alignment=TA_CENTER)))
            
            # Build PDF
            doc.build(story)
            pdf_buffer.seek(0)
            
            st.download_button(
                label="📈 Generate PDF Report",
                data=pdf_buffer.getvalue(),
                file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.button("📈 Generate PDF Report", use_container_width=True, disabled=True)
            st.caption("Install reportlab: pip install reportlab")


if __name__ == "__main__":
    main()
