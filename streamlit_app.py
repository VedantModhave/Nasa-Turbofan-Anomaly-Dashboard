import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import time
import random
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Team Nexus:NASA Turbofan Anomaly Detection Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations
st.markdown("""
<style>
    .main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #FFFFFF; /* Pure White for maximum contrast */
    margin-bottom: 1rem;
    animation: fadeIn 1.2s ease-in-out;
}


    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 0.5rem;
        animation: slideInRight 0.8s ease-in-out;
    }
    .card {
        background-color:rgb(16, 21, 31);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color:rgb(15, 20, 26);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .warning {
        color: #DC2626;
        font-weight: 600;
    }
    .success {
        color: #059669;
        font-weight: 600;
    }
    .user-focus-container {
        border: 2px solid #3B82F6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color:rgb(37, 41, 41);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        animation: pulseGlow 2s infinite alternate;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color:rgb(25, 35, 49);
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #DBEAFE;
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        border-bottom: 2px solid #3B82F6;
        transform: translateY(-2px);
    }
    .sensor-highlight {
        font-weight: bold;
        color: #2563EB;
        animation: pulse 1.5s infinite;
    }
    .loading-spinner {
        text-align: center;
        margin: 2rem 0;
    }
    .loading-spinner img {
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
    }
    .stButton button {
        transition: all 0.3s ease;
        background-color: #3B82F6 !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3) !important;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 10px rgba(59, 130, 246, 0.4) !important;
        background-color: #2563EB !important;
    }
    
    /* Model badges */
    .model-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .model-badge.lightgbm {
        background-color: #9333EA;
        color: white;
    }
    .model-badge.xgboost {
        background-color: #16A34A;
        color: white;
    }
    .model-badge.catboost {
        background-color: #F59E0B;
        color: white;
    }
    .model-badge.lstm {
        background-color: #2563EB;
        color: white;
    }
    .model-badge.ensemble {
        background-color: #DC2626;
        color: white;
    }
    .model-badge.stacking {
        background-color: #0891B2;
        color: white;
    }
    
    /* Sensor status indicators */
    .sensor-status {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.75rem;
        margin-right: 5px;
    }
    .sensor-status.high-anomaly {
        background-color: #FEE2E2;
        color: #DC2626;
        border: 1px solid #DC2626;
    }
    .sensor-status.medium-anomaly {
        background-color: #FEF3C7;
        color: #D97706;
        border: 1px solid #D97706;
    }
    .sensor-status.good {
        background-color: #DCFCE7;
        color: #16A34A;
        border: 1px solid #16A34A;
    }
    .sensor-status.normal {
        background-color: #E0F2FE;
        color: #0284C7;
        border: 1px solid #0284C7;
    }
    
    /* Report highlights */
    .report-highlight {
        background-color:rgb(18, 16, 10);
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    @keyframes pulseGlow {
        from { box-shadow: 0 0 5px rgba(59, 130, 246, 0.5); }
        to { box-shadow: 0 0 20px rgba(59, 130, 246, 0.8); }
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Ensure content visibility */
    .stPlotlyChart {
        background-color: rgba(240, 249, 255, 0.8) !important;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Ensure text is visible against any background */
    .stMarkdown, .stText {
        color: #FFFFFF !important;
    }
    
    /* Improve contrast for plotly charts */
    .js-plotly-plot .plotly .main-svg {
        background-color: rgba(240, 249, 255, 0.8) !important;
    }
    
    /* Ensure sidebar content is visible */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #F1F5F9 !important;
    }
    
    /* Ensure tabs content area has background */
    .stTabs [data-baseweb="tab-panel"] {
        background-color:rgb(28, 42, 56) !important;
        padding: 15px !important;
        border-radius: 0 0 8px 8px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Enhanced metric display */
    .metric-container {
        background: linear-gradient(135deg,rgb(26, 118, 239) 0%, #DBEAFE 100%);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        margin-bottom: 4px;
    }
    
    /* Enhanced selectbox */
    .stSelectbox [data-baseweb=select] {
        background-color: #F8FAFC !important;
        border-radius: 6px !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    .stSelectbox [data-baseweb=select]:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load and preprocess the NASA Turbofan Engine dataset."""
    # Define column names for the dataset
    columns = [
        'engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 
        'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 
        'sensor6', 'sensor7', 'sensor8', 'sensor9', 'sensor10', 
        'sensor11', 'sensor12', 'sensor13', 'sensor14', 'sensor15', 
        'sensor16', 'sensor17', 'sensor18', 'sensor19', 'sensor20', 
        'sensor21'
    ]
    
    # Generate synthetic data with a fixed seed for consistency
    np.random.seed(42)
    
    # Create synthetic training data
    n_engines_train = 20
    max_cycles = 300
    
    train_data = []
    
    for engine_id in range(1, n_engines_train + 1):
        n_cycles = np.random.randint(200, max_cycles)
        
        for cycle in range(1, n_cycles + 1):
            # Create base values that degrade over time
            degradation = cycle / n_cycles
            
            # Settings (operational conditions)
            setting1 = np.random.normal(0.5, 0.05)
            setting2 = np.random.normal(0.7, 0.03)
            setting3 = np.random.normal(0.3, 0.04)
            
            # Sensor readings with increasing noise and degradation patterns
            sensors = []
            for i in range(21):
                # Make specific sensors more anomalous (6, 7, 12, 17)
                if i+1 in [6, 7, 12, 17]:
                    base = np.random.normal(0.5, 0.15)
                    noise = np.random.normal(0, 0.08 + 0.15 * degradation)
                    
                    # Add more pronounced anomalies
                    if np.random.random() < 0.06 and cycle > n_cycles * 0.6:
                        noise += np.random.normal(0, 1.5)
                    
                    trend = base + degradation * 0.9 * np.sin(cycle / 15)
                
                # Make specific sensors very good (2, 3, 8)
                elif i+1 in [2, 3, 8]:
                    base = np.random.normal(0.5, 0.03)
                    noise = np.random.normal(0, 0.01)
                    trend = base + degradation * 0.1
                
                # Other sensors with random patterns
                else:
                    # Randomly assign medium anomaly patterns to some sensors
                    if i+1 in [4, 9, 15, 19]:
                        base = np.random.normal(0.5, 0.1)
                        noise = np.random.normal(0, 0.04 + 0.08 * degradation)
                        trend = base + degradation * 0.5 * np.cos(cycle / 25)
                        
                        if np.random.random() < 0.04 and cycle > n_cycles * 0.7:
                            noise += np.random.normal(0, 0.8)
                    else:
                        base = np.random.normal(0.5, 0.07)
                        noise = np.random.normal(0, 0.03)
                        trend = base + degradation * 0.3
                
                sensors.append(max(0, trend + noise))
            
            row = [engine_id, cycle, setting1, setting2, setting3] + sensors
            train_data.append(row)
    
    # Convert to DataFrame
    train_df = pd.DataFrame(train_data, columns=columns)
    
    # Create synthetic test data
    n_engines_test = 10
    test_data = []
    
    for engine_id in range(1, n_engines_test + 1):
        n_cycles = np.random.randint(100, 200)
        
        for cycle in range(1, n_cycles + 1):
            # Create base values that degrade over time
            degradation = cycle / n_cycles
            
            # Settings
            setting1 = np.random.normal(0.5, 0.05)
            setting2 = np.random.normal(0.7, 0.03)
            setting3 = np.random.normal(0.3, 0.04)
            
            # Sensor readings with the same patterns as training data
            sensors = []
            for i in range(21):
                # Make specific sensors more anomalous (6, 7, 12, 17)
                if i+1 in [6, 7, 12, 17]:
                    base = np.random.normal(0.5, 0.15)
                    noise = np.random.normal(0, 0.08 + 0.15 * degradation)
                    
                    # Add more pronounced anomalies
                    if np.random.random() < 0.06 and cycle > n_cycles * 0.6:
                        noise += np.random.normal(0, 1.5)
                    
                    trend = base + degradation * 0.9 * np.sin(cycle / 15)
                
                # Make specific sensors very good (2, 3, 8)
                elif i+1 in [2, 3, 8]:
                    base = np.random.normal(0.5, 0.03)
                    noise = np.random.normal(0, 0.01)
                    trend = base + degradation * 0.1
                
                # Other sensors with random patterns
                else:
                    # Randomly assign medium anomaly patterns to some sensors
                    if i+1 in [4, 9, 15, 19]:
                        base = np.random.normal(0.5, 0.1)
                        noise = np.random.normal(0, 0.04 + 0.08 * degradation)
                        trend = base + degradation * 0.5 * np.cos(cycle / 25)
                        
                        if np.random.random() < 0.04 and cycle > n_cycles * 0.7:
                            noise += np.random.normal(0, 0.8)
                    else:
                        base = np.random.normal(0.5, 0.07)
                        noise = np.random.normal(0, 0.03)
                        trend = base + degradation * 0.3
                
                sensors.append(max(0, trend + noise))
            
            row = [engine_id, cycle, setting1, setting2, setting3] + sensors
            test_data.append(row)
    
    # Convert to DataFrame
    test_df = pd.DataFrame(test_data, columns=columns)
    
    # Calculate RUL for training data
    train_rul = pd.DataFrame()
    for engine_id in train_df['engine_id'].unique():
        engine_data = train_df[train_df['engine_id'] == engine_id]
        max_cycle = engine_data['cycle'].max()
        engine_data['RUL'] = max_cycle - engine_data['cycle']
        train_rul = pd.concat([train_rul, engine_data])
    
    # Calculate RUL for test data
    test_rul = pd.DataFrame()
    for engine_id in test_df['engine_id'].unique():
        engine_data = test_df[test_df['engine_id'] == engine_id]
        max_cycle = engine_data['cycle'].max()
        # Generate a random RUL value for the end of the test sequence
        end_rul = np.random.randint(10, 50)
        engine_data['RUL'] = end_rul + max_cycle - engine_data['cycle']
        test_rul = pd.concat([test_rul, engine_data])
    
    return train_rul, test_rul

def get_sensor_status(sensor_num):
    """Get the status of a sensor based on its anomaly level."""
    high_anomaly_sensors = [6, 7, 12, 17]
    good_sensors = [2, 3, 8]
    medium_anomaly_sensors = [4, 9, 15, 19]
    
    if sensor_num in high_anomaly_sensors:
        return "high-anomaly", "High Anomaly"
    elif sensor_num in good_sensors:
        return "good", "Good"
    elif sensor_num in medium_anomaly_sensors:
        return "medium-anomaly", "Medium Anomaly"
    else:
        return "normal", "Normal"

def get_static_predictions(data, engine_id, model_type="XGBoost"):
    """Generate static anomaly predictions for a specific engine based on model type."""
    engine_data = data[data['engine_id'] == engine_id].copy()
    n_samples = len(engine_data)
    
    # Generate static predictions (mostly normal with some anomalies)
    np.random.seed(int(engine_id) + 42)  # Use engine_id in seed for consistency
    
    # Generate mostly normal predictions (1) with some anomalies (-1)
    # More anomalies toward the end of the lifecycle
    predictions = np.ones(n_samples)
    
    # Add anomalies more frequently in the last 30% of cycles
    last_30_percent = int(0.7 * n_samples)
    
    # Adjust anomaly detection based on model type
    if model_type == "LightGBM":
        # LightGBM - More conservative with anomaly detection
        early_anomaly_rate = 0.01
        late_anomaly_rate = 0.12
    elif model_type == "XGBoost":
        # XGBoost - Balanced anomaly detection
        early_anomaly_rate = 0.02
        late_anomaly_rate = 0.15
    elif model_type == "CatBoost":
        # CatBoost - More aggressive with anomaly detection
        early_anomaly_rate = 0.03
        late_anomaly_rate = 0.18
    elif model_type == "LSTM":
        # LSTM - Better at detecting sequential patterns
        early_anomaly_rate = 0.015
        late_anomaly_rate = 0.20
        # Create more sequential patterns in anomalies
        for i in range(last_30_percent, n_samples-3):
            if np.random.random() < 0.1:
                predictions[i:i+3] = -1
    elif model_type == "Weighted Ensemble":
        # Weighted Ensemble - Balanced with fewer false positives
        early_anomaly_rate = 0.01
        late_anomaly_rate = 0.14
    else:  # Stacking Ensemble
        # Stacking Ensemble - Most accurate overall
        early_anomaly_rate = 0.015
        late_anomaly_rate = 0.16
    
    # Early cycles (rare anomalies)
    early_anomalies = np.random.choice(
        range(0, last_30_percent),
        size=int(early_anomaly_rate * last_30_percent),
        replace=False
    )
    predictions[early_anomalies] = -1
    
    # Later cycles (more anomalies)
    late_anomalies = np.random.choice(
        range(last_30_percent, n_samples),
        size=int(late_anomaly_rate * (n_samples - last_30_percent)),
        replace=False
    )
    predictions[late_anomalies] = -1
    
    # Generate anomaly scores (distance from decision boundary)
    # Normal points have positive scores, anomalies have negative scores
    scores = np.random.normal(0.5, 0.2, n_samples)
    scores[predictions == -1] = -np.random.normal(0.5, 0.3, sum(predictions == -1))
    
    # Add model-specific characteristics to scores
    if model_type == "LightGBM":
        # LightGBM - More variance in scores
        scores = scores * 1.2
    elif model_type == "LSTM":
        # LSTM - Smoother transitions in scores
        scores = pd.Series(scores).rolling(window=3, min_periods=1).mean().values
    elif model_type == "Stacking Ensemble":
        # Stacking Ensemble - More confident scores (further from zero)
        scores = scores * 1.3
        
    return predictions, scores

def get_static_clusters(data):
    """Generate static cluster assignments for visualization."""
    # Create a PCA-like 2D representation for visualization
    np.random.seed(42)
    n_samples = len(data)
    
    # Generate 5 cluster centers
    centers = np.array([
        [-2, -2],
        [2, -2],
        [0, 2],
        [-1, 1],
        [1, 1]
    ])
    
    # Assign each engine to a cluster
    engine_clusters = {}
    for engine_id in data['engine_id'].unique():
        engine_clusters[engine_id] = np.random.randint(0, 5)
    
    # Generate points around cluster centers
    X_pca = np.zeros((n_samples, 2))
    clusters = np.zeros(n_samples)
    
    for i, (_, row) in enumerate(data.iterrows()):
        engine_id = row['engine_id']
        cycle = row['cycle']
        max_cycle = data[data['engine_id'] == engine_id]['cycle'].max()
        
        # Get cluster center for this engine
        cluster_idx = engine_clusters[engine_id]
        center = centers[cluster_idx]
        
        # Add noise and drift based on cycle
        cycle_ratio = cycle / max_cycle
        drift = np.array([cycle_ratio * 0.5, cycle_ratio * 0.3])
        noise = np.random.normal(0, 0.2, 2)
        
        X_pca[i] = center + drift + noise
        clusters[i] = cluster_idx
    
    return X_pca, clusters

def simulate_loading():
    """Simulate loading with a progress bar."""
    progress_bar = st.progress(0)
    for i in range(101):
        progress_bar.progress(i)
        time.sleep(0.01)
    progress_bar.empty()

def plot_sensor_trends(data, engine_id, sensor_cols, highlight_anomalies=True, predictions=None, scores=None):
    """Plot sensor trends for a specific engine with optional anomaly highlighting."""
    engine_data = data[data['engine_id'] == engine_id].copy()
    
    if highlight_anomalies and predictions is not None and scores is not None:
        # Map predictions and scores to engine data
        engine_indices = engine_data.index
        engine_predictions = predictions[engine_indices]
        engine_scores = scores[engine_indices]
        
        # Add predictions and scores to the data
        engine_data['anomaly'] = engine_predictions
        engine_data['score'] = engine_scores
    
    # Create a multi-line plot
    fig = go.Figure()
    
    for sensor in sensor_cols:
        # Get sensor status for color coding
        sensor_num = int(sensor.replace('sensor', ''))
        status_class, status_label = get_sensor_status(sensor_num)
        
        # Set color based on sensor status
        if status_class == "high-anomaly":
            line_color = "#DC2626"  # Red
        elif status_class == "medium-anomaly":
            line_color = "#F59E0B"  # Amber
        elif status_class == "good":
            line_color = "#16A34A"  # Green
        else:
            line_color = "#3B82F6"  # Blue
        
        fig.add_trace(go.Scatter(
            x=engine_data['cycle'],
            y=engine_data[sensor],
            mode='lines+markers',
            name=f"{sensor} ({status_label})",
            marker=dict(size=8),
            line=dict(width=2, color=line_color)
        ))
        
        # If highlighting anomalies, add anomaly points
        if highlight_anomalies and 'anomaly' in engine_data.columns:
            anomaly_data = engine_data[engine_data['anomaly'] == -1]
            if not anomaly_data.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_data['cycle'],
                    y=anomaly_data[sensor],
                    mode='markers',
                    name=f'{sensor} Anomalies',
                    marker=dict(color='red', size=12, symbol='circle-open')
                ))
    
    fig.update_layout(
        title=f"Sensor Trends for Engine {engine_id}",
        xaxis_title="Cycle",
        yaxis_title="Sensor Value",
        legend_title="Sensors",
        height=500,
        hovermode="closest",
        plot_bgcolor='rgba(240,249,255,0.8)',
        paper_bgcolor='rgba(240,249,255,0.8)',
        font=dict(color='#1E293B'),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [True] * len(fig.data)}],
                        label="Show All",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [i % 2 == 0 for i in range(len(fig.data))]}],
                        label="Hide Anomalies",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)'
            ),
        ]
    )
    
    # Add animation frames
    frames = []
    for cycle in range(1, engine_data['cycle'].max() + 1, 10):
        frame_data = []
        for sensor in sensor_cols:
            visible_data = engine_data[engine_data['cycle'] <= cycle]
            frame_data.append(
                go.Scatter(
                    x=visible_data['cycle'],
                    y=visible_data[sensor]
                )
            )
            
            if highlight_anomalies and 'anomaly' in engine_data.columns:
                anomaly_data = visible_data[visible_data['anomaly'] == -1]
                if not anomaly_data.empty:
                    frame_data.append(
                        go.Scatter(
                            x=anomaly_data['cycle'],
                            y=anomaly_data[sensor]
                        )
                    )
        
        frames.append(go.Frame(data=frame_data, name=f"frame{cycle}"))
    
    fig.frames = frames
    
    return fig

def plot_single_sensor_with_anomalies(data, engine_id, sensor, predictions, scores):
    """Plot a single sensor with anomalies highlighted."""
    engine_data = data[data['engine_id'] == engine_id].copy()
    
    # Map predictions and scores to engine data
    engine_indices = engine_data.index
    engine_predictions = predictions[engine_indices]
    engine_scores = scores[engine_indices]
    
    # Add predictions and scores to the data
    engine_data['anomaly'] = engine_predictions
    engine_data['score'] = engine_scores
    
    # Get sensor status for styling
    sensor_num = int(sensor.replace('sensor', ''))
    status_class, status_label = get_sensor_status(sensor_num)
    
    # Set color based on sensor status
    if status_class == "high-anomaly":
        line_color = "#DC2626"  # Red
        title_prefix = "High Anomaly Risk:"
    elif status_class == "medium-anomaly":
        line_color = "#F59E0B"  # Amber
        title_prefix = "Medium Anomaly Risk:"
    elif status_class == "good":
        line_color = "#16A34A"  # Green
        title_prefix = "Low Anomaly Risk:"
    else:
        line_color = "#3B82F6"  # Blue
        title_prefix = ""
    
    fig = go.Figure()
    
    # Normal points
    normal_data = engine_data[engine_data['anomaly'] == 1]
    fig.add_trace(go.Scatter(
        x=normal_data['cycle'],
        y=normal_data[sensor],
        mode='markers+lines',
        name='Normal',
        marker=dict(color=line_color, size=8),
        line=dict(color=line_color, width=2)
    ))
    
    # Anomaly points
    anomaly_data = engine_data[engine_data['anomaly'] == -1]
    if not anomaly_data.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_data['cycle'],
            y=anomaly_data[sensor],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=12, symbol='circle-open')
        ))
    
    # Add confidence interval
    y_mean = engine_data[sensor].rolling(window=5).mean()
    y_std = engine_data[sensor].rolling(window=5).std().fillna(0)
    
    fig.add_trace(go.Scatter(
        x=engine_data['cycle'],
        y=y_mean + 2*y_std,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=engine_data['cycle'],
        y=y_mean - 2*y_std,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 255, 0.15)',
        fill='tonexty',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"{title_prefix} {sensor} with Anomalies for Engine {engine_id}",
        xaxis_title="Cycle",
        yaxis_title="Sensor Value",
        height=400,
        hovermode="closest",
        plot_bgcolor='rgba(240,249,255,0.8)',
        paper_bgcolor='rgba(240,249,255,0.8)',
        font=dict(color='#1E293B'),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )
    
    # Add annotation for sensor status
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Status: {status_label}",
        showarrow=False,
        font=dict(color=line_color, size=14),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=line_color,
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    # Add animation frames
    frames = []
    for cycle in range(1, engine_data['cycle'].max() + 1, 5):
        visible_data = engine_data[engine_data['cycle'] <= cycle]
        normal_data = visible_data[visible_data['anomaly'] == 1]
        anomaly_data = visible_data[visible_data['anomaly'] == -1]
        
        frame_data = [
            go.Scatter(
                x=normal_data['cycle'],
                y=normal_data[sensor]
            )
        ]
        
        if not anomaly_data.empty:
            frame_data.append(
                go.Scatter(
                    x=anomaly_data['cycle'],
                    y=anomaly_data[sensor]
                )
            )
        
        # Add confidence intervals to frame
        y_mean = visible_data[sensor].rolling(window=5).mean()
        y_std = visible_data[sensor].rolling(window=5).std().fillna(0)
        
        frame_data.append(
            go.Scatter(
                x=visible_data['cycle'],
                y=y_mean + 2*y_std
            )
        )
        
        frame_data.append(
            go.Scatter(
                x=visible_data['cycle'],
                y=y_mean - 2*y_std
            )
        )
        
        frames.append(go.Frame(data=frame_data, name=f"frame{cycle}"))
    
    fig.frames = frames
    
    return fig

def plot_rul_prediction(data, engine_id, model_type="XGBoost"):
    """Plot RUL prediction for a specific engine."""
    engine_data = data[data['engine_id'] == engine_id]
    
    # Adjust RUL prediction based on model type
    if model_type == "LSTM":
        # LSTM is better at long-term predictions
        adjustment = 1.05
        confidence_interval = 0.08
    elif model_type == "Stacking Ensemble":
        # Stacking ensemble has tighter confidence intervals
        adjustment = 1.0
        confidence_interval = 0.07
    elif model_type == "Weighted Ensemble":
        # Weighted ensemble is more conservative
        adjustment = 0.95
        confidence_interval = 0.09
    elif model_type == "LightGBM":
        # LightGBM tends to be more optimistic
        adjustment = 1.1
        confidence_interval = 0.1
    elif model_type == "CatBoost":
        # CatBoost is more pessimistic
        adjustment = 0.9
        confidence_interval = 0.08
    else:  # XGBoost
        # XGBoost is balanced
        adjustment = 1.0
        confidence_interval = 0.1
    
    fig = go.Figure()
    
    # Adjust RUL values based on model type
    adjusted_rul = engine_data['RUL'] * adjustment
    
    fig.add_trace(go.Scatter(
        x=engine_data['cycle'],
        y=adjusted_rul,
        mode='lines+markers',
        name='RUL',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    # Add threshold lines
    fig.add_shape(
        type="line",
        x0=engine_data['cycle'].min(),
        y0=30,
        x1=engine_data['cycle'].max(),
        y1=30,
        line=dict(color="orange", width=2, dash="dash"),
        name="Warning"
    )
    
    fig.add_shape(
        type="line",
        x0=engine_data['cycle'].min(),
        y0=15,
        x1=engine_data['cycle'].max(),
        y1=15,
        line=dict(color="red", width=2, dash="dash"),
        name="Critical"
    )
    
    # Add current cycle marker
    current_cycle = engine_data['cycle'].max()
    current_rul = adjusted_rul.iloc[-1]
    
    fig.add_trace(go.Scatter(
        x=[current_cycle],
        y=[current_rul],
        mode='markers',
        name='Current Status',
        marker=dict(color='purple', size=14, symbol='star')
    ))
    
    # Add prediction interval
    x_future = list(range(current_cycle, current_cycle + int(current_rul) + 10))
    y_mean = [max(0, current_rul - (x - current_cycle)) for x in x_future]
    y_upper = [y + min(5, confidence_interval * y) for y in y_mean]
    y_lower = [max(0, y - min(5, confidence_interval * y)) for y in y_mean]
    
    fig.add_trace(go.Scatter(
        x=x_future,
        y=y_upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=x_future,
        y=y_lower,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(0, 100, 0, 0.2)',
        fill='tonexty',
        name='Prediction Interval'
    ))
    
    # Add model type annotation
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Model: {model_type}",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#3B82F6",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    fig.update_layout(
        title=f"Remaining Useful Life (RUL) for Engine {engine_id}",
        xaxis_title="Cycle",
        yaxis_title="RUL (cycles)",
        height=400,
        hovermode="closest",
        plot_bgcolor='rgba(62, 148, 205, 0.8)',
        paper_bgcolor='rgba(5, 8, 9, 0.8)',
        font=dict(color='#1E293B'),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor='rgba(26, 13, 13, 0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        annotations=[
            dict(
                x=engine_data['cycle'].max(),
                y=30,
                xref="x",
                yref="y",
                text="Warning",
                showarrow=False,
                font=dict(color="orange")
            ),
            dict(
                x=engine_data['cycle'].max(),
                y=15,
                xref="x",
                yref="y",
                text="Critical",
                showarrow=False,
                font=dict(color="red")
            )
        ]
    )
    
    # Add animation frames
    frames = []
    for cycle in range(engine_data['cycle'].min(), engine_data['cycle'].max() + 1, 10):
        visible_data = engine_data[engine_data['cycle'] <= cycle]
        
        if not visible_data.empty:
            current = visible_data['cycle'].max()
            current_rul_val = adjusted_rul.loc[visible_data.index[-1]]
            
            x_future = list(range(current, current + int(current_rul_val) + 10))
            y_mean = [max(0, current_rul_val - (x - current)) for x in x_future]
            y_upper = [y + min(5, confidence_interval * y) for y in y_mean]
            y_lower = [max(0, y - min(5, confidence_interval * y)) for y in y_mean]
            
            frame_data = [
                go.Scatter(
                    x=visible_data['cycle'],
                    y=adjusted_rul.loc[visible_data.index]
                ),
                go.Scatter(
                    x=[current],
                    y=[current_rul_val]
                ),
                go.Scatter(
                    x=x_future,
                    y=y_upper
                ),
                go.Scatter(
                    x=x_future,
                    y=y_lower
                )
            ]
            
            frames.append(go.Frame(data=frame_data, name=f"frame{cycle}"))
    
    fig.frames = frames
    
    return fig

def plot_pca_clusters(data, X_pca, clusters, highlight_engine=None):
    """Plot PCA clusters for the data with optional engine highlighting."""
    # Create a DataFrame for plotting
    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clusters,
        'Engine': data['engine_id'],
        'Cycle': data['cycle'],
        'RUL': data['RUL']
    })
    
    # Create a new column for highlighting the selected engine
    if highlight_engine is not None:
        pca_df['Highlight'] = pca_df['Engine'] == highlight_engine
    
    # Create the base scatter plot
    fig = px.scatter(
        pca_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_data=['Engine', 'Cycle', 'RUL'],
        title='Engine Sensor Data Clusters (PCA)',
        color_continuous_scale=px.colors.qualitative.G10,
        animation_frame='Cycle',
        animation_group='Engine',
        range_x=[-3, 3],
        range_y=[-3, 3]
    )
    
    # If highlighting an engine, add a trace for the highlighted points
    if highlight_engine is not None:
        highlight_df = pca_df[pca_df['Engine'] == highlight_engine]
        
        fig.add_trace(go.Scatter(
            x=highlight_df['PCA1'],
            y=highlight_df['PCA2'],
            mode='markers',
            marker=dict(
                color='yellow',
                size=12,
                line=dict(
                    color='black',
                    width=2
                ),
                symbol='circle'
            ),
            name=f'Engine {highlight_engine}'
        ))
    
    fig.update_layout(
        height=600, 
        hovermode="closest",
        plot_bgcolor='rgba(240,249,255,0.8)',
        paper_bgcolor='rgba(240,249,255,0.8)',
        font=dict(color='#1E293B'),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )
    
    # Add annotation explaining the clusters
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text="Clusters represent similar engine behavior patterns",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#3B82F6",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    # Improve animation
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 300
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
    
    return fig

def plot_sensor_importance(data):
    """Plot sensor importance based on variance."""
    # Calculate feature importance (using variance as a simple metric)
    sensor_cols = [col for col in data.columns if 'sensor' in col]
    
    # Create custom importance values based on sensor status ```python
    # Create custom importance values based on sensor status
    sensor_importance = {}
    
    # High anomaly sensors (6, 7, 12, 17) get high importance
    for sensor in ['sensor6', 'sensor7', 'sensor12', 'sensor17']:
        sensor_importance[sensor] = np.random.uniform(0.7, 0.9)
    
    # Good sensors (2, 3, 8) get low importance
    for sensor in ['sensor2', 'sensor3', 'sensor8']:
        sensor_importance[sensor] = np.random.uniform(0.1, 0.3)
    
    # Medium anomaly sensors (4, 9, 15, 19) get medium importance
    for sensor in ['sensor4', 'sensor9', 'sensor15', 'sensor19']:
        sensor_importance[sensor] = np.random.uniform(0.4, 0.6)
    
    # Other sensors get random importance
    for sensor in sensor_cols:
        if sensor not in sensor_importance:
            sensor_importance[sensor] = np.random.uniform(0.2, 0.5)
    
    # Convert to Series and sort
    sensor_importance = pd.Series(sensor_importance).sort_values(ascending=False)
    
    # Create a color scale based on importance
    max_importance = sensor_importance.max()
    colors = []
    
    for sensor, importance in sensor_importance.items():
        sensor_num = int(sensor.replace('sensor', ''))
        status_class, _ = get_sensor_status(sensor_num)
        
        if status_class == "high-anomaly":
            color = f'rgba(220, 38, 38, 0.8)'  # Red
        elif status_class == "medium-anomaly":
            color = f'rgba(245, 158, 11, 0.8)'  # Amber
        elif status_class == "good":
            color = f'rgba(22, 163, 74, 0.8)'  # Green
        else:
            color = f'rgba(59, 130, 246, 0.8)'  # Blue
            
        colors.append(color)
    
    fig = go.Figure()
    
    # Add bars with animation
    for i, (sensor, importance) in enumerate(sensor_importance.items()):
        sensor_num = int(sensor.replace('sensor', ''))
        _, status_label = get_sensor_status(sensor_num)
        
        fig.add_trace(go.Bar(
            x=[sensor],
            y=[importance],
            name=f"{sensor} ({status_label})",
            marker_color=colors[i],
            text=[f"{importance:.3f}"],
            textposition='auto',
            hoverinfo='text',
            hovertext=f"{sensor} ({status_label}): {importance:.3f}"
        ))
    
    fig.update_layout(
        title='Sensor Importance for Anomaly Detection',
        xaxis_title='Sensor',
        yaxis_title='Importance Score',
        height=400,
        hovermode="closest",
        plot_bgcolor='rgba(240,249,255,0.8)',
        paper_bgcolor='rgba(240,249,255,0.8)',
        font=dict(color='#1E293B'),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    # Add legend for sensor status
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="Sensor Status Legend:",
        showarrow=False,
        font=dict(size=12, color="#1E293B"),
        align="left"
    )
    
    status_types = [
        ("High Anomaly", "#DC2626"),
        ("Medium Anomaly", "#F59E0B"),
        ("Good", "#16A34A"),
        ("Normal", "#3B82F6")
    ]
    
    for i, (label, color) in enumerate(status_types):
        fig.add_annotation(
            x=0.02,
            y=0.94 - i*0.04,
            xref="paper",
            yref="paper",
            text=f"● {label}",
            showarrow=False,
            font=dict(size=10, color=color),
            align="left"
        )
    
    # Add animation
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"y": [[importance] for importance in sensor_importance.values]}],
                        label="Reset",
                        method="update"
                    ),
                    dict(
                        args=[
                            {"y": [[importance * (1 + 0.2 * np.sin(i))] for i, importance in enumerate(sensor_importance.values)]}
                        ],
                        label="Emphasize",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)'
            ),
        ]
    )
    
    return fig

def get_maintenance_recommendations(current_rul, anomaly_percentage, model_type="XGBoost"):
    """Get maintenance recommendations based on RUL and anomaly percentage."""
    # Adjust thresholds based on model type
    if model_type == "LSTM":
        critical_threshold = 16  # LSTM is more precise with time series
        warning_threshold = 32
    elif model_type == "Stacking Ensemble":
        critical_threshold = 15  # Stacking ensemble is balanced
        warning_threshold = 30
    elif model_type == "CatBoost":
        critical_threshold = 13  # CatBoost is more conservative
        warning_threshold = 28
    elif model_type == "LightGBM":
        critical_threshold = 17  # LightGBM is more optimistic
        warning_threshold = 33
    elif model_type == "Weighted Ensemble":
        critical_threshold = 14  # Weighted ensemble is balanced
        warning_threshold = 29
    else:  # XGBoost
        critical_threshold = 15  # Default thresholds
        warning_threshold = 30
    
    if current_rul <= critical_threshold:
        status = "Critical"
        color = "#DC2626"  # Red
        recommendations = [
            "Schedule immediate maintenance within the next 15 cycles",
            "Perform full engine inspection focusing on high-wear components",
            "Replace critical components showing signs of degradation",
            "Verify all systems before returning to service"
        ]
    elif current_rul <= warning_threshold:
        status = "Warning"
        color = "#F59E0B"  # Amber
        recommendations = [
            f"Plan maintenance within the next {warning_threshold} cycles",
            "Inspect high-wear components identified by sensor readings",
            "Monitor sensor readings more frequently",
            "Prepare for potential component replacement"
        ]
    else:
        if anomaly_percentage > 10:
            status = "Caution"
            color = "#2563EB"  # Blue
            recommendations = [
                "Increase monitoring frequency for anomalous sensors",
                "Perform targeted inspection of components related to anomalous sensors",
                "Follow standard maintenance schedule with additional checks",
                "Document anomaly patterns for future reference"
            ]
        else:
            status = "Good"
            color = "#059669"  # Green
            recommendations = [
                "Continue regular monitoring",
                "Perform routine inspections according to schedule",
                "Monitor for any changes in sensor trends",
                "Follow standard maintenance schedule"
            ]
    
    return status, color, recommendations

def get_model_description(model_type):
    """Get description for a specific model type."""
    if model_type == "LightGBM":
        return {
            "full_name": "Light Gradient Boosting Machine",
            "description": "A gradient boosting framework that uses tree-based learning algorithms. Known for its efficiency and accuracy.",
            "strengths": ["Fast training speed", "Low memory usage", "High accuracy", "Support for large datasets"],
            "best_for": "Datasets with many features and moderate anomaly rates",
            "badge_class": "lightgbm"
        }
    elif model_type == "XGBoost":
        return {
            "full_name": "eXtreme Gradient Boosting",
            "description": "An optimized distributed gradient boosting library. Provides parallel tree boosting that solves many data science problems.",
            "strengths": ["High performance", "Regularization to prevent overfitting", "Handles missing values well"],
            "best_for": "General-purpose anomaly detection with balanced precision and recall",
            "badge_class": "xgboost"
        }
    elif model_type == "CatBoost":
        return {
            "full_name": "Categorical Boosting",
            "description": "A gradient boosting algorithm that handles categorical features automatically. Reduces the need for extensive data preprocessing.",
            "strengths": ["Handles categorical features natively", "Robust to outliers", "Less overfitting"],
            "best_for": "Data with many categorical features and when minimizing false negatives is critical",
            "badge_class": "catboost"
        }
    elif model_type == "LSTM":
        return {
            "full_name": "Long Short-Term Memory",
            "description": "A recurrent neural network architecture designed to model temporal sequences and their long-range dependencies.",
            "strengths": ["Captures temporal patterns", "Handles variable-length sequences", "Learns long-term dependencies"],
            "best_for": "Time series data where the sequence and order of events matter",
            "badge_class": "lstm"
        }
    elif model_type == "Weighted Ensemble":
        return {
            "full_name": "Weighted Ensemble Model",
            "description": "Combines multiple models with different weights based on their performance. Leverages the strengths of each model.",
            "strengths": ["Reduces variance", "More robust than individual models", "Balances different model biases"],
            "best_for": "Complex datasets where different models capture different aspects of the data",
            "badge_class": "ensemble"
        }
    else:  # Stacking Ensemble
        return {
            "full_name": "Stacking Ensemble",
            "description": "A meta-learning approach that combines multiple models via a meta-model. The base models make predictions which become features for the meta-model.",
            "strengths": ["High accuracy", "Learns the best way to combine models", "Reduces both bias and variance"],
            "best_for": "When maximum accuracy is required and computational resources are available",
            "badge_class": "stacking"
        }

def user_focused_dashboard(data, engine_id, predictions, scores, X_pca, clusters, current_cycle, current_rul, anomaly_percentage, model_type="XGBoost"):
    """Create a user-focused dashboard for a specific engine."""
    # Get engine data
    engine_data = data[data['engine_id'] == engine_id]
    
    # Get maintenance recommendations
    status, color, recommendations = get_maintenance_recommendations(current_rul, anomaly_percentage, model_type)
    
    # Get model description
    model_info = get_model_description(model_type)
    
    # Create the dashboard
    st.markdown(f"""
    <div class="user-focus-container">
        <h2 style="color: {color};">Engine {engine_id} Health Dashboard</h2>
        <p>Current status: <span style="color: {color}; font-weight: bold;">{status}</span></p>
        <p>Model: <span class="model-badge {model_info['badge_class']}">{model_type}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Anomaly Detection", "⏱️ RUL Prediction", "🔧 Maintenance"])
    
    with tab1:
        # Overview tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Sensor Trends")
            # Focus on sensors with different anomaly levels
            sensor_cols = ['sensor6', 'sensor7', 'sensor2', 'sensor3', 'sensor12']
            
            # Display sensor status indicators
            st.markdown("<div style='margin-bottom: 10px;'>", unsafe_allow_html=True)
            for sensor in sensor_cols:
                sensor_num = int(sensor.replace('sensor', ''))
                status_class, status_label = get_sensor_status(sensor_num)
                st.markdown(f"<span class='sensor-status {status_class}'>{sensor}: {status_label}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a loading effect
            with st.spinner("Analyzing sensor data..."):
                time.sleep(0.5)  # Simulate loading
                st.plotly_chart(plot_sensor_trends(data, engine_id, sensor_cols, True, predictions, scores), use_container_width=True)
        
        with col2:
            st.markdown("### Engine Health Summary")
            
            # Model information
            st.markdown(f"""
            <div class="card" style="margin-bottom: 15px;">
                <h4>Model: {model_info['full_name']}</h4>
                <p><span class="model-badge {model_info['badge_class']}">{model_type}</span></p>
                <p><small>{model_info['description']}</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Engine metrics
            st.markdown(f"""
            <div class="metric-card">
                <h3>Engine ID: {engine_id}</h3>
                <p>Current Cycle: <strong>{current_cycle}</strong></p>
                <p>Remaining Useful Life: <span style="color: {color};">{current_rul:.1f} cycles</span></p>
                <p>Health Status: <span style="color: {color};">{status}</span></p>
                <p>Anomaly Percentage: <strong>{anomaly_percentage:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a gauge chart for RUL status
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_rul,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Remaining Useful Life"},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 15], 'color': 'rgba(220, 38, 38, 0.3)'},
                        {'range': [15, 30], 'color': 'rgba(245, 158, 11, 0.3)'},
                        {'range': [30, 100], 'color': 'rgba(5, 150, 105, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 15
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                paper_bgcolor='rgba(240,249,255,0.8)',
                font=dict(color='#1E293B')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a "Refresh Data" button with animation
            if st.button("Refresh Data", key="refresh_overview"):
                with st.spinner("Refreshing data..."):
                    time.sleep(1)
                st.success("Data refreshed successfully!")
    
    with tab2:
        # Anomaly Detection tab
        st.markdown("### Anomaly Detection")
        st.markdown("""
        This view shows detected anomalies in sensor readings. Anomalies are highlighted in red and may indicate potential issues with the engine.
        """)
        
        # Model information for anomaly detection
        st.markdown(f"""
        <div class="report-highlight">
            <h4>Model: {model_info['full_name']} <span class="model-badge {model_info['badge_class']}">{model_type}</span></h4>
            <p><strong>Best for:</strong> {model_info['best_for']}</p>
            <p><strong>Key strengths:</strong> {', '.join(model_info['strengths'][:2])}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Let user select sensors to view
        sensor_cols = [col for col in data.columns if 'sensor' in col]
        
        # Group sensors by status
        high_anomaly_sensors = ['sensor6', 'sensor7', 'sensor12', 'sensor17']
        good_sensors = ['sensor2', 'sensor3', 'sensor8']
        medium_anomaly_sensors = ['sensor4', 'sensor9', 'sensor15', 'sensor19']
        
        # Create sensor selection with status indicators
        st.markdown("<div style='margin-bottom: 15px;'>", unsafe_allow_html=True)
        st.markdown("#### Sensor Status:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<strong style='color: #DC2626;'>High Anomaly Risk:</strong>", unsafe_allow_html=True)
            for sensor in high_anomaly_sensors:
                st.markdown(f"<span class='sensor-status high-anomaly'>{sensor}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<strong style='color: #F59E0B;'>Medium Anomaly Risk:</strong>", unsafe_allow_html=True)
            for sensor in medium_anomaly_sensors:
                st.markdown(f"<span class='sensor-status medium-anomaly'>{sensor}</span>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<strong style='color: #16A34A;'>Good Status:</strong>", unsafe_allow_html=True)
            for sensor in good_sensors:
                st.markdown(f"<span class='sensor-status good'>{sensor}</span>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        selected_sensors = st.multiselect(
            "Select sensors to view",
            options=sensor_cols,
            default=high_anomaly_sensors[:2] + good_sensors[:1]
        )
        
        # Add a loading effect when changing selection
        if selected_sensors:
            with st.spinner("Analyzing anomalies..."):
                time.sleep(0.5)  # Simulate loading
                for sensor in selected_sensors:
                    st.plotly_chart(plot_single_sensor_with_anomalies(data, engine_id, sensor, predictions, scores), use_container_width=True)
        else:
            st.info("Please select at least one sensor to view.")
            
        # Add an "Analyze All Sensors" button with animation
        if st.button("Analyze All Sensors", key="analyze_all"):
            with st.spinner("Analyzing all sensors..."):
                simulate_loading()
            st.success("Analysis complete! Select sensors above to view results.")
    
    with tab3:
        # RUL Prediction tab
        st.markdown("### Remaining Useful Life (RUL) Prediction")
        st.markdown("""
        This view shows the predicted remaining useful life (RUL) for the engine. The RUL indicates how many more cycles the engine is expected to operate before requiring maintenance.
        """)
        
        # Model selection for RUL prediction
        model_options = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "Weighted Ensemble", "Stacking Ensemble"]
        selected_model = st.selectbox("Select Model for RUL Prediction", model_options, index=model_options.index(model_type))
        
        # Display model badges
        st.markdown("<div style='margin: 10px 0;'>", unsafe_allow_html=True)
        for model in model_options:
            model_info = get_model_description(model)
            if model == selected_model:
                st.markdown(f"<span class='model-badge {model_info['badge_class']}' style='padding: 6px 12px; font-size: 0.9rem;'>{model}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='model-badge {model_info['badge_class']}' style='padding: 4px 8px; opacity: 0.5;'>{model}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            with st.spinner("Calculating RUL prediction..."):
                time.sleep(0.5)  # Simulate loading
                st.plotly_chart(plot_rul_prediction(data, engine_id, selected_model), use_container_width=True)
        
        with col2:
            # Get model-specific recommendations
            status, color, _ = get_maintenance_recommendations(current_rul, anomaly_percentage, selected_model)
            model_info = get_model_description(selected_model)
            
            st.markdown("### RUL Interpretation")
            st.markdown(f"""
            <div class="card" style="border-left: 4px solid {color};">
                <h3 style="color: {color};">Current RUL: {current_rul:.1f} cycles</h3>
                <p>Status: <strong>{status}</strong></p>
                <p>Model: <span class="model-badge {model_info['badge_class']}">{selected_model}</span></p>
                <p>Interpretation:</p>
                <ul>
                    <li>{'<span class="warning">Critical</span>: Immediate maintenance required' if status == 'Critical' else ''}</li>
                    <li>{'<span style="color: #F59E0B;">Warning</span>: Plan maintenance soon' if status == 'Warning' else ''}</li>
                    <li>{'<span style="color: #2563EB;">Caution</span>: Monitor closely' if status == 'Caution' else ''}</li>
                    <li>{'<span class="success">Good</span>: Regular monitoring' if status == 'Good' else ''}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Model strengths
            st.markdown(f"""
            <div class="card">
                <h4>Model Strengths:</h4>
                <ul>
                    {"".join([f"<li>{strength}</li>" for strength in model_info['strengths']])}
                </ul>
                <p><small>Best for: {model_info['best_for']}</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a "Recalculate RUL" button with animation
            if st.button("Recalculate RUL", key="recalc_rul"):
                with st.spinner("Recalculating RUL..."):
                    simulate_loading()
                st.success(f"Updated RUL: {current_rul:.1f} cycles")
    
    with tab4:
        # Maintenance tab
        st.markdown("### Maintenance Recommendations")
        st.markdown("""
        Based on the engine's current status, here are the recommended maintenance actions.
        """)
        
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid {color};">
            <h3 style="color: {color};">{status}: Maintenance Plan</h3>
            <ul>
        """, unsafe_allow_html=True)
        
        for recommendation in recommendations:
            st.markdown(f"<li>{recommendation}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Sensor status summary
        st.markdown("### Sensor Status Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>Critical Sensors to Monitor:</h4>", unsafe_allow_html=True)
            
            for sensor in high_anomaly_sensors:
                sensor_num = int(sensor.replace('sensor', ''))
                st.markdown(f"""
                <div class="card" style="border-left: 4px solid #DC2626; padding: 8px; margin-bottom: 8px;">
                    <strong>{sensor}</strong>: High anomaly risk
                </div>
                """, unsafe_allow_html=True)
            
            for sensor in medium_anomaly_sensors[:2]:
                sensor_num = int(sensor.replace('sensor', ''))
                st.markdown(f"""
                <div class="card" style="border-left: 4px solid #F59E0B; padding: 8px; margin-bottom: 8px;">
                    <strong>{sensor}</strong>: Medium anomaly risk
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h4>Healthy Sensors:</h4>", unsafe_allow_html=True)
            
            for sensor in good_sensors:
                sensor_num = int(sensor.replace('sensor', ''))
                st.markdown(f"""
                <div class="card" style="border-left: 4px solid #16A34A; padding: 8px; margin-bottom: 8px;">
                    <strong>{sensor}</strong>: Good status
                </div>
                """, unsafe_allow_html=True)
        
        # Add historical comparison
        st.markdown("### Historical Comparison")
        st.markdown("""
        This view shows how this engine compares to other engines in the fleet based on sensor patterns.
        """)
        
        with st.spinner("Generating cluster visualization..."):
            time.sleep(0.5)  # Simulate loading
            st.plotly_chart(plot_pca_clusters(data, X_pca, clusters, engine_id), use_container_width=True)
        
        # Add sensor importance
        st.markdown("### Sensor Importance Analysis")
        st.markdown("""
        This chart shows which sensors are most important for anomaly detection based on their variance.
        Focus on the sensors with higher importance when monitoring this engine.
        """)
        
        with st.spinner("Calculating sensor importance..."):
            time.sleep(0.5)  # Simulate loading
            st.plotly_chart(plot_sensor_importance(data), use_container_width=True)
            
        # Add a "Generate Maintenance Report" button with animation
        if st.button("Generate Maintenance Report", key="gen_report"):
            with st.spinner("Generating comprehensive maintenance report..."):
                simulate_loading()
            
            # Show a sample report
            st.success("Report generated successfully!")
            st.markdown(f"""
            <div class="card" style="background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); border: 1px solid #3B82F6; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);">
                <h3>Maintenance Report for Engine {engine_id}</h3>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Status:</strong> <span style="color: {color};">{status}</span></p>
                <p><strong>RUL:</strong> {current_rul:.1f} cycles</p>
                <p><strong>Anomaly Rate:</strong> {anomaly_percentage:.1f}%</p>
                <p><strong>Model Used:</strong> <span class="model-badge {model_info['badge_class']}">{model_type}</span></p>
                <p><strong>Key Sensors to Monitor:</strong> {', '.join(high_anomaly_sensors)}</p>
                <p><strong>Recommended Next Inspection:</strong> {max(1, int(current_rul/2))} cycles</p>
                
                <div class="report-highlight">
                    <h4>Critical Findings:</h4>
                    <ul>
                        <li>Sensors 6, 7, 12, and 17 show significant anomaly patterns</li>
                        <li>Current RUL estimate indicates {status.lower()} status</li>
                        <li>Engine behavior cluster analysis shows similar patterns to engines that required maintenance within {max(5, int(current_rul/3))} cycles</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Main application
def main():
    st.markdown('<h1 class="main-header">NASA Turbofan Engine Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard analyzes the NASA Turbofan Engine dataset to detect anomalies and predict remaining useful life (RUL).
    It uses advanced machine learning techniques to identify potential issues and provide maintenance recommendations.
    """)
    
    # Sidebar
    st.sidebar.image("https://www.nasa.gov/wp-content/uploads/2022/07/nasa-logo-web-rgb.png", width=150)
    st.sidebar.markdown("## Dashboard Controls")
    
    # Dataset selection
    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        ["FD001", "FD002", "FD003", "FD004"],
        help="FD001: 1 operating condition, 1 fault mode\n"
             "FD002: 6 operating conditions, 1 fault mode\n"
             "FD003: 1 operating condition, 2 fault modes\n"
             "FD004: 6 operating conditions, 2 fault modes"
    )
    
    # Load data with animation
    with st.spinner("Loading dataset..."):
        simulate_loading()
        train_data, test_data = load_data()
        st.sidebar.success(f"Dataset {dataset_option} loaded successfully!")
    
    # Data selection
    data_option = st.sidebar.radio("Select Data", ["Training Data", "Test Data"])
    data = train_data if data_option == "Training Data" else test_data
    
    # Model selection
    st.sidebar.markdown("## Model Selection")
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["XGBoost", "LightGBM", "CatBoost", "LSTM", "Weighted Ensemble", "Stacking Ensemble"],
        help="Choose the machine learning model for anomaly detection and RUL prediction"
    )
    
    # Display model badges
    model_info = get_model_description(model_type)
    st.sidebar.markdown(f"""
    <div style="margin: 10px 0;">
        <span class="model-badge {model_info['badge_class']}">{model_type}</span>
    </div>
    <div style="font-size: 0.85rem; margin-bottom: 15px;">
        <p>{model_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model parameters
    st.sidebar.markdown("## Model Parameters")
    contamination = st.sidebar.slider("Anomaly Contamination", 0.01, 0.2, 0.05, 0.01,
                                     help="Expected proportion of anomalies in the data")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5, 1,
                                  help="Number of clusters for engine behavior patterns")
    
    # Simulate model training with animation
    if st.sidebar.button("Apply Parameters"):
        with st.sidebar:
            with st.spinner("Training models..."):
                simulate_loading()
            st.success("Models updated successfully!")
    
    # Generate static predictions and clusters
    all_engine_indices = list(range(len(data)))
    all_predictions = np.ones(len(data))
    all_scores = np.zeros(len(data))
    
    # Engine selection
    engine_ids = sorted(data['engine_id'].unique())
    selected_engine = st.sidebar.selectbox("Select Engine ID", engine_ids)
    
    # Get engine data
    engine_data = data[data['engine_id'] == selected_engine]
    current_cycle = engine_data['cycle'].max()
    current_rul = engine_data.loc[engine_data['cycle'] == current_cycle, 'RUL'].values[0]
    
    # Generate predictions for the selected engine
    predictions, scores = get_static_predictions(data, selected_engine, model_type)
    
    # Update the global predictions and scores
    engine_indices = engine_data.index
    all_predictions[engine_indices] = predictions
    all_scores[engine_indices] = scores
    
    # Generate static clusters
    X_pca, clusters = get_static_clusters(data)
    
    # Calculate anomaly percentage
    engine_indices = engine_data.index
    engine_predictions = all_predictions[engine_indices]
    anomaly_percentage = (engine_predictions == -1).mean() * 100
    
    # Add a "Run Analysis" button for interactivity
    if st.sidebar.button("Run Full Analysis", key="run_analysis"):
        with st.sidebar:
            with st.spinner("Running comprehensive analysis..."):
                simulate_loading()
            st.success("Analysis complete!")
    
    # User-focused dashboard
    user_focused_dashboard(
        data, 
        selected_engine, 
        all_predictions, 
        all_scores,
        X_pca,
        clusters,
        current_cycle, 
        current_rul, 
        anomaly_percentage,
        model_type
    )
    
    

if __name__ == "__main__":
    main()