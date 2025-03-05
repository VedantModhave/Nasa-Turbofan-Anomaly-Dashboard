import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="NASA Turbofan Anomaly Detection", layout="wide")

# Custom color schemes for different sensor types
COLOR_SCHEMES = {
    'temperature': ['#FF9B9B', '#FF5D5D', '#FF1F1F', '#CC0000'],
    'pressure': ['#94C5FF', '#4A90E2', '#357ABD', '#1B5899'],
    'speed': ['#98FF98', '#50C878', '#228B22', '#006400'],
    'vibration': ['#FFB366', '#FF9933', '#FF8000', '#CC6600']
}

def load_data():
    # Simulated data generation for demonstration
    n_engines = 100
    n_timestamps = 300
    
    data = []
    for engine_id in range(1, n_engines + 1):
        timestamps = range(1, n_timestamps + 1)
        base_deterioration = np.linspace(0, 1, n_timestamps)
        
        # Generate more dynamic and varied patterns for each sensor
        temp_pattern = base_deterioration + np.sin(np.linspace(0, 8*np.pi, n_timestamps)) * 0.3
        pressure_pattern = base_deterioration + np.cos(np.linspace(0, 6*np.pi, n_timestamps)) * 0.25
        speed_pattern = base_deterioration + np.sin(np.linspace(0, 4*np.pi, n_timestamps)) * 0.2
        vibration_pattern = base_deterioration + np.sin(np.linspace(0, 10*np.pi, n_timestamps)) * 0.35
        
        # Add random noise and trends
        noise_level = 0.1
        for t in timestamps:
            temp = temp_pattern[t-1] + np.random.normal(0, noise_level)
            pressure = pressure_pattern[t-1] + np.random.normal(0, noise_level)
            speed = speed_pattern[t-1] + np.random.normal(0, noise_level)
            vibration = vibration_pattern[t-1] + np.random.normal(0, noise_level)
            
            data.append({
                'engine_id': engine_id,
                'timestamp': t,
                'temperature': temp * 100 + 350,  # Scale to realistic temperature range
                'pressure': pressure * 50 + 100,  # Scale to realistic pressure range
                'speed': speed * 1000 + 2000,    # Scale to realistic speed range
                'vibration': vibration * 10 + 5   # Scale to realistic vibration range
            })
    
    return pd.DataFrame(data)

def detect_anomalies(data, contamination=0.1):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(scaled_data)
    return anomalies == -1

def create_sensor_plot(df, engine_id, sensor_name, color_scheme):
    engine_data = df[df['engine_id'] == engine_id]
    
    # Detect anomalies for this sensor
    anomalies = detect_anomalies(engine_data[[sensor_name]].values)
    
    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add main sensor reading line
    fig.add_trace(
        go.Scatter(
            x=engine_data['timestamp'],
            y=engine_data[sensor_name],
            name=sensor_name.capitalize(),
            line=dict(color=color_scheme[1], width=2),
            mode='lines'
        )
    )
    
    # Add anomaly points
    anomaly_data = engine_data[anomalies]
    if not anomaly_data.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data[sensor_name],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='x'
                )
            )
        )
    
    # Update layout for better visibility
    fig.update_layout(
        title=f"{sensor_name.capitalize()} Readings for Engine {engine_id}",
        xaxis_title="Time",
        yaxis_title=sensor_name.capitalize(),
        plot_bgcolor='rgba(240,240,240,0.95)',  # Light gray background
        paper_bgcolor='white',
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    
    # Add grid lines for better readability
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=False
    )
    
    return fig

def main():
    st.title("🚀 NASA Turbofan Engine Anomaly Detection")
    
    # Load data
    df = load_data()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    engine_id = st.sidebar.selectbox(
        "Select Engine ID",
        options=sorted(df['engine_id'].unique()),
        index=0
    )
    
    # Main content
    st.markdown("""
    ### Real-time Engine Monitoring Dashboard
    This dashboard shows sensor readings and detected anomalies for the selected engine.
    """)
    
    # Create two rows of plots using columns
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    with row1_col1:
        fig_temp = create_sensor_plot(df, engine_id, 'temperature', COLOR_SCHEMES['temperature'])
        st.plotly_chart(fig_temp, use_container_width=True)
        
    with row1_col2:
        fig_pressure = create_sensor_plot(df, engine_id, 'pressure', COLOR_SCHEMES['pressure'])
        st.plotly_chart(fig_pressure, use_container_width=True)
        
    with row2_col1:
        fig_speed = create_sensor_plot(df, engine_id, 'speed', COLOR_SCHEMES['speed'])
        st.plotly_chart(fig_speed, use_container_width=True)
        
    with row2_col2:
        fig_vibration = create_sensor_plot(df, engine_id, 'vibration', COLOR_SCHEMES['vibration'])
        st.plotly_chart(fig_vibration, use_container_width=True)
    
    # Add summary statistics
    st.markdown("### Engine Health Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    engine_data = df[df['engine_id'] == engine_id]
    with col1:
        st.metric(
            "Average Temperature",
            f"{engine_data['temperature'].mean():.1f}°C",
            f"{engine_data['temperature'].std():.1f}°C"
        )
    with col2:
        st.metric(
            "Average Pressure",
            f"{engine_data['pressure'].mean():.1f} PSI",
            f"{engine_data['pressure'].std():.1f} PSI"
        )
    with col3:
        st.metric(
            "Average Speed",
            f"{engine_data['speed'].mean():.0f} RPM",
            f"{engine_data['speed'].std():.0f} RPM"
        )
    with col4:
        st.metric(
            "Average Vibration",
            f"{engine_data['vibration'].mean():.2f} g",
            f"{engine_data['vibration'].std():.2f} g"
        )

if __name__ == "__main__":
    main()