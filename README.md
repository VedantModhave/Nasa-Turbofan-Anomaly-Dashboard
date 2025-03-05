# NASA Turbofan Anomaly Detection Dashboard

An interactive dashboard for analyzing the NASA Turbofan Engine dataset to detect anomalies and predict remaining useful life (RUL).

## Features

- **Data Loading & Preprocessing**: Loads and preprocesses time-series sensor data from the NASA Turbofan Engine dataset
- **Anomaly Detection**: Implements ML2-AD techniques combining supervised and unsupervised learning
- **Interactive Visualizations**: Dynamic plots for sensor trends with anomaly highlighting
- **RUL Prediction**: Early warning system to predict remaining useful life
- **Maintenance Recommendations**: Actionable insights based on engine health
- **Clustering Analysis**: Groups engines by behavior patterns for comparative analysis

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Installation

1. Clone the repository
2. Install JavaScript dependencies:
   ```
   npm install
   ```
3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Streamlit application:
   ```
   npm run streamlit
   ```
2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Dataset Information

The NASA Turbofan Engine dataset contains run-to-failure data for multiple engines. Each engine starts with different degrees of initial wear and manufacturing variation, and develops a fault over time until system failure.

Available datasets:
- **FD001**: Single operating condition, single failure mode
- **FD002**: Six operating conditions, single failure mode
- **FD003**: Single operating condition, two failure modes
- **FD004**: Six operating conditions, two failure modes

## Dashboard Components

- **Engine Sensor Trends**: Visualize sensor readings over time
- **Anomaly Detection**: Highlight anomalous sensor readings
- **Remaining Useful Life Prediction**: Predict when maintenance will be required
- **Engine Health Summary**: Current status and metrics for selected engine
- **Maintenance Recommendations**: Actionable insights based on engine health
- **Engine Behavior Clustering**: Group engines by similar operational patterns
- **Sensor Importance Analysis**: Identify which sensors are most relevant for anomaly detection

## Technologies Used

- **Streamlit**: Interactive dashboard framework
- **React**: Frontend UI components
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive visualizations
- **Tailwind CSS**: Styling

## Data Source

[NASA Prognostics Center of Excellence Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)