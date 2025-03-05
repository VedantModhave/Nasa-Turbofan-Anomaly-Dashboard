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
- Streamlit
- Node.js

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

## Dashboard(Apologies in video as during making the video we had some design erros for the graph)
![1](https://github.com/user-attachments/assets/d6047a46-dbda-4030-9932-33abe9434dc9)

![2](https://github.com/user-attachments/assets/1179df88-5906-4e10-b9be-95bc861d2319)


![3](https://github.com/user-attachments/assets/330609a6-a9f3-4939-8db7-01308fb378e9)

![4](https://github.com/user-attachments/assets/043b0769-8f83-4fd9-8151-06b22ca427c0)

![5](https://github.com/user-attachments/assets/ef27f28d-ea3a-450f-99c6-35af550153c5)
![6](https://github.com/user-attachments/assets/1ee20598-9922-4b7d-9e47-adde7d96d272)
![7](https://github.com/user-attachments/assets/3d97d54e-e817-47ce-9a8d-354dc8f55f60)
![8](https://github.com/user-attachments/assets/f3ae2fe0-5d43-4dfa-8f81-fb60856f1b01)
![9](https://github.com/user-attachments/assets/57ddca18-55c8-4263-8fd5-9bb3f7d92556)

