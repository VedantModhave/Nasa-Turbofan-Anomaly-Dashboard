import React from 'react';
import { Rocket, AlertTriangle, Activity, BarChart3, Gauge, Settings } from 'lucide-react';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-blue-900 text-white p-4 shadow-md">
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Rocket size={24} />
            <h1 className="text-xl font-bold">NASA Turbofan Anomaly Detection</h1>
          </div>
          <div>
            <p className="text-sm">Powered by Streamlit & React</p>
          </div>
        </div>
      </header>

      <main className="container mx-auto p-6">
        <div className="bg-white p-8 rounded-lg shadow-md">
          <div className="flex items-center justify-center space-x-4 mb-8">
            <Rocket size={48} className="text-blue-600" />
            <div>
              <h2 className="text-2xl font-bold text-gray-800">NASA Turbofan Engine Anomaly Detection Dashboard</h2>
              <p className="text-gray-600">Interactive dashboard for detecting anomalies in turbofan engine data</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
              <h3 className="flex items-center text-lg font-semibold text-blue-800 mb-4">
                <Activity className="mr-2" size={20} />
                Real-time Anomaly Detection
              </h3>
              <p className="text-gray-700">
                Monitor engine health with advanced ML2-AD techniques that combine supervised and unsupervised learning
                to detect both known failure patterns and novel anomalies.
              </p>
            </div>

            <div className="bg-green-50 p-6 rounded-lg border border-green-200">
              <h3 className="flex items-center text-lg font-semibold text-green-800 mb-4">
                <Gauge className="mr-2" size={20} />
                Remaining Useful Life (RUL) Prediction
              </h3>
              <p className="text-gray-700">
                Predict when engines will require maintenance with our early warning system that estimates
                remaining useful life based on current sensor readings and historical patterns.
              </p>
            </div>

            <div className="bg-purple-50 p-6 rounded-lg border border-purple-200">
              <h3 className="flex items-center text-lg font-semibold text-purple-800 mb-4">
                <BarChart3 className="mr-2" size={20} />
                Interactive Visualizations
              </h3>
              <p className="text-gray-700">
                Explore sensor data through dynamic plots that highlight anomalies and trends.
                Compare different engines and sensor readings with our interactive interface.
              </p>
            </div>

            <div className="bg-yellow-50 p-6 rounded-lg border border-yellow-200">
              <h3 className="flex items-center text-lg font-semibold text-yellow-800 mb-4">
                <AlertTriangle className="mr-2" size={20} />
                Maintenance Recommendations
              </h3>
              <p className="text-gray-700">
                Receive actionable maintenance recommendations based on detected anomalies
                and predicted RUL to prevent failures and optimize maintenance schedules.
              </p>
            </div>
          </div>

          <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 mb-8">
            <h3 className="flex items-center text-lg font-semibold text-gray-800 mb-4">
              <Settings className="mr-2" size={20} />
              Getting Started
            </h3>
            <ol className="list-decimal list-inside space-y-2 text-gray-700">
              <li>Install the required Python packages using the provided requirements.txt file</li>
              <li>Run the Streamlit application with <code className="bg-gray-200 px-2 py-1 rounded">npm run streamlit</code></li>
              <li>Select a dataset and engine ID from the sidebar</li>
              <li>Explore sensor trends, anomalies, and RUL predictions</li>
              <li>Adjust model parameters to fine-tune anomaly detection</li>
            </ol>
          </div>

          <div className="text-center">
            <a 
              href="https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              Learn more about the NASA Turbofan Engine dataset
            </a>
          </div>
        </div>
      </main>

      <footer className="bg-gray-800 text-white p-4 mt-8">
        <div className="container mx-auto text-center">
          <p>NASA Turbofan Anomaly Detection Dashboard</p>
          <p className="text-sm text-gray-400">Using Streamlit, React, and Machine Learning</p>
        </div>
      </footer>
    </div>
  );
}

export default App;