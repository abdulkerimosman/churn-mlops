from prefect import flow, task
import pandas as pd
import numpy as np
from src.utils.config import Config
from src.monitoring.drift import DriftDetector
from loguru import logger

@task(name="Generate Drifted Data")
def generate_drifted_data():
    """Generate new data with intentional drift."""
    logger.info("Generating drifted data...")
    np.random.seed(99) # Different seed
    n_samples = 500
    
    # Simulate drift: Older customers, higher charges
    data = {
        "customerID": [f"DriftC{i:04d}" for i in range(n_samples)],
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples),
        # DRIFT: Tenure is much higher on average
        "tenure": np.random.randint(50, 72, n_samples), 
        "PhoneService": np.random.choice(["Yes", "No"], n_samples),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n_samples),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n_samples),
        # DRIFT: Monthly charges are higher
        "MonthlyCharges": np.random.uniform(80, 150, n_samples),
        "TotalCharges": np.random.uniform(20, 8000, n_samples),
        "Churn": np.random.choice(["Yes", "No"], n_samples),
    }
    
    return pd.DataFrame(data)

@task(name="Run Drift Analysis")
def run_drift_analysis(current_data):
    """Run drift analysis using DriftDetector."""
    logger.info("Running drift analysis...")
    
    detector = DriftDetector()
    drift_metrics = detector.detect_drift(current_data, save_report=True)
    
    return drift_metrics

@flow(name="Data Drift Monitoring")
def monitoring_flow():
    cur_df = generate_drifted_data()
    run_drift_analysis(cur_df)

if __name__ == "__main__":
    monitoring_flow()
