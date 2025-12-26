"""Shared test fixtures and configuration."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_churn_data():
    """Generate sample churn dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "customerID": [f"C{i:04d}" for i in range(n_samples)],
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples),
        "tenure": np.random.randint(0, 73, n_samples),
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
        "MonthlyCharges": np.random.uniform(18.25, 118.75, n_samples),
        "TotalCharges": np.random.uniform(18.8, 8684.8, n_samples),
        "Churn": np.random.choice(["Yes", "No"], n_samples)
    }
    return pd.DataFrame(data)

@pytest.fixture
def feature_engineer():
    """Initialize FeatureEngineer for testing."""
    from src.features.engineering import FeatureEngineer
    return FeatureEngineer()

@pytest.fixture
def model_trainer():
    """Initialize ModelTrainer for testing."""
    from src.models.train import ModelTrainer
    return ModelTrainer(model_type="logistic_regression", random_state=42)