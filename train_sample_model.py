#!/usr/bin/env python3
"""Quick script to train a model for API testing."""

import pandas as pd
import numpy as np
from pathlib import Path

from src.models.train import ModelTrainer
from src.utils.config import Config

def create_sample_data():
    """Create sample churn data for training (matching Telco Churn schema)."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "customerID": [f"C{i:04d}" for i in range(n_samples)],
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples),
        "tenure": np.random.randint(0, 72, n_samples),
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
        "MonthlyCharges": np.random.uniform(20, 120, n_samples),
        "TotalCharges": np.random.uniform(20, 8000, n_samples),
        "Churn": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
    }
    
    return pd.DataFrame(data)

def main():
    """Train and save a model."""
    print("Creating sample data...")
    df = create_sample_data()
    
    # Save to processed data directory
    config = Config()
    config.data_processed_path.mkdir(parents=True, exist_ok=True)
    data_path = config.data_processed_path / "sample_churn_data.csv"
    df.to_csv(data_path, index=False)
    print(f"Saved sample data to {data_path}")
    
    # Train model
    print("Training model...")
    # Use xgboost as configured in model_config.yaml
    trainer = ModelTrainer(model_type="xgboost")
    
    X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(
        data_path=config.data_processed_path,
        filename="sample_churn_data.csv",
        target_column="Churn"
    )
    
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    
    # Save model
    model_path = config.models_path / "churn_model.pkl"
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Model metrics: {metrics}")
    
    # Log to MLflow
    trainer.log_to_mlflow()
    print("Logged to MLflow")

    # Save model
    config.models_path.mkdir(parents=True, exist_ok=True)
    model_path = config.models_path / "churn_model.pkl"
    saved_path = trainer.save_model(model_path)
    print(f"Model saved to {saved_path}")

if __name__ == "__main__":
    main()
