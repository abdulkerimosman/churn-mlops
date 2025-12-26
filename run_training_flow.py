from prefect import flow, task
import pandas as pd
import numpy as np
from src.models.train import ModelTrainer
from src.utils.config import Config
from loguru import logger

@task(name="Ingest Data", retries=3)
def ingest_data_task():
    """Load real data from raw directory and save to processed."""
    logger.info("Ingesting real data...")
    config = Config()
    
    # Source path (Raw Data)
    raw_data_path = config.data_raw_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_data_path}")
        
    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded raw data: {df.shape}")
    
    # Save to processed data directory
    config.data_processed_path.mkdir(parents=True, exist_ok=True)
    processed_data_path = config.data_processed_path / "churn_data.csv"
    df.to_csv(processed_data_path, index=False)
    logger.info(f"Saved processed data to {processed_data_path}")
    
    return processed_data_path

@task(name="Train and Evaluate")
def train_model_task(data_path, hyperparameters=None):
    """Train the model using the generated data."""
    logger.info("Starting training task...")
    config = Config()
    trainer = ModelTrainer(model_type="xgboost")
    
    # Load and split
    # Note: data_path is the full path to the file, but load_and_prepare_data expects dir and filename
    # We can extract them
    
    X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(
        data_path=data_path.parent,
        filename=data_path.name,
        target_column="Churn"
    )
    
    # Train
    trainer.train(X_train, y_train, hyperparameters=hyperparameters)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    
    # Log to MLflow
    trainer.log_to_mlflow(hyperparameters=hyperparameters)
    
    # Save model
    model_path = config.models_path / "churn_model.pkl"
    trainer.save_model(model_path)
    
    return metrics

@flow(name="Churn Prediction Pipeline")
def main_flow(max_depth: int = 3, learning_rate: float = 0.1):
    data_path = ingest_data_task()
    
    # Define hyperparameters
    params = {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": 100
    }
    
    metrics = train_model_task(data_path, hyperparameters=params)
    logger.info(f"Flow completed with metrics: {metrics}")
    params = {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": 100
    }
    
    metrics = train_model_task(data_path, hyperparameters=params)
    logger.info(f"Flow completed with metrics: {metrics}")

if __name__ == "__main__":
    main_flow()
