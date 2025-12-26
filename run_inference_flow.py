from prefect import flow, task
import pandas as pd
import numpy as np
from src.models.predict import ModelPredictor
from src.utils.config import Config
from loguru import logger

@task(name="Load Batch Data")
def load_batch_data():
    """Simulate loading a batch of new customers for prediction."""
    logger.info("Loading batch data...")
    # Create dummy data matching the schema
    n_samples = 50
    data = {
        "customerID": [f"NewC{i:04d}" for i in range(n_samples)],
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
    }
    return pd.DataFrame(data)

@task(name="Run Predictions")
def predict_batch(df: pd.DataFrame):
    """Run predictions on the batch."""
    config = Config()
    model_path = config.models_path / "churn_model.pkl"
    
    logger.info(f"Loading model from {model_path}")
    predictor = ModelPredictor(model_path)
    
    logger.info("Running inference...")
    predictions = predictor.predict(df)
    probabilities = predictor.predict_proba(df)
    
    # Add results to dataframe
    df["Churn_Prediction"] = predictions
    # Take probability of positive class (1)
    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
        df["Churn_Probability"] = probabilities[:, 1]
    else:
        df["Churn_Probability"] = probabilities
    
    return df

@task(name="Save Predictions")
def save_results(df: pd.DataFrame):
    """Save prediction results."""
    config = Config()
    output_path = config.data_processed_path / "batch_predictions.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    return output_path

@flow(name="Batch Inference Pipeline")
def inference_flow():
    df = load_batch_data()
    results_df = predict_batch(df)
    save_results(results_df)

if __name__ == "__main__":
    inference_flow()
