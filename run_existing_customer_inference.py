import sys
import os
from pathlib import Path
# Add project root to path
sys.path.append(os.getcwd())

from prefect import flow, task
from dashboard import database
from src.models.predict import ModelPredictor
from src.utils.config import Config
import pandas as pd
from loguru import logger
from datetime import datetime

@task(name="Load DB Customers")
def load_db_customers():
    logger.info("Loading customers from DB...")
    df = database.load_customers()
    logger.info(f"Loaded {len(df)} customers.")
    return df

@task(name="Predict and Update")
def predict_and_update(df):
    if df.empty:
        logger.warning("No customers found.")
        return
    
    config = Config()
    model_path = config.models_path / "churn_model.pkl"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    predictor = ModelPredictor(model_path)
    
    # Prepare features
    # The ModelPredictor.predict method calls FeatureEngineer.preprocess_data
    # which drops customerID and handles TotalCharges.
    # However, we need to ensure the input df has the right columns and types.
    
    # We need to handle TotalCharges conversion here similar to crm.py just in case
    # because FeatureEngineer expects it to be convertible or already numeric.
    # But FeatureEngineer.preprocess_data does:
    # df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    # So passing the raw df from DB (which might have strings) should be fine.
    
    logger.info("Running predictions...")
    try:
        # Predict
        # Note: predictor.predict expects a DataFrame.
        # It will drop customerID internally for prediction, but we need to keep it to map back results.
        
        # We pass the whole DF. The FeatureEngineer inside predictor will handle dropping customerID for the model input.
        predictions = predictor.predict(df)
        probabilities = predictor.predict_proba(df)
        
        # Extract probability of class 1
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            churn_probs = probabilities[:, 1]
        else:
            churn_probs = probabilities
            
        # Update DataFrame
        df['churn_prediction'] = predictions
        df['churn_probability'] = churn_probs
        
        # Calculate confidence (max probability)
        # If prob > 0.5, confidence is prob. If prob <= 0.5, confidence is 1 - prob.
        # Or just max(p, 1-p)
        df['confidence'] = df['churn_probability'].apply(lambda p: max(p, 1-p))
        
        df['risk_level'] = df['churn_probability'].apply(
            lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low' if pd.notna(x) else 'Unknown'
        )
        
        df['prediction_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        logger.info("Saving predictions to DB...")
        database.save_predictions(df)
        logger.info("Database updated.")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e

@flow(name="Existing Customer Inference")
def db_inference_flow():
    df = load_db_customers()
    predict_and_update(df)

if __name__ == "__main__":
    db_inference_flow()
