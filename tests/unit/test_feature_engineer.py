"""Tests for feature engineering."""
import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer
from src.utils.exceptions import FeatureEngineeringError

def test_identify_features(sample_churn_data):
    """Test feature identification."""
    fe = FeatureEngineer()
    numeric, categorical = fe.identify_features(sample_churn_data)
    
    assert "tenure" in numeric
    assert "MonthlyCharges" in numeric
    assert "TotalCharges" in numeric
    assert "InternetService" in categorical
    assert "Contract" in categorical

def test_preprocess_data_training(sample_churn_data):
    """Test preprocessing for training data."""
    fe = FeatureEngineer()
    df_processed = fe.preprocess_data(sample_churn_data, is_training=True)
    
    assert df_processed.shape[0] == sample_churn_data.shape[0]
    # Check if TotalCharges is log transformed (should be float)
    assert df_processed["TotalCharges"].dtype == float
    # Check if categorical variables are encoded (should be numeric)
    assert "InternetService_Fiber optic" in df_processed.columns

def test_preprocess_data_inference(sample_churn_data):
    """Test preprocessing for inference data."""
    fe = FeatureEngineer()
    # Fit on training data first
    df_train = fe.preprocess_data(sample_churn_data, is_training=True)
    
    # Simulate what happens in train.py: remove target from feature_columns
    if "Churn" in fe.feature_columns:
        fe.feature_columns.remove("Churn")
    
    # Transform inference data (drop target)
    inference_data = sample_churn_data.drop("Churn", axis=1)
    df_processed = fe.preprocess_data(inference_data, is_training=False)
    
    assert df_processed.shape[0] == inference_data.shape[0]
    assert "Churn" not in df_processed.columns
    assert "InternetService_Fiber optic" in df_processed.columns