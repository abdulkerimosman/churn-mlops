"""Tests for data validation."""
import pytest
import pandas as pd
from src.data.validator import DataValidator
from src.utils.exceptions import DataValidationError

def test_check_missing_values(sample_churn_data):
    """Test missing value detection."""
    df = sample_churn_data.copy()
    df.loc[0, "age"] = None
    
    missing = DataValidator.check_missing_values(df, threshold=0.5)
    assert "age" in missing
    assert missing["age"] > 0

def test_check_duplicates(sample_churn_data):
    """Test duplicate detection."""
    df = sample_churn_data.copy()
    df = pd.concat([df, df.iloc[0:5]], ignore_index=True)
    
    duplicates = DataValidator.check_duplicates(df)
    assert duplicates > 0

def test_get_data_summary(sample_churn_data):
    """Test data summary generation."""
    summary = DataValidator.get_data_summary(sample_churn_data)
    
    assert "shape" in summary
    assert "columns" in summary
    assert "dtypes" in summary
    assert summary["shape"] == sample_churn_data.shape