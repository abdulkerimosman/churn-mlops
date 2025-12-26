"""Tests for data loading functionality."""
import pandas as pd
import pytest
from src.data.loader import DataLoader
from src.utils.exceptions import DataValidationError

def test_load_csv_file_not_found(temp_data_dir):
    """Test loading non-existent CSV raises error."""
    loader = DataLoader(temp_data_dir)
    with pytest.raises(DataValidationError):
        loader.load_csv("nonexistent.csv")

def test_load_csv_success(temp_data_dir, sample_churn_data):
    """Test successful CSV loading."""
    file_path = temp_data_dir / "test_data.csv"
    sample_churn_data.to_csv(file_path, index=False)
    
    loader = DataLoader(temp_data_dir)
    df = loader.load_csv("test_data.csv")
    
    assert df.shape == sample_churn_data.shape
    assert list(df.columns) == list(sample_churn_data.columns)

def test_save_csv(temp_data_dir, sample_churn_data):
    """Test CSV saving."""
    loader = DataLoader(temp_data_dir)
    saved_path = loader.save_csv(sample_churn_data, "output.csv", temp_data_dir)
    
    assert saved_path.exists()
    df = pd.read_csv(saved_path)
    assert df.shape == sample_churn_data.shape
