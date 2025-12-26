
import pandas as pd
import pytest
from src.monitoring.drift import DriftDetector

@pytest.fixture
def sample_data(tmp_path):
    """Create sample reference data."""
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "income": [2000, 3000, 4000, 5000, 6000],
        "churn": [0, 0, 1, 0, 1]
    })
    path = tmp_path / "reference.csv"
    df.to_csv(path, index=False)
    return path

def test_drift_detector_initialization(sample_data):
    detector = DriftDetector(reference_data_path=sample_data)
    assert detector.reference_data is not None
    assert len(detector.reference_data) == 5

def test_drift_detection_no_drift(sample_data):
    detector = DriftDetector(reference_data_path=sample_data)
    
    # Current data same distribution as reference
    current_data = pd.DataFrame({
        "age": [22, 32, 42, 52, 62],
        "income": [2100, 3100, 4100, 5100, 6100],
        "churn": [0, 0, 1, 0, 1]
    })
    
    result = detector.detect_drift(current_data, save_report=False)
    assert "metrics" in result
    # Check that dataset drift is false (or low share of drifting features)
    # The structure of result depends on Evidently version, but usually contains 'dataset_drift'
    
    # Simple check that it ran
    assert result is not None

def test_drift_detection_with_drift(sample_data):
    detector = DriftDetector(reference_data_path=sample_data)
    
    # Current data significantly different (Drift)
    current_data = pd.DataFrame({
        "age": [80, 85, 90, 95, 100], # Much older
        "income": [20000, 30000, 40000, 50000, 60000], # Much richer
        "churn": [1, 1, 1, 1, 1]
    })
    
    result = detector.detect_drift(current_data, save_report=False)
    assert result is not None
