"""Tests for model training."""
import pytest
import tempfile
from pathlib import Path
from src.models.train import ModelTrainer
from src.utils.exceptions import ModelTrainingError
from src.features.engineering import FeatureEngineer
from sklearn.preprocessing import LabelEncoder

@pytest.fixture
def processed_data(sample_churn_data):
    """Preprocess data for training tests."""
    fe = FeatureEngineer()
    df_processed = fe.preprocess_data(sample_churn_data, is_training=True)
    
    le = LabelEncoder()
    if "Churn" in df_processed.columns:
        df_processed["Churn"] = le.fit_transform(df_processed["Churn"])
        
    X = df_processed.drop(columns=["Churn"])
    y = df_processed["Churn"]
    
    # Update feature columns
    fe.feature_columns = X.columns.tolist()
    
    return X, y, fe, le

def test_train_logistic_regression(processed_data):
    """Test logistic regression training."""
    X, y, fe, le = processed_data
    trainer = ModelTrainer(model_type="logistic_regression")
    trainer.feature_engineer = fe
    trainer.le = le
    
    trainer.train(X, y)
    assert trainer.model is not None

def test_train_random_forest(processed_data):
    """Test random forest training."""
    X, y, fe, le = processed_data
    trainer = ModelTrainer(model_type="random_forest")
    trainer.feature_engineer = fe
    trainer.le = le
    
    trainer.train(X, y)
    assert trainer.model is not None

def test_evaluate_metrics(processed_data):
    """Test model evaluation."""
    X, y, fe, le = processed_data
    trainer = ModelTrainer(model_type="xgboost")
    trainer.feature_engineer = fe
    trainer.le = le
    
    trainer.train(X, y)
    metrics = trainer.evaluate(X, y)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1

def test_save_model(temp_data_dir, processed_data):
    """Test model saving."""
    X, y, fe, le = processed_data
    trainer = ModelTrainer(model_type="xgboost")
    trainer.feature_engineer = fe
    trainer.le = le
    
    trainer.train(X, y)
    model_path = temp_data_dir / "test_model.pkl"
    saved_path = trainer.save_model(model_path)
    
    assert saved_path.exists()