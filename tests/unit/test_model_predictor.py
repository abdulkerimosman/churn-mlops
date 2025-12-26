"""Tests for model prediction."""
import pytest
import tempfile
from pathlib import Path
from src.models.predict import ModelPredictor
from src.utils.exceptions import PredictionError

def test_predictor_model_not_found():
    """Test predictor with missing model file."""
    with pytest.raises(PredictionError):
        ModelPredictor(Path("nonexistent_model.pkl"))

def test_predict_proba(sample_churn_data, temp_data_dir):
    """Test probability predictions."""
    from src.models.train import ModelTrainer
    from src.features.engineering import FeatureEngineer
    from sklearn.preprocessing import LabelEncoder
    
    trainer = ModelTrainer(model_type="xgboost")
    fe = FeatureEngineer()
    
    df = sample_churn_data.copy()
    df_processed = fe.preprocess_data(df, is_training=True)
    
    if "Churn" in df_processed.columns:
        le = LabelEncoder()
        df_processed["Churn"] = le.fit_transform(df_processed["Churn"])
        
        X = df_processed.drop(columns=["Churn"])
        y = df_processed["Churn"]
        
        fe.feature_columns = X.columns.tolist()
        trainer.feature_engineer = fe
        trainer.le = le
        
        trainer.train(X, y)
        model_path = temp_data_dir / "test_model.pkl"
        trainer.save_model(model_path)
        
        predictor = ModelPredictor(model_path)
        
        # Predictor expects raw data (without target)
        X_raw = sample_churn_data.drop("Churn", axis=1)
        proba = predictor.predict_proba(X_raw)
        
        assert proba.shape == (len(X_raw), 2)
        assert (proba >= 0).all() and (proba <= 1).all()

def test_predict_confidence(sample_churn_data, temp_data_dir):
    """Test predictions with confidence."""
    from src.models.train import ModelTrainer
    
    trainer = ModelTrainer(model_type="xgboost")
    
    # Manually process
    from src.features.engineering import FeatureEngineer
    fe = FeatureEngineer()
    
    df = sample_churn_data.copy()
    df_processed = fe.preprocess_data(df, is_training=True)
    
    if "Churn" in df_processed.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_processed["Churn"] = le.fit_transform(df_processed["Churn"])
        
        X = df_processed.drop(columns=["Churn"])
        y = df_processed["Churn"]
        
        fe.feature_columns = X.columns.tolist()
        trainer.feature_engineer = fe
        trainer.le = le
        
        trainer.train(X, y)
        model_path = temp_data_dir / "test_model.pkl"
        trainer.save_model(model_path)
        
        predictor = ModelPredictor(model_path)
        
        # Predictor expects raw data (without target)
        X_raw = sample_churn_data.drop("Churn", axis=1)
        result = predictor.predict_with_confidence(X_raw)
        
        assert "predictions" in result
        assert "probabilities" in result
        assert "confidence" in result