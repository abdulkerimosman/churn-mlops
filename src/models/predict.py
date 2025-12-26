"""Model inference and prediction."""

from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.features.engineering import FeatureEngineer
from src.utils.exceptions import PredictionError


class ModelPredictor:
    """Handle model inference and predictions."""
    
    def __init__(self, model_path: Path):
        """
        Initialize ModelPredictor.
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model from disk."""
        try:
            if not self.model_path.exists():
                raise PredictionError(f"Model file not found: {self.model_path}")
            
            artifact = joblib.load(self.model_path)
            if isinstance(artifact, dict) and "model" in artifact and "feature_engineer" in artifact:
                self.model = artifact["model"]
                self.feature_engineer = artifact["feature_engineer"]
            else:
                self.model = artifact
            logger.info(f"Loaded model from {self.model_path}")
        
        except Exception as e:
            raise PredictionError(f"Failed to load model: {str(e)}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels (0 or 1)
        """
        try:
            if self.model is None:
                raise PredictionError("Model not loaded")
            
            if isinstance(X, pd.DataFrame):
                # Use preprocess_data instead of transform
                X_transformed = self.feature_engineer.preprocess_data(X, is_training=False)
            else:
                # If numpy array, assume already transformed (not recommended)
                X_transformed = X
            
            predictions = self.model.predict(X_transformed)
            return predictions
        
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        try:
            if self.model is None:
                raise PredictionError("Model not loaded")
            
            if isinstance(X, pd.DataFrame):
                # Use preprocess_data instead of transform
                X_transformed = self.feature_engineer.preprocess_data(X, is_training=False)
            else:
                X_transformed = X
            
            probabilities = self.model.predict_proba(X_transformed)
            return probabilities
        
        except Exception as e:
            raise PredictionError(f"Probability prediction failed: {str(e)}")
    
    def predict_with_confidence(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            predictions = self.predict(X)
            probabilities = self.predict_proba(X)
            
            # Get maximum probability (confidence)
            confidence = np.max(probabilities, axis=1)
            
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "confidence": confidence,
            }
        
        except Exception as e:
            raise PredictionError(f"Confidence prediction failed: {str(e)}")