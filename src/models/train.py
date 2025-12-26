"""Model training pipeline."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.features.engineering import FeatureEngineer
from src.utils.exceptions import ModelTrainingError


class ModelTrainer:
    """Train and evaluate churn prediction models."""
    
    def __init__(
        self,
        experiment_name: str = "churn-prediction",
        model_type: str = "xgboost",
        random_state: int = 42,
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            experiment_name: MLflow experiment name
            model_type: Type of model ("logistic_regression", "random_forest", "xgboost", "gradient_boosting")
            random_state: Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.metrics = {}
        self.le = LabelEncoder()
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
    
    def load_and_prepare_data(
        self,
        data_path: Path,
        filename: str,
        target_column: str = "Churn",
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to data directory
            filename: Name of data file
            target_column: Name of target column
            test_size: Proportion of test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Load data
            loader = DataLoader(data_path)
            df = loader.load_csv(filename)
            
            # Validate data
            validator = DataValidator()
            # validator.check_missing_values(df) # Skipped as we handle it in preprocessing
            # validator.check_duplicates(df)
            
            logger.info(f"Loaded data shape: {df.shape}")
            
            # Preprocess data (Feature Engineering)
            # We pass the whole dataframe to handle encoding correctly
            df_processed = self.feature_engineer.preprocess_data(df, is_training=True)
            
            # Separate features and target
            # Note: preprocess_data might have encoded the target if it was in the df
            # But our preprocess_data implementation currently doesn't explicitly handle target encoding
            # except if it was part of the binary/nominal columns.
            # The notebook does explicit LabelEncoding for Churn.
            
            if target_column in df_processed.columns:
                # Encode target if it's string
                if df_processed[target_column].dtype == 'object' or df_processed[target_column].dtype.name == 'category':
                     df_processed[target_column] = self.le.fit_transform(df_processed[target_column])
                
                X = df_processed.drop(columns=[target_column])
                y = df_processed[target_column]
                
                # Update feature columns to exclude target
                self.feature_engineer.feature_columns = X.columns.tolist()
            else:
                # If target was dropped or not present (should not happen in training)
                raise ModelTrainingError(f"Target column '{target_column}' not found in processed data")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            raise ModelTrainingError(f"Data preparation failed: {str(e)}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Optional[Dict] = None,
    ) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameters: Model hyperparameters
        """
        try:
            # Features are already transformed in load_and_prepare_data for training
            # But if we were to use a pipeline, we would do it here.
            # Since we did it eagerly, we just fit the model.
            
            # Initialize model
            if self.model_type == "logistic_regression":
                self.model = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced',
                    **(hyperparameters or {}),
                )
            elif self.model_type == "random_forest":
                self.model = RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    **(hyperparameters or {}),
                )
            elif self.model_type == "gradient_boosting":
                self.model = GradientBoostingClassifier(
                    random_state=self.random_state,
                    **(hyperparameters or {}),
                )
            elif self.model_type == "xgboost":
                # Calculate scale_pos_weight for imbalance
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                self.model = xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    scale_pos_weight=scale_pos_weight,
                    **(hyperparameters or {}),
                )
            else:
                raise ModelTrainingError(f"Unknown model type: {self.model_type}")
            
            # Train model
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.model_type} model")
        
        except Exception as e:
            raise ModelTrainingError(f"Training failed: {str(e)}")
    
    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if self.model is None:
                raise ModelTrainingError("Model not trained yet")
            
            # Features are already transformed
            
            # Get predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            self.metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
            }
            
            logger.info(f"Evaluation metrics: {self.metrics}")
            
            return self.metrics
        
        except Exception as e:
            raise ModelTrainingError(f"Evaluation failed: {str(e)}")
        
        except Exception as e:
            raise ModelTrainingError(f"Evaluation failed: {str(e)}")
    
    def log_to_mlflow(self, hyperparameters: Optional[Dict] = None) -> None:
        """
        Log model and metrics to MLflow.
        
        Args:
            hyperparameters: Model hyperparameters
        """
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("model_type", self.model_type)
                if hyperparameters:
                    for key, value in hyperparameters.items():
                        mlflow.log_param(key, value)
                
                # Log metrics
                for metric_name, metric_value in self.metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                logger.info("Logged model and metrics to MLflow")
        
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
    
    def save_model(self, model_path: Path) -> Path:
        """
        Save trained model to disk.
        
        Args:
            model_path: Path to save model
            
        Returns:
            Path to saved model
        """
        import joblib
        
        try:
            if self.model is None:
                raise ModelTrainingError("Model not trained yet")
            
            model_path.parent.mkdir(parents=True, exist_ok=True)
            artifact = {"model": self.model, "feature_engineer": self.feature_engineer}
            joblib.dump(artifact, model_path)
            logger.info(f"Saved model + transformer to {model_path}")
            return model_path
        
        except Exception as e:
            raise ModelTrainingError(f"Failed to save model: {str(e)}")
        

import argparse
from src.utils.config import load_yaml_config
from src.utils.logger import setup_logging

def train_from_config(config_path: Path, disable_mlflow: bool = False, model_type_override: str = None) -> Path:
    cfg = load_yaml_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    trainer = ModelTrainer(
        experiment_name=cfg["mlflow"]["experiment_name"],
        model_type=model_type_override or cfg["model"]["type"],
        random_state=cfg["training"]["random_state"],
    )
    X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(
        Path(cfg["data"]["processed_data_path"]),
        cfg["data"]["filename"],
        cfg["data"]["target_column"],
        cfg["data"]["test_size"],
    )
    trainer.train(X_train, y_train, cfg["model"].get("hyperparameters"))
    trainer.evaluate(X_test, y_test)
    if not disable_mlflow:
        trainer.log_to_mlflow(cfg["model"].get("hyperparameters"))
    return trainer.save_model(Path(cfg["artifacts"]["model_path"]))

def main():
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--model-type", type=str, default=None)
    args = parser.parse_args()
    setup_logging()
    model_path = train_from_config(args.config, disable_mlflow=args.no_mlflow, model_type_override=args.model_type)
    logger.info(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    main()