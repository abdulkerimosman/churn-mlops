"""Custom exceptions for the pipeline."""


class ChurnPipelineException(Exception):
    """Base exception for churn pipeline."""
    pass


class DataValidationError(ChurnPipelineException):
    """Raised when data validation fails."""
    pass


class FeatureEngineeringError(ChurnPipelineException):
    """Raised during feature engineering."""
    pass


class ModelTrainingError(ChurnPipelineException):
    """Raised during model training."""
    pass


class PredictionError(ChurnPipelineException):
    """Raised during model prediction."""
    pass


class ConfigurationError(ChurnPipelineException):
    """Raised when configuration is invalid."""
    pass