"""Data validation using Pandera schemas."""

from typing import Dict, List, Optional

import pandas as pd
import pandera as pa
from loguru import logger

from src.utils.exceptions import DataValidationError


class DataValidator:
    """Validate data against defined schemas."""
    
    # Define schema for customer churn data
    CHURN_SCHEMA = pa.DataFrameSchema(
        {
            "customer_id": pa.Column(pa.String, nullable=False, unique=True),
            "age": pa.Column(pa.Int64, checks=pa.Check.in_range(min_value=18, max_value=100)),
            "tenure_months": pa.Column(pa.Int64, checks=pa.Check.ge(0)),
            "monthly_charges": pa.Column(pa.Float64, checks=pa.Check.ge(0)),
            "total_charges": pa.Column(pa.Float64, checks=pa.Check.ge(0)),
            "churn": pa.Column(pa.Bool, nullable=False),
        },
        strict=False,  # Allow extra columns
        coerce=True,
    )
    
    @classmethod
    def validate_churn_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate customer churn data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            validated_df = cls.CHURN_SCHEMA.validate(df)
            logger.info(f"Data validation passed: {validated_df.shape[0]} rows")
            return validated_df
        except pa.errors.SchemaError as e:
            raise DataValidationError(f"Schema validation failed: {str(e)}")
        except Exception as e:
            raise DataValidationError(f"Data validation error: {str(e)}")
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        """
        Check for missing values.
        
        Args:
            df: DataFrame to check
            threshold: Maximum allowed missing value ratio
            
        Returns:
            Dictionary of columns with missing values and their ratios
        """
        missing = df.isnull().sum() / len(df)
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if (missing > threshold).any():
            high_missing = missing[missing > threshold]
            logger.warning(f"Columns with >{threshold*100}% missing values:\n{high_missing}")
        
        return missing.to_dict()
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
        """
        Check for duplicate rows.
        
        Args:
            df: DataFrame to check
            subset: Columns to consider for duplicates
            
        Returns:
            Number of duplicate rows
        """
        duplicates = df.duplicated(subset=subset, keep=False).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        return duplicates
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with data summary
        """
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "numeric_stats": df.describe().to_dict(),
        }