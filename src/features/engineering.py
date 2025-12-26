"""Feature engineering and transformation functions."""

from typing import List, Tuple

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.exceptions import FeatureEngineeringError


class FeatureEngineer:
    """Handle feature engineering and transformations."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.scaler = StandardScaler()
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_columns: List[str] = []
        self.is_fitted = False
    
    def identify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (numeric_features, categorical_features)
        """
        self.numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(self.numeric_features)} numeric features")
        logger.info(f"Identified {len(self.categorical_features)} categorical features")
        
        return self.numeric_features, self.categorical_features
    
    def create_time_based_features(self, df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """
        Create time-based features from date columns.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column (if present)
            
        Returns:
            DataFrame with additional time-based features
        """
        df_copy = df.copy()
        
        if date_column and date_column in df_copy.columns:
            try:
                df_copy[date_column] = pd.to_datetime(df_copy[date_column])
                df_copy['days_since_signup'] = (pd.Timestamp.now() - df_copy[date_column]).dt.days
                logger.info(f"Created time-based features from {date_column}")
            except Exception as e:
                raise FeatureEngineeringError(f"Failed to create time features: {str(e)}")
        
        return df_copy
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Apply all preprocessing steps from the notebook.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (to fit scalers/encoders)
            
        Returns:
            Processed DataFrame ready for model
        """
        df_processed = df.copy()
        
        # 1. Drop customerID if exists
        if 'customerID' in df_processed.columns:
            df_processed = df_processed.drop('customerID', axis=1)
            
        # 2. Handle TotalCharges
        # Convert to numeric, coercing errors to NaN
        if 'TotalCharges' in df_processed.columns:
            df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
            # Fill NaNs with 0
            df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(0)
            # Log transformation (log1p)
            df_processed['TotalCharges'] = np.log1p(df_processed['TotalCharges'])
            
        # 3. Convert SeniorCitizen to int (it's already 0/1 but good to ensure)
        if 'SeniorCitizen' in df_processed.columns:
            df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)
            
        # 4. Encoding
        # Binary encoding for binary columns
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df_processed.columns:
                # Simple mapping for binary columns
                df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}).fillna(0).astype(int)

        # One-hot encoding for nominal multi-class columns
        nominal_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaymentMethod']
        
        # Only encode columns that exist
        existing_nominal = [col for col in nominal_cols if col in df_processed.columns]
        
        if existing_nominal:
            df_processed = pd.get_dummies(df_processed, columns=existing_nominal, drop_first=True)
            
        # 5. Scaling
        target_scaling_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        cols_to_scale = [col for col in target_scaling_cols if col in df_processed.columns]
        
        if cols_to_scale:
            if is_training:
                df_processed[cols_to_scale] = self.scaler.fit_transform(df_processed[cols_to_scale])
            else:
                df_processed[cols_to_scale] = self.scaler.transform(df_processed[cols_to_scale])
                
        # 6. Ensure column consistency
        if is_training:
            self.feature_columns = df_processed.columns.tolist()
        elif self.feature_columns:
            # Add missing columns with 0
            for col in self.feature_columns:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            # Drop extra columns
            df_processed = df_processed[self.feature_columns]
            
        return df_processed
