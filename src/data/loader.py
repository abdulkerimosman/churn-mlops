"""Data loading and ingestion functionality."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from src.utils.exceptions import DataValidationError


class DataLoader:
    """Load and manage data from various sources."""
    
    def __init__(self, data_path: Path = Path("data/raw")):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Base path for data files
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load CSV file.
        
        Args:
            filename: Name of CSV file in data directory
            
        Returns:
            DataFrame containing the data
            
        Raises:
            DataValidationError: If file not found or loading fails
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise DataValidationError(f"File not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            raise DataValidationError(f"Failed to load {filename}: {str(e)}")
    
    def load_parquet(self, filename: str) -> pd.DataFrame:
        """
        Load Parquet file.
        
        Args:
            filename: Name of parquet file in data directory
            
        Returns:
            DataFrame containing the data
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise DataValidationError(f"File not found: {file_path}")
        
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            raise DataValidationError(f"Failed to load {filename}: {str(e)}")
    
    def save_csv(self, df: pd.DataFrame, filename: str, output_path: Optional[Path] = None) -> Path:
        """
        Save DataFrame to CSV.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            output_path: Output directory (defaults to processed data path)
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = Path("data/processed")
        
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / filename
        
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {filename} to {file_path}")
        return file_path
    
    def save_parquet(self, df: pd.DataFrame, filename: str, output_path: Optional[Path] = None) -> Path:
        """
        Save DataFrame to Parquet.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            output_path: Output directory (defaults to processed data path)
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = Path("data/processed")
        
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / filename
        
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved {filename} to {file_path}")
        return file_path