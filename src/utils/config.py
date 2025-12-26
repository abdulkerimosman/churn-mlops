"""Configuration management utilities."""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
from loguru import logger


class Config(BaseSettings):
    """Application configuration settings."""
    
    # Data paths
    data_raw_path: Path = Field(default=Path("data/raw"), description="Raw data directory")
    data_processed_path: Path = Field(default=Path("data/processed"), description="Processed data directory")
    data_interim_path: Path = Field(default=Path("data/interim"), description="Interim data directory")
    
    # Model paths
    models_path: Path = Field(default=Path("models"), description="Models directory")
    
    # Logging
    logs_path: Path = Field(default=Path("logs"), description="Logs directory")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow"  # Allow extra fields from environment
    }


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise


def save_yaml_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {str(e)}")
        raise
