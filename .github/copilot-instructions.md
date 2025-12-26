# Churn MLOps Copilot Instructions

## Project Context
This is a production-ready MLOps pipeline for customer churn prediction using Python 3.9+, FastAPI, and Scikit-learn. The project emphasizes modularity, type safety, and robust error handling.

## Architecture & Core Components
- **Service Boundaries**:
  - `src/api`: FastAPI application (`app.py`) handling inference requests.
  - `src/models`: ML logic (`train.py`, `predict.py`) using Scikit-learn and MLflow.
  - `src/data`: Data ingestion (`loader.py`) and validation (`validator.py`).
  - `src/features`: Feature engineering logic (`engineering.py`).
  - `src/utils`: Shared utilities for config, logging, and exceptions.
- **Configuration**: Managed via `src/utils/config.py` using `pydantic-settings`. Always use the `Config` class instead of hardcoded paths.
- **Logging**: Uses `loguru`. Import `logger` from `loguru` or `src/utils/logger.py`. Do not use standard `logging`.

## Coding Conventions
- **Type Hinting**: strict type hints are required for all function arguments and return values. Use `typing` and `pydantic` models.
- **Path Handling**: Always use `pathlib.Path` for file system operations. Never use string concatenation for paths.
- **Error Handling**: Raise specific exceptions from `src/utils/exceptions.py` (e.g., `DataValidationError`, `ModelTrainingError`) rather than generic `Exception`.
- **Docstrings**: Use Google-style docstrings for all classes and functions.

## Critical Workflows
- **Run API**: `python run_api.py` (starts Uvicorn server on port 8000).
- **Train Model**: `python train_sample_model.py` (generates sample data and trains a Random Forest model).
- **Testing**: Run `pytest` from the root. Tests are located in `tests/`.
- **Dependency Management**: `pyproject.toml` is the source of truth.

## Key Patterns & Examples

### Loading Configuration
```python
from src.utils.config import Config
config = Config()
data_path = config.data_processed_path / "file.csv"
```

### Logging
```python
from loguru import logger
logger.info("Processing started", extra={"context": "training"})
```

### API Request Validation
Use Pydantic models defined in `src/api/app.py` for all request/response schemas.
```python
class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100)
    # ...
```

### Data Loading
Always use `DataLoader` to ensure consistent error handling.
```python
from src.data.loader import DataLoader
loader = DataLoader()
df = loader.load_csv("data.csv")
```
