"""FastAPI service for churn prediction."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from src.models.predict import ModelPredictor
from src.utils.config import Config
from src.utils.exceptions import PredictionError


# Pydantic models for API
class CustomerFeatures(BaseModel):
    """Customer features for churn prediction (Telco Churn Schema)."""
    
    customer_id: str = Field(..., description="Unique customer identifier")
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, description="Months as customer")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: int = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse]
    total_customers: int
    average_churn_probability: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    model_path: Optional[str] = None


# Global predictor instance
predictor: Optional[ModelPredictor] = None


def get_predictor() -> ModelPredictor:
    """Get or create model predictor instance."""
    global predictor
    
    if predictor is None:
        config = Config()
        model_path = config.models_path / "churn_model.pkl"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not found at {model_path}. Please train the model first."
            )
        
        try:
            predictor = ModelPredictor(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
    
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info("Starting Churn Prediction API")
    try:
        get_predictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Model not available on startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Churn Prediction API")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Churn Prediction API",
    description="Machine Learning API for predicting customer churn",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)



@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    config = Config()
    model_path = config.models_path / "churn_model.pkl"
    
    model_loaded = False
    if model_path.exists():
        try:
            get_predictor()
            model_loaded = True
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_path=str(model_path) if model_path.exists() else None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.
    """
    try:
        predictor = get_predictor()
        
        # Convert to DataFrame
        customer_data = customer.model_dump()
        # Rename customer_id to customerID to match training schema (though it's dropped)
        customer_data['customerID'] = customer_data.pop('customer_id')
        
        customer_df = pd.DataFrame([customer_data])
        
        # Make prediction
        result = predictor.predict_with_confidence(customer_df)
        
        # Handle numpy types
        prob = float(result["probabilities"][0][1]) if hasattr(result["probabilities"][0][1], "item") else result["probabilities"][0][1]
        pred = int(result["predictions"][0]) if hasattr(result["predictions"][0], "item") else result["predictions"][0]
        conf = float(result["confidence"][0]) if hasattr(result["confidence"][0], "item") else result["confidence"][0]

        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=prob,
            churn_prediction=pred,
            confidence=conf
        )
        
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(customers: List[CustomerFeatures]):
    """
    Predict churn for multiple customers.
    """
    try:
        if len(customers) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size limited to 1000 customers"
            )
        
        predictor = get_predictor()
        
        # Convert to DataFrame
        customers_data = []
        for customer in customers:
            data = customer.model_dump()
            data['customerID'] = data.pop('customer_id')
            customers_data.append(data)
        
        customers_df = pd.DataFrame(customers_data)
        
        # Make predictions
        result = predictor.predict_with_confidence(customers_df)
        
        # Build response
        predictions = []
        total_prob = 0.0
        
        for i, customer in enumerate(customers):
            prob = float(result["probabilities"][i][1])
            pred = int(result["predictions"][i])
            conf = float(result["confidence"][i])
            
            predictions.append(PredictionResponse(
                customer_id=customer.customer_id,
                churn_probability=prob,
                churn_prediction=pred,
                confidence=conf
            ))
            
            total_prob += prob
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(customers),
            average_churn_probability=total_prob / len(customers)
        )
        
    except PredictionError as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected batch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
