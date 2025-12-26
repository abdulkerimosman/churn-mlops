"""Batch inference for churn model."""
from pathlib import Path
import argparse
import pandas as pd
from loguru import logger
from src.models.predict import ModelPredictor
from src.data.loader import DataLoader
from src.utils.logger import setup_logging

def run_inference(model_path: Path, data_path: Path, output_path: Path) -> Path:
    loader = DataLoader(data_path.parent)
    df = loader.load_csv(data_path.name)
    predictor = ModelPredictor(model_path)
    preds = predictor.predict_proba(df)
    result = df.copy()
    result["churn_proba"] = preds[:, 1]
    result["churn_pred"] = (preds[:, 1] >= 0.5).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Wrote predictions to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Run churn inference")
    parser.add_argument("--model-path", type=Path, default=Path("models/churn_model.pkl"))
    parser.add_argument("--data-path", type=Path, required=True, help="CSV to score")
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/predictions.csv"))
    args = parser.parse_args()
    setup_logging()
    run_inference(args.model_path, args.data_path, args.output_path)

if __name__ == "__main__":
    main()