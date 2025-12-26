"""Data drift detection using Evidently."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.utils.config import Config
from src.utils.logger import logger


class DriftDetector:
    """Detects data drift between reference and current data."""

    def __init__(self, reference_data_path: Optional[Union[str, Path]] = None):
        """
        Initialize drift detector.

        Args:
            reference_data_path: Path to reference dataset (training data).
                               If None, tries to load from config default.
        """
        self.config = Config()
        
        if reference_data_path:
            self.ref_path = Path(reference_data_path)
        else:
            self.ref_path = self.config.data_processed_path / "sample_churn_data.csv"

        self.reference_data: Optional[pd.DataFrame] = None
        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Load reference data from disk."""
        if not self.ref_path.exists():
            logger.warning(f"Reference data not found at {self.ref_path}")
            return

        try:
            self.reference_data = pd.read_csv(self.ref_path)
            logger.info(f"Loaded reference data with {len(self.reference_data)} rows")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")

    def detect_drift(self, current_data: pd.DataFrame, save_report: bool = True) -> Dict:
        """
        Run drift detection against current data.

        Args:
            current_data: New data to check for drift.
            save_report: Whether to save HTML report to disk.

        Returns:
            Dictionary containing drift metrics.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded. Cannot detect drift.")

        # Align columns (Evidently needs matching columns)
        # We only check features that exist in both
        common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
        
        if not common_cols:
            raise ValueError("No common columns between reference and current data")

        # Create and run report
        report = Report(metrics=[DataDriftPreset()])
        
        try:
            report.run(
                reference_data=self.reference_data[common_cols],
                current_data=current_data[common_cols]
            )
        except Exception as e:
            logger.error(f"Evidently report failed: {e}")
            raise

        # Save HTML report
        if save_report:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            report_path = reports_dir / "drift_report.html"
            report.save_html(str(report_path))
            logger.info(f"Drift report saved to {report_path}")

        # Return summary as dict
        return report.as_dict()
