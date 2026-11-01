"""
Automated model retraining and candidate evaluation workflow.
Ingests newly labeled fleet data, trains candidate models, validates against benchmark sets,
and enforces objective promotion criteria before updating production champion models.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import os
import datetime
import pandas as pd
import numpy as np
import logging
import mlflow

import os
import sys

# Adds the project root directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_validation import DataValidator
from src.preprocessing import CMAPSSPreprocessor
from src.feature_engineering import FeatureEngineer
from src.rul_labels import add_rul_labels
from src.rul_model import RULPredictor, create_engine_grouped_split
from src.evaluation import evaluate_predictions

logger = logging.getLogger(__name__)


@dataclass
class RetrainingResult:
    """Outcome of an automated retraining cycle."""
    success: bool
    promoted: bool
    champion_rmse: float
    candidate_rmse: float
    improvement_pct: float
    message: str
    run_timestamp: str
    candidate_metrics: Dict[str, Any]
    champion_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "promoted": self.promoted,
            "champion_rmse": round(self.champion_rmse, 4),
            "candidate_rmse": round(self.candidate_rmse, 4),
            "improvement_pct": round(self.improvement_pct, 2),
            "message": self.message,
            "run_timestamp": self.run_timestamp,
            "candidate_metrics": self.candidate_metrics,
            "champion_metrics": self.champion_metrics,
        }


class RetrainingManager:
    """
    Manages controlled candidate model training, evaluation, comparison, and promotion.
    """

    def __init__(
        self,
        champion_model_path: str = "artifacts/models/rul_model.joblib",
        preprocessor_path: str = "artifacts/models/preprocessor.joblib",
        min_improvement_pct: float = 2.0,
        max_acceptable_rmse: float = 35.0,
        mlflow_experiment: str = "Self_Healing_Machines_CMAPSS",
    ):
        self.champion_model_path = champion_model_path
        self.preprocessor_path = preprocessor_path
        self.min_improvement_pct = min_improvement_pct
        self.max_acceptable_rmse = max_acceptable_rmse
        self.mlflow_experiment = mlflow_experiment

    def execute_retraining(
        self,
        new_data_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        feature_cols: list,
        candidate_model_type: str = "hist_gradient_boosting",
        candidate_params: Optional[Dict[str, Any]] = None,
    ) -> RetrainingResult:
        """
        Execute retraining pipeline on newly arrived data and compare against champion model.
        """
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Step 1: Validate incoming data
        validator = DataValidator()
        val_report = validator.validate(new_data_df)
        if not val_report.is_valid:
            return RetrainingResult(
                success=False,
                promoted=False,
                champion_rmse=0.0,
                candidate_rmse=0.0,
                improvement_pct=0.0,
                message=f"Retraining rejected: Incoming data validation failed: {val_report.errors}",
                run_timestamp=now_str,
                candidate_metrics={},
            )

        if "rul" not in new_data_df.columns:
            return RetrainingResult(
                success=False,
                promoted=False,
                champion_rmse=0.0,
                candidate_rmse=0.0,
                improvement_pct=0.0,
                message="Retraining halted: Supervised RUL retraining requires ground-truth RUL labels. New data lacks 'rul' column.",
                run_timestamp=now_str,
                candidate_metrics={},
            )

        # Step 2: Evaluate existing Champion model on validation set
        champion_rmse = 999.0
        champion_metrics = None
        if os.path.exists(self.champion_model_path):
            try:
                champion_model = RULPredictor.load(self.champion_model_path)
                champ_preds = champion_model.predict(validation_df)
                champion_metrics = evaluate_predictions(validation_df["rul"].values, champ_preds)
                champion_rmse = champion_metrics["rmse"]
                logger.info(f"Loaded existing Champion model: Validation RMSE = {champion_rmse:.2f}")
            except Exception as e:
                logger.warning(f"Could not load champion model ({e}). Will treat candidate as first model.")

        # Step 3: Train Candidate Model
        candidate_model = RULPredictor(
            model_type=candidate_model_type,
            model_params=candidate_params or {},
            random_state=42,
        )
        candidate_model.fit(new_data_df, new_data_df["rul"], feature_names=feature_cols)

        # Step 4: Evaluate Candidate Model
        cand_preds = candidate_model.predict(validation_df)
        candidate_metrics = evaluate_predictions(validation_df["rul"].values, cand_preds)
        candidate_rmse = candidate_metrics["rmse"]

        # Step 5: Decision Logic
        if champion_rmse >= 900.0:
            # No existing champion model
            improvement_pct = 100.0
            promoted = True
            msg = f"Candidate accepted as initial champion (Validation RMSE: {candidate_rmse:.2f})."
        else:
            improvement = champion_rmse - candidate_rmse
            improvement_pct = (improvement / champion_rmse) * 100.0

            if candidate_rmse <= self.max_acceptable_rmse and improvement_pct >= self.min_improvement_pct:
                promoted = True
                msg = (
                    f"Candidate PROMOTED! RMSE improved from {champion_rmse:.2f} to {candidate_rmse:.2f} "
                    f"({improvement_pct:+.2f}% gain, required >= {self.min_improvement_pct}%)."
                )
            else:
                promoted = False
                msg = (
                    f"Candidate REJECTED. RMSE was {candidate_rmse:.2f} vs Champion {champion_rmse:.2f} "
                    f"({improvement_pct:+.2f}% change, required >= {self.min_improvement_pct}%)."
                )

        # Step 6: Log to MLflow
        try:
            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(run_name=f"Retrain_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("candidate_type", candidate_model_type)
                mlflow.log_param("promoted", promoted)
                mlflow.log_metric("candidate_rmse", candidate_rmse)
                mlflow.log_metric("champion_rmse", champion_rmse)
                mlflow.log_metric("improvement_pct", improvement_pct)
                mlflow.log_metrics({f"cand_{k}": v for k, v in candidate_metrics.items() if isinstance(v, (int, float))})
        except Exception as e:
            logger.warning(f"MLflow logging notice: {e}")

        # Step 7: Promote artifact if approved
        if promoted:
            candidate_model.save(self.champion_model_path)
            logger.info(f"Promoted candidate model saved to {self.champion_model_path}")

        return RetrainingResult(
            success=True,
            promoted=promoted,
            champion_rmse=champion_rmse,
            candidate_rmse=candidate_rmse,
            improvement_pct=improvement_pct,
            message=msg,
            run_timestamp=now_str,
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
        )
