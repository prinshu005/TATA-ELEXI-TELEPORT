"""
End-to-end execution pipelines for model training, offline evaluation,
and real-time inference with automated work-order generation.
"""

from typing import Dict, Any, Optional, Tuple, List
import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
import joblib
import mlflow

import os
import sys

# Adds the project root directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import CMAPSSDataLoader
from src.data_validation import DataValidator
from src.preprocessing import CMAPSSPreprocessor
from src.feature_engineering import FeatureEngineer, generate_feature_selection_report
from src.rul_labels import (
    add_rul_labels,
    extract_last_cycle_per_engine,
    align_test_predictions_with_ground_truth,
)
from src.anomaly_detection import FleetAnomalyDetector
from src.rul_model import RULPredictor, create_engine_grouped_split
from src.evaluation import evaluate_predictions, format_evaluation_markdown
from src.triage_agent import RuleBasedTriageAgent, TriageDecision
from src.work_order_service import WorkOrderService
from src.database import init_db

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load system configuration safely."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TrainingPipeline:
    """Orchestrates end-to-end training, validation, testing, and artifact serialization."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.artifacts_dir = self.config["mlops"]["artifacts_dir"]
        self.models_dir = self.config["mlops"]["models_dir"]
        self.reports_dir = self.config["mlops"]["reports_dir"]

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Execute complete training pipeline and evaluate on official C-MAPSS benchmark."""
        logger.info(">>> Starting Self-Healing Machines Training Pipeline")
        subset = self.config["dataset"]["name"]

        # 1. Ingestion
        loader = CMAPSSDataLoader()
        train_df, test_df, rul_df = loader.load_subset(subset)

        # 2. Validation
        validator = DataValidator()
        val_report = validator.validate(train_df, strict=True)
        logger.info(f"Training data validation passed: {val_report.summary_stats}")

        # 3. Label Generation (Run-to-failure with Piecewise Capping)
        cap_rul = self.config["rul_labeling"]["cap_rul"]
        rul_cap_val = self.config["rul_labeling"]["rul_cap_value"]
        train_df = add_rul_labels(train_df, cap_rul=cap_rul, rul_cap_value=rul_cap_val)

        # 4. Train-Validation Split (Grouped strictly by engine_id to prevent leakage)
        train_split, val_split = create_engine_grouped_split(
            train_df,
            test_size=self.config["rul_regression"]["validation_ratio"],
            random_state=self.config["rul_regression"]["random_state"],
        )

        # 5. Preprocessing (Learned solely on train_split)
        preprocessor = CMAPSSPreprocessor(
            variance_threshold=self.config["preprocessing"]["drop_constant_variance_threshold"],
            scaling=self.config["preprocessing"]["scaler_type"],
            smoothing_window=self.config["preprocessing"]["smoothing_window"],
        )
        train_prep = preprocessor.fit_transform(train_split)
        val_prep = preprocessor.transform(val_split)
        test_prep = preprocessor.transform(test_df)

        preprocessor.save(os.path.join(self.models_dir, "preprocessor.joblib"))

        # 6. Feature Engineering
        feat_eng = FeatureEngineer(
            rolling_windows=self.config["feature_engineering"]["rolling_windows"],
            include_rolling_mean=self.config["feature_engineering"]["include_rolling_mean"],
            include_rolling_std=self.config["feature_engineering"]["include_rolling_std"],
            include_diff=self.config["feature_engineering"]["include_diff"],
            include_trend=self.config["feature_engineering"]["include_trend"],
        )
        train_feat = feat_eng.fit_transform(train_prep, sensor_columns=preprocessor.retained_sensors_)
        val_feat = feat_eng.transform(val_prep, sensor_columns=preprocessor.retained_sensors_)
        test_feat = feat_eng.transform(test_prep, sensor_columns=preprocessor.retained_sensors_)

        joblib.dump(feat_eng, os.path.join(self.models_dir, "feature_engineer.joblib"))

        feature_cols = feat_eng.get_feature_names()
        logger.info(f"Feature engineering complete: {len(feature_cols)} total features.")

        # Save feature selection & correlation report
        feat_report = generate_feature_selection_report(train_feat, feature_cols)
        with open(os.path.join(self.reports_dir, "feature_report.json"), "w", encoding="utf-8") as f:
            json.dump(feat_report, f, indent=2)

        # 7. Anomaly Detector Training (Isolation Forest on training data)
        detector = FleetAnomalyDetector(
            contamination=self.config["anomaly_detection"]["contamination"],
            n_estimators=self.config["anomaly_detection"]["n_estimators"],
            random_state=self.config["anomaly_detection"]["random_state"],
            warning_threshold=self.config["anomaly_detection"]["score_threshold_warning"],
            critical_threshold=self.config["anomaly_detection"]["score_threshold_critical"],
        )
        detector.fit(train_feat, feature_cols=feature_cols)
        detector.save(os.path.join(self.models_dir, "isolation_forest.joblib"))

        # 8. RUL Model Comparison & Training
        model_type = self.config["rul_regression"]["primary_model"]
        model_params = self.config["rul_regression"]["models"].get(model_type, {})

        rul_predictor = RULPredictor(
            model_type=model_type,
            model_params=model_params,
            random_state=self.config["rul_regression"]["random_state"],
        )
        rul_predictor.fit(train_feat, train_feat["rul"], feature_names=feature_cols)
        rul_predictor.save(os.path.join(self.models_dir, "rul_model.joblib"))

        # Save feature importances
        feat_imp = rul_predictor.get_feature_importances()
        with open(os.path.join(self.reports_dir, "feature_importance.json"), "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in feat_imp.items()}, f, indent=2)

        # 9. Validation Evaluation
        val_preds = rul_predictor.predict(val_feat)
        val_metrics = evaluate_predictions(
            val_feat["rul"].values,
            val_preds,
            engine_ids=val_feat["engine_id"].values,
        )
        logger.info(
            f"Validation Results: MAE={val_metrics['mae']:.2f}, "
            f"RMSE={val_metrics['rmse']:.2f}, R2={val_metrics['r2']:.4f}, "
            f"NASA Score={val_metrics['nasa_score']:.2f}"
        )

        # 10. Official Test Set Evaluation (Extract last cycle per engine)
        test_last_cycle = extract_last_cycle_per_engine(test_feat)
        test_preds = rul_predictor.predict(test_last_cycle)
        test_last_cycle["predicted_rul"] = test_preds

        aligned_test = align_test_predictions_with_ground_truth(
            test_last_cycle, rul_df, pred_col="predicted_rul", gt_col="rul"
        )
        test_metrics = evaluate_predictions(
            aligned_test["actual_rul"].values,
            aligned_test["predicted_rul"].values,
            engine_ids=aligned_test["engine_id"].values,
        )
        logger.info(
            f"Official Test Set Results: MAE={test_metrics['mae']:.2f}, "
            f"RMSE={test_metrics['rmse']:.2f}, R2={test_metrics['r2']:.4f}, "
            f"NASA Score={test_metrics['nasa_score']:.2f}"
        )

        # Save Evaluation Reports
        eval_summary = {
            "dataset_subset": subset,
            "rul_cap_applied": rul_cap_val if cap_rul else None,
            "validation_metrics": val_metrics,
            "official_test_metrics": test_metrics,
        }
        with open(os.path.join(self.reports_dir, "evaluation_report.json"), "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

        test_report_md = format_evaluation_markdown(
            "NASA C-MAPSS FD001 Official Test Benchmark Report",
            test_metrics,
            context="Evaluation on held-out test engines at final observed operating cycle",
        )
        with open(os.path.join(self.reports_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
            f.write(test_report_md)

        # Save aligned test predictions table for quick dashboard access
        aligned_test.to_csv(os.path.join(self.reports_dir, "test_predictions.csv"), index=False)

        # 11. Log Experiment with MLflow
        try:
            mlflow.set_experiment(self.config["mlops"]["experiment_name"])
            with mlflow.start_run(run_name=f"Train_{subset}_{model_type}"):
                mlflow.log_param("dataset", subset)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("rul_cap", rul_cap_val)
                mlflow.log_metrics({
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "val_nasa_score": val_metrics["nasa_score"],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                    "test_nasa_score": test_metrics["nasa_score"],
                })
                mlflow.log_artifact(os.path.join(self.reports_dir, "evaluation_report.json"))
        except Exception as e:
            logger.warning(f"MLflow experiment logging notice: {e}")

        logger.info(">>> Training Pipeline Successfully Completed!")
        return eval_summary


class InferencePipeline:
    """Orchestrates end-to-end inference, anomaly detection, deterministic triage, and work order creation."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.models_dir = self.config["mlops"]["models_dir"]
        self.db_path = self.config["work_orders"]["db_path"]

        init_db(self.db_path)
        self.work_order_service = WorkOrderService(self.db_path)

        # Load fitted components
        self.preprocessor = CMAPSSPreprocessor.load(os.path.join(self.models_dir, "preprocessor.joblib"))
        self.feat_eng = joblib.load(os.path.join(self.models_dir, "feature_engineer.joblib"))
        self.anomaly_detector = FleetAnomalyDetector.load(os.path.join(self.models_dir, "isolation_forest.joblib"))
        self.rul_predictor = RULPredictor.load(os.path.join(self.models_dir, "rul_model.joblib"))

        # Initialize triage agent
        t_cfg = self.config["triage"]
        self.triage_agent = RuleBasedTriageAgent(
            critical_rul_threshold=t_cfg["thresholds"]["critical_rul"],
            high_rul_threshold=t_cfg["thresholds"]["high_rul"],
            medium_rul_threshold=t_cfg["thresholds"]["medium_rul"],
            anomaly_escalation_enabled=t_cfg["anomaly_escalation_enabled"],
            severe_anomaly_threshold=t_cfg["escalation_anomaly_threshold"],
        )

    def process_telemetry_batch(
        self,
        raw_df: pd.DataFrame,
        dispatch_work_orders: bool = True,
        last_cycle_only: bool = True,
    ) -> pd.DataFrame:
        """
        Process incoming fleet telemetry batch, generate health scores, and dispatch work orders.
        """
        # Validate data
        validator = DataValidator()
        val_rep = validator.validate(raw_df)

        if not val_rep.is_valid:
            logger.error(f"Inference input validation error: {val_rep.errors}")

        # Preprocess & engineer features
        prep_df = self.preprocessor.transform(raw_df)
        feat_df = self.feat_eng.transform(prep_df, sensor_columns=self.preprocessor.retained_sensors_)

        # Filter to target evaluation points (typically final cycle for fleet overview)
        if last_cycle_only:
            eval_df = extract_last_cycle_per_engine(feat_df)
        else:
            eval_df = feat_df.copy()

        # Anomaly scoring
        eval_df = self.anomaly_detector.transform(eval_df)

        # RUL prediction
        eval_df["predicted_rul"] = self.rul_predictor.predict(eval_df)

        # Triage and Work Order Dispatch
        decisions: List[TriageDecision] = []
        for _, row in eval_df.iterrows():
            decision = self.triage_agent.evaluate_engine(
                engine_id=int(row["engine_id"]),
                cycle=int(row["cycle"]),
                predicted_rul=float(row["predicted_rul"]),
                anomaly_score=float(row["anomaly_score"]),
                anomaly_status=str(row["anomaly_status"]),
                data_valid=val_rep.is_valid,
                validation_error="; ".join(val_rep.errors) if not val_rep.is_valid else None,
            )
            decisions.append(decision)

            if dispatch_work_orders and decision.should_create_work_order:
                self.work_order_service.create_work_order_from_decision(
                    decision,
                    assignee=self.config["work_orders"]["default_assignee"],
                )

        eval_df["priority"] = [d.priority for d in decisions]
        eval_df["triage_reason"] = [d.reason for d in decisions]
        eval_df["recommended_action"] = [d.recommended_action for d in decisions]

        logger.info(f"Scored {len(eval_df)} engine snapshots. Priorities: {eval_df['priority'].value_counts().to_dict()}")
        return eval_df
