"""
Script: evaluate.py
Evaluates trained models on the official test set and outputs diagnostic reports.
"""

import sys
import os
import argparse
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import CMAPSSDataLoader
from src.preprocessing import CMAPSSPreprocessor
from src.rul_labels import extract_last_cycle_per_engine, align_test_predictions_with_ground_truth
from src.rul_model import RULPredictor
from src.evaluation import evaluate_predictions, format_evaluation_markdown
from src.pipeline import load_config
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Self-Healing Machines Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    models_dir = config["mlops"]["models_dir"]
    reports_dir = config["mlops"]["reports_dir"]

    # Load artifacts
    preprocessor = CMAPSSPreprocessor.load(os.path.join(models_dir, "preprocessor.joblib"))
    feat_eng = joblib.load(os.path.join(models_dir, "feature_engineer.joblib"))
    model = RULPredictor.load(os.path.join(models_dir, "rul_model.joblib"))

    # Load test data and ground truth
    loader = CMAPSSDataLoader()
    _, test_df, rul_df = loader.load_subset(config["dataset"]["name"])

    # Preprocess & engineer features
    test_prep = preprocessor.transform(test_df)
    test_feat = feat_eng.transform(test_prep, sensor_columns=preprocessor.retained_sensors_)

    # Extract final cycle snapshot per engine
    test_last = extract_last_cycle_per_engine(test_feat)
    preds = model.predict(test_last)
    test_last["predicted_rul"] = preds

    # Align with official ground truth
    aligned = align_test_predictions_with_ground_truth(test_last, rul_df)
    metrics = evaluate_predictions(
        aligned["actual_rul"].values,
        aligned["predicted_rul"].values,
        engine_ids=aligned["engine_id"].values,
    )

    report_md = format_evaluation_markdown(
        f"NASA C-MAPSS {config['dataset']['name']} Official Benchmark Evaluation",
        metrics,
        context="Final cycle per test engine against official NASA ground truth",
    )
    print(report_md)

    out_csv = os.path.join(reports_dir, "evaluation_aligned_predictions.csv")
    aligned.to_csv(out_csv, index=False)
    print(f"Per-engine aligned predictions saved to: {out_csv}")


if __name__ == "__main__":
    main()
