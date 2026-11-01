"""
Script: train.py
Executes model training, validation, testing, and MLflow logging.
"""

import sys
import os
import argparse
import logging

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Train Self-Healing Machines ML Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    pipeline = TrainingPipeline(config_path=args.config)
    results = pipeline.run()

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE — NASA C-MAPSS BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Dataset Subset    : {results['dataset_subset']}")
    print(f"RUL Capping       : {results['rul_cap_applied']} cycles")
    print("-" * 70)
    print("VALIDATION PERFORMANCE (Grouped by Engine):")
    val_m = results["validation_metrics"]
    print(f"  MAE             : {val_m['mae']:.2f} cycles")
    print(f"  RMSE            : {val_m['rmse']:.2f} cycles")
    print(f"  R^2 Score       : {val_m['r2']:.4f}")
    print(f"  NASA Score      : {val_m['nasa_score']:.2f}")
    print("-" * 70)
    print("OFFICIAL TEST SET PERFORMANCE (Final Cycle Snapshot vs Ground Truth):")
    test_m = results["official_test_metrics"]
    print(f"  MAE             : {test_m['mae']:.2f} cycles")
    print(f"  RMSE            : {test_m['rmse']:.2f} cycles")
    print(f"  R^2 Score       : {test_m['r2']:.4f}")
    print(f"  NASA Score      : {test_m['nasa_score']:.2f}")
    print(f"  Evaluated Units : {test_m['total_evaluated_engines']} engines")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
