"""
Script: run_pipeline.py
Runs the complete inference and automated maintenance work-order generation pipeline
against the test fleet.
"""

import sys
import os
import argparse
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import CMAPSSDataLoader
from src.pipeline import InferencePipeline, load_config
from src.work_order_service import WorkOrderService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run Fleet Predictive Maintenance Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--export-csv", type=str, default="artifacts/reports/dispatched_work_orders.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    loader = CMAPSSDataLoader()
    _, test_df, _ = loader.load_subset(config["dataset"]["name"])

    print(">>> Initializing Inference Pipeline...")
    pipeline = InferencePipeline(config_path=args.config)

    print(f">>> Processing {test_df['engine_id'].nunique()} test engines...")
    scored_fleet = pipeline.process_telemetry_batch(
        test_df,
        dispatch_work_orders=True,
        last_cycle_only=True,
    )

    wo_service = WorkOrderService(config["work_orders"]["db_path"])
    stats = wo_service.get_summary_statistics()

    print("\n" + "=" * 65)
    print(" FLEET PREDICTIVE MAINTENANCE PIPELINE SUMMARY")
    print("=" * 65)
    print(f"Total Engines Evaluated : {len(scored_fleet)}")
    print(f"Priority Distribution   : {scored_fleet['priority'].value_counts().to_dict()}")
    print(f"Anomaly Distribution    : {scored_fleet['anomaly_status'].value_counts().to_dict()}")
    print("-" * 65)
    print(f"Total Work Orders in DB : {stats['total_work_orders']}")
    print(f"Active Open Work Orders : {stats['open_work_orders']}")
    print(f"Critical Alerts         : {stats['critical_active_alerts']}")
    print(f"High Priority Alerts    : {stats['high_active_alerts']}")
    print("=" * 65)

    if args.export_csv:
        csv_path = wo_service.export_to_csv(args.export_csv)
        print(f"\nWork orders exported to: {csv_path}")


if __name__ == "__main__":
    main()
