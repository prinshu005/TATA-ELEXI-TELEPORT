"""
MLOps monitoring and data quality tracking module.
Monitors prediction volumes, RUL distributions, anomaly prevalence, and detects
sensor distribution drift (distinguishing genuine physical degradation from covariate shift).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)


def compute_population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between a baseline and current distribution.
    PSI < 0.10: No significant shift.
    0.10 <= PSI < 0.25: Moderate shift; investigate.
    PSI >= 0.25: Significant shift; potential data drift.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Quantile binning on reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = exp_counts / len(expected) + epsilon
    act_pct = act_counts / len(actual) + epsilon

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


class FleetMonitor:
    """
    Continuous operational telemetry and model performance monitor.
    Tracks prediction logs, operational health distributions, and data integrity.
    """

    def __init__(self, baseline_df: Optional[pd.DataFrame] = None):
        self.baseline_df = baseline_df

    def analyze_fleet_telemetry(
        self,
        scored_df: pd.DataFrame,
        sensor_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze current fleet scoring snapshot for operations and drift.
        """
        total_engines = len(scored_df)
        avg_rul = float(scored_df["predicted_rul"].mean()) if "predicted_rul" in scored_df.columns else 0.0
        min_rul = float(scored_df["predicted_rul"].min()) if "predicted_rul" in scored_df.columns else 0.0

        # Anomaly counts
        anomaly_count = 0
        warning_count = 0
        if "anomaly_status" in scored_df.columns:
            anomaly_count = int((scored_df["anomaly_status"] == "ANOMALOUS").sum())
            warning_count = int((scored_df["anomaly_status"] == "WARNING").sum())

        # Priority breakdown
        priority_dist = {}
        if "priority" in scored_df.columns:
            priority_dist = scored_df["priority"].value_counts().to_dict()

        # Data quality checks
        null_cells = int(scored_df.isnull().sum().sum())
        inf_cells = int(np.isinf(scored_df.select_dtypes(include=[np.number])).sum().sum())

        # Covariate Shift / Drift Analysis
        drift_report = {}
        if self.baseline_df is not None and sensor_cols is not None:
            for s in sensor_cols:
                if s in self.baseline_df.columns and s in scored_df.columns:
                    base_vals = self.baseline_df[s].dropna().values
                    curr_vals = scored_df[s].dropna().values
                    psi_val = compute_population_stability_index(base_vals, curr_vals)
                    ks_stat, ks_pval = ks_2samp(base_vals, curr_vals)
                    drift_report[s] = {
                        "psi": round(psi_val, 4),
                        "ks_statistic": round(float(ks_stat), 4),
                        "p_value": round(float(ks_pval), 4),
                        "drift_detected": bool(psi_val >= 0.25 or ks_pval < 0.01),
                    }

        return {
            "total_engines_evaluated": total_engines,
            "average_predicted_rul": round(avg_rul, 2),
            "minimum_predicted_rul": round(min_rul, 2),
            "critical_engines": priority_dist.get("CRITICAL", 0),
            "high_engines": priority_dist.get("HIGH", 0),
            "medium_engines": priority_dist.get("MEDIUM", 0),
            "low_engines": priority_dist.get("LOW", 0),
            "anomalous_observations": anomaly_count,
            "warning_observations": warning_count,
            "data_quality": {
                "missing_values": null_cells,
                "infinite_values": inf_cells,
                "is_healthy": (null_cells == 0 and inf_cells == 0),
            },
            "sensor_drift_analysis": drift_report,
            "notes": (
                "Diagnostic Note: Individual engine sensor drift across cycles represents physical mechanical degradation. "
                "Covariate fleet-wide drift reflects shifts in flight operating conditions or sensor calibrations."
            ),
        }
