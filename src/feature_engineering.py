"""
Feature engineering pipeline for turbofan engine time-series sensor data.
Computes multi-window rolling statistics, lag differences, and trends strictly within
individual engine trajectories, preventing engine boundary contamination and future-data leakage.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Constructs informative temporal and statistical features from engine sensor time series.
    Guarantees strict causal ordering (past and current cycles only).
    """

    def __init__(
        self,
        rolling_windows: List[int] = [5, 10, 20],
        include_rolling_mean: bool = True,
        include_rolling_std: bool = True,
        include_diff: bool = True,
        include_trend: bool = True,
        include_settings: bool = True,
        include_cycle: bool = True,
        engine_id_col: str = "engine_id",
        cycle_col: str = "cycle",
    ):
        self.rolling_windows = rolling_windows
        self.include_rolling_mean = include_rolling_mean
        self.include_rolling_std = include_rolling_std
        self.include_diff = include_diff
        self.include_trend = include_trend
        self.include_settings = include_settings
        self.include_cycle = include_cycle
        self.engine_id_col = engine_id_col
        self.cycle_col = cycle_col
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Fit by determining feature names that will be produced."""
        transformed = self.transform(df.head(25))
        # Identify non-identifier, non-label columns
        excluded = [self.engine_id_col, self.cycle_col, "rul", "raw_rul"]
        self.feature_names_ = [c for c in transformed.columns if c not in excluded]
        return self

    def transform(self, df: pd.DataFrame, sensor_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate engineered features for the provided dataset.

        Args:
            df: DataFrame containing engine_id, cycle, settings, and sensor columns.
            sensor_columns: List of active sensor columns. If None, auto-detected.

        Returns:
            DataFrame with all original metadata and newly engineered features.
        """
        df_out = df.copy()

        # Identify target sensor columns to engineer features for
        if sensor_columns is None:
            sensor_columns = [
                c for c in df_out.columns if c.startswith("sensor_")
            ]

        # Group by engine_id once for performance and boundary safety
        grouped = df_out.groupby(self.engine_id_col)

        new_features: Dict[str, pd.Series] = {}

        # 1. First-order differences (cycle-to-cycle change)
        if self.include_diff:
            for s in sensor_columns:
                diff_series = grouped[s].diff(1).fillna(0.0)
                new_features[f"{s}_diff1"] = diff_series

        # 2. Rolling window features (mean, std, short-term deviation trend)
        for w in self.rolling_windows:
            for s in sensor_columns:
                # Rolling mean
                roll_mean = grouped[s].transform(
                    lambda x: x.rolling(w, min_periods=1).mean()
                )
                if self.include_rolling_mean:
                    new_features[f"{s}_roll_mean_{w}"] = roll_mean

                # Rolling standard deviation
                if self.include_rolling_std:
                    roll_std = grouped[s].transform(
                        lambda x: x.rolling(w, min_periods=1).std()
                    ).fillna(0.0)
                    new_features[f"{s}_roll_std_{w}"] = roll_std

                # Trend: difference between current reading and rolling mean
                if self.include_trend and w == self.rolling_windows[0]:
                    trend = df_out[s] - roll_mean
                    new_features[f"{s}_trend_{w}"] = trend

        # Construct new DataFrame with engineered features concatenated
        feat_df = pd.DataFrame(new_features, index=df_out.index)
        result = pd.concat([df_out, feat_df], axis=1)

        # Update feature_names_ if empty
        if not self.feature_names_:
            excluded = [self.engine_id_col, self.cycle_col, "rul", "raw_rul"]
            self.feature_names_ = [c for c in result.columns if c not in excluded]

        return result

    def fit_transform(
        self, df: pd.DataFrame, sensor_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fit and transform in one operation."""
        return self.fit(df).transform(df, sensor_columns=sensor_columns)

    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names."""
        return self.feature_names_


def generate_feature_selection_report(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    rul_col: str = "rul",
) -> Dict[str, Any]:
    """
    Generate an explainability and diagnostic report for engineered features.
    Computes correlation with RUL and variance metrics.
    """
    report: Dict[str, Any] = {
        "total_features": len(feature_cols),
        "feature_correlations": {},
        "top_positive_correlated": [],
        "top_negative_correlated": [],
        "low_variance_features": [],
    }

    if rul_col in train_df.columns:
        corrs = train_df[feature_cols].apply(lambda col: col.corr(train_df[rul_col]))
        corrs = corrs.dropna()
        sorted_corrs = corrs.sort_values()

        report["top_negative_correlated"] = [
            {"feature": k, "correlation": float(v)}
            for k, v in sorted_corrs.head(10).items()
        ]
        report["top_positive_correlated"] = [
            {"feature": k, "correlation": float(v)}
            for k, v in sorted_corrs.tail(10)[::-1].items()
        ]
        report["feature_correlations"] = {k: float(v) for k, v in corrs.items()}

    variances = train_df[feature_cols].var()
    low_vars = variances[variances < 1e-4].index.tolist()
    report["low_variance_features"] = low_vars

    return report
