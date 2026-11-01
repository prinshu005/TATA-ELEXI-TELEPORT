"""
Data preprocessing module.
Handles constant/low-variance sensor detection, engine-boundary-preserving smoothing,
and feature scaling with strict training-only fitting to prevent data leakage.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import logging

logger = logging.getLogger(__name__)


class CMAPSSPreprocessor:
    """
    Stateful preprocessor for C-MAPSS dataset.
    Learns constant columns and scaling statistics solely from training splits.
    """

    def __init__(
        self,
        variance_threshold: float = 1e-4,
        scaling: Optional[str] = "standard",  # 'standard', 'minmax', or None
        smoothing_window: int = 5,
        engine_id_col: str = "engine_id",
        cycle_col: str = "cycle",
    ):
        self.variance_threshold = float(variance_threshold)
        self.scaling = scaling
        self.smoothing_window = smoothing_window
        self.engine_id_col = engine_id_col
        self.cycle_col = cycle_col

        self.dropped_columns_: List[str] = []
        self.retained_sensors_: List[str] = []
        self.scaler_: Optional[Any] = None
        self.feature_columns_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "CMAPSSPreprocessor":
        """
        Fit preprocessor on training data.
        Identifies constant sensor columns and fits the standard/minmax scaler.
        """
        sensor_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("setting_")]

        # 1. Identify constant or near-zero variance columns
        stds = df[sensor_cols].std()
        self.dropped_columns_ = stds[stds <= self.variance_threshold].index.tolist()
        self.retained_sensors_ = [c for c in sensor_cols if c not in self.dropped_columns_]

        logger.info(
            f"Fitted preprocessor: Identified {len(self.dropped_columns_)} low-variance columns "
            f"to drop: {self.dropped_columns_}"
        )
        logger.info(f"Retained {len(self.retained_sensors_)} active sensor/setting columns.")

        # 2. Fit Scaler if configured
        if self.scaling in ["standard", "minmax"]:
            self.scaler_ = StandardScaler() if self.scaling == "standard" else MinMaxScaler()
            self.scaler_.fit(df[self.retained_sensors_])

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, apply_smoothing: bool = False) -> pd.DataFrame:
        """
        Transform dataset by dropping constant features, optionally smoothing, and scaling.
        Never allows cross-engine leakage.
        """
        if not self.is_fitted_:
            raise RuntimeError("CMAPSSPreprocessor must be fitted before transforming data.")

        df_out = df.copy()

        # 1. Drop learned constant columns that exist in the input dataframe
        cols_to_drop = [c for c in self.dropped_columns_ if c in df_out.columns]
        df_out = df_out.drop(columns=cols_to_drop)

        # 2. Optional smoothing per engine (grouped by engine_id to prevent boundary leakage)
        if apply_smoothing and self.smoothing_window > 1:
            active_cols = [c for c in self.retained_sensors_ if c in df_out.columns]
            df_out[active_cols] = (
                df_out.groupby(self.engine_id_col)[active_cols]
                .transform(lambda s: s.rolling(self.smoothing_window, min_periods=1).mean())
            )

        # 3. Apply scaling if fitted
        if self.scaler_ is not None:
            active_cols = [c for c in self.retained_sensors_ if c in df_out.columns]
            df_out[active_cols] = self.scaler_.transform(df_out[active_cols])

        return df_out

    def fit_transform(self, df: pd.DataFrame, apply_smoothing: bool = False) -> pd.DataFrame:
        """Fit on training data and return transformed DataFrame."""
        return self.fit(df).transform(df, apply_smoothing=apply_smoothing)

    def save(self, filepath: str) -> None:
        """Serialize fitted preprocessor to disk."""
        joblib.dump(self, filepath)
        logger.info(f"Saved preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CMAPSSPreprocessor":
        """Load serialized preprocessor from disk."""
        return joblib.load(filepath)
