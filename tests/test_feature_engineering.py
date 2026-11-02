"""
Unit tests for feature engineering pipeline.
Crucially tests prevention of engine-boundary contamination and future data leakage.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Adds the project root directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import FeatureEngineer


def test_engine_boundary_isolation():
  """
  CRITICAL DATA-LEAKAGE TEST:
  Verify that rolling window calculations for Engine 2 NEVER include
  measurements from the end of Engine 1.
  """
  # Create two engines: Engine 1 has sensor=100.0, Engine 2 has sensor=0.0
  data = []
  for c in range(1, 11):
    data.append({"engine_id": 1, "cycle": c, "sensor_2": 100.0})
  for c in range(1, 11):
    data.append({"engine_id": 2, "cycle": c, "sensor_2": 0.0})

  df = pd.DataFrame(data)
  fe = FeatureEngineer(rolling_windows=[5], include_rolling_mean=True)
  transformed = fe.transform(df, sensor_columns=["sensor_2"])

  # At cycle 1 of Engine 2, rolling mean of window 5 must be 0.0, NOT contaminated by Engine 1's 100.0
  e2_c1 = transformed[
      (transformed["engine_id"] == 2) & (transformed["cycle"] == 1)
  ]
  assert e2_c1["sensor_2_roll_mean_5"].values[0] == 0.0, (
      "Engine boundary leakage detected! Expected 0.0, got"
      f" {e2_c1['sensor_2_roll_mean_5'].values[0]}"
  )


def test_no_future_data_leakage():
  """
  Verify that features at cycle t depend only on cycles <= t.
  Modifying future cycles (t+1) must not change the features at cycle t.
  """
  # Create 1 engine with increasing values
  data1 = [
      {"engine_id": 1, "cycle": c, "sensor_2": float(c * 10)}
      for c in range(1, 11)
  ]
  df1 = pd.DataFrame(data1)

  # Modify future cycle 10 in data2
  data2 = [
      {"engine_id": 1, "cycle": c, "sensor_2": float(c * 10)}
      for c in range(1, 10)
  ]
  data2.append(
      {"engine_id": 1, "cycle": 10, "sensor_2": 9999.0}
  )  # drastically change cycle 10
  df2 = pd.DataFrame(data2)

  fe = FeatureEngineer(
      rolling_windows=[5], include_rolling_mean=True, include_diff=True
  )
  res1 = fe.transform(df1, sensor_columns=["sensor_2"])
  res2 = fe.transform(df2, sensor_columns=["sensor_2"])

  # Check cycle 5 features in both datasets: they MUST be identical
  feat_c5_1 = res1[res1["cycle"] == 5]["sensor_2_roll_mean_5"].values[0]
  feat_c5_2 = res2[res2["cycle"] == 5]["sensor_2_roll_mean_5"].values[0]
  assert (
      feat_c5_1 == feat_c5_2
  ), "Future data leakage detected! Modifying cycle 10 altered cycle 5."


def test_feature_engineering_schema(synthetic_cmapss_df):
  """Verify that feature engineer produces valid numeric features without NaNs."""
  fe = FeatureEngineer(
      rolling_windows=[5, 10],
      include_rolling_mean=True,
      include_rolling_std=True,
  )
  result = fe.fit_transform(synthetic_cmapss_df)

  assert not result.empty
  assert len(result) == len(synthetic_cmapss_df)
  # Check no NaNs exist in newly created rolling columns
  roll_cols = [c for c in result.columns if "roll_" in c]
  assert result[roll_cols].isnull().sum().sum() == 0