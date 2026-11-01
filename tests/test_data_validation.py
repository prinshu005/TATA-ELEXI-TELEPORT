"""
Unit tests for data validation module.
Verifies schema compliance, missing-value guards, non-monotonic cycle detection,
and duplicate prevention.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_validation import DataValidator, validate_dataframe

def test_valid_data_passes(synthetic_cmapss_df):
    """Test that conforming C-MAPSS dataset passes validation."""
    validator = DataValidator()
    report = validator.validate(synthetic_cmapss_df)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.summary_stats["num_engines"] == 3
    assert report.summary_stats["num_rows"] == 120


def test_missing_required_column_fails(synthetic_cmapss_df):
    """Test that dropping a required sensor column is flagged as an error."""
    df_missing = synthetic_cmapss_df.drop(columns=["sensor_3"])
    validator = DataValidator()
    report = validator.validate(df_missing)
    assert report.is_valid is False
    assert any("sensor_3" in err for err in report.errors)


def test_null_values_flagged(synthetic_cmapss_df):
    """Test that NaN values in sensor columns trigger validation failure."""
    df_null = synthetic_cmapss_df.copy()
    df_null.loc[5, "sensor_2"] = np.nan
    report = validate_dataframe(df_null)
    assert report.is_valid is False
    assert any("sensor_2" in err for err in report.errors)


def test_infinite_values_flagged(synthetic_cmapss_df):
    """Test that Inf values are detected and rejected."""
    df_inf = synthetic_cmapss_df.copy()
    df_inf.loc[10, "sensor_4"] = np.inf
    report = validate_dataframe(df_inf)
    assert report.is_valid is False
    assert any("sensor_4" in err for err in report.errors)


def test_non_monotonic_cycles_flagged(synthetic_cmapss_df):
    """Test that out-of-order cycles within an engine are flagged."""
    df_bad_cycles = synthetic_cmapss_df.copy()
    # Swap cycle 5 and cycle 6 of engine 1
    idx5 = df_bad_cycles[(df_bad_cycles["engine_id"] == 1) & (df_bad_cycles["cycle"] == 5)].index[0]
    idx6 = df_bad_cycles[(df_bad_cycles["engine_id"] == 1) & (df_bad_cycles["cycle"] == 6)].index[0]
    df_bad_cycles.loc[idx5, "cycle"] = 6
    df_bad_cycles.loc[idx6, "cycle"] = 5

    report = validate_dataframe(df_bad_cycles)
    assert report.is_valid is False
    assert any("Cycles are not strictly ascending" in err for err in report.errors)


def test_duplicate_records_flagged(synthetic_cmapss_df):
    """Test that duplicate (engine_id, cycle) records are caught."""
    duplicate_row = synthetic_cmapss_df.iloc[[0]]
    df_dup = pd.concat([synthetic_cmapss_df, duplicate_row], ignore_index=True)
    report = validate_dataframe(df_dup)
    assert report.is_valid is False
    assert any("duplicate" in err for err in report.errors)
