"""
Unit tests for RUL label generation and alignment.
Verifies run-to-failure calculations, piecewise capping, and test alignment.
"""

import pytest
import pandas as pd
import numpy as np
from src.rul_labels import (
    add_rul_labels,
    extract_last_cycle_per_engine,
    align_test_predictions_with_ground_truth,
)


def test_rul_label_calculation(synthetic_cmapss_df):
    """
    Verify exact RUL calculation:
    For Engine 1 (max_cycle = 30):
    Cycle 1 -> RUL = 29
    Cycle 30 -> RUL = 0
    """
    df_rul = add_rul_labels(synthetic_cmapss_df, cap_rul=False)
    e1 = df_rul[df_rul["engine_id"] == 1].sort_values(by="cycle")
    assert e1[e1["cycle"] == 1]["rul"].values[0] == 29
    assert e1[e1["cycle"] == 30]["rul"].values[0] == 0

    e2 = df_rul[df_rul["engine_id"] == 2].sort_values(by="cycle")
    assert e2[e2["cycle"] == 1]["rul"].values[0] == 49
    assert e2[e2["cycle"] == 50]["rul"].values[0] == 0


def test_rul_capping_logic(synthetic_cmapss_df):
    """
    Verify that with RUL cap = 20:
    Any raw RUL > 20 is capped to exactly 20.
    """
    df_capped = add_rul_labels(synthetic_cmapss_df, cap_rul=True, rul_cap_value=20.0)
    assert df_capped["rul"].max() == 20.0

    # Engine 2 at cycle 1 has raw_rul = 49, so capped rul must be 20
    e2_c1 = df_capped[(df_capped["engine_id"] == 2) & (df_capped["cycle"] == 1)]
    assert e2_c1["raw_rul"].values[0] == 49
    assert e2_c1["rul"].values[0] == 20.0

    # Engine 2 at cycle 45 has raw_rul = 5, so capped rul must be 5
    e2_c45 = df_capped[(df_capped["engine_id"] == 2) & (df_capped["cycle"] == 45)]
    assert e2_c45["rul"].values[0] == 5.0


def test_extract_last_cycle(synthetic_test_and_rul):
    """Test extracting the last observed snapshot per engine."""
    test_df, _ = synthetic_test_and_rul
    last_snapshots = extract_last_cycle_per_engine(test_df)
    assert len(last_snapshots) == 2

    e1 = last_snapshots[last_snapshots["engine_id"] == 1]
    assert e1["cycle"].values[0] == 20

    e2 = last_snapshots[last_snapshots["engine_id"] == 2]
    assert e2["cycle"].values[0] == 35


def test_align_test_predictions(synthetic_test_and_rul):
    """Test aligning predictions with test ground truth by engine_id."""
    test_df, rul_df = synthetic_test_and_rul
    last_snapshots = extract_last_cycle_per_engine(test_df)
    last_snapshots["predicted_rul"] = [12.0, 14.0]

    aligned = align_test_predictions_with_ground_truth(last_snapshots, rul_df)
    assert len(aligned) == 2
    assert "error" in aligned.columns

    # Engine 1: actual 10, pred 12 -> error = +2.0
    row1 = aligned[aligned["engine_id"] == 1].iloc[0]
    assert row1["actual_rul"] == 10
    assert row1["predicted_rul"] == 12.0
    assert row1["error"] == 2.0
