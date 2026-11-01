"""
Remaining Useful Life (RUL) label generation module.
Implements run-to-failure ground truth calculation, piecewise linear degradation (capping),
and test sequence alignment with strict leakage prevention.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_rul_labels(
    df: pd.DataFrame,
    cap_rul: bool = True,
    rul_cap_value: float = 125.0,
    engine_id_col: str = "engine_id",
    cycle_col: str = "cycle",
) -> pd.DataFrame:
    """
    Generate ground truth Remaining Useful Life (RUL) labels for run-to-failure sequences.

    For each engine, the final observed cycle is defined as its failure point (RUL = 0).
    For any preceding cycle t:
        RUL(t) = max_cycle_i - cycle(t)

    If cap_rul is True, a piecewise linear degradation model is applied:
        RUL_capped(t) = min(RUL(t), rul_cap_value)

    Justification for RUL Capping:
    In early operational cycles, machinery operates in a healthy steady-state where physical
    sensors exhibit baseline manufacturing variance rather than degradation signatures.
    Uncapped linear degradation forces ML models to distinguish between cycle 300 and cycle 200
    even though the underlying physical health is indistinguishable. Capping focuses model
    capacity on active wear propagation stages (e.g., last 125 cycles).

    Args:
        df: Training DataFrame with engine_id and cycle columns.
        cap_rul: Whether to apply upper threshold capping.
        rul_cap_value: Maximum RUL threshold (typically 120-130 cycles).
        engine_id_col: Column name identifying engines.
        cycle_col: Column name identifying operational cycles.

    Returns:
        DataFrame with added 'rul' (and 'raw_rul' if capped) column(s).
    """
    df = df.copy()

    # Determine failure cycle for each engine
    max_cycles = df.groupby(engine_id_col)[cycle_col].transform("max")
    raw_rul = max_cycles - df[cycle_col]

    if cap_rul:
        df["raw_rul"] = raw_rul
        df["rul"] = np.minimum(raw_rul, float(rul_cap_value))
    else:
        df["rul"] = raw_rul

    return df


def extract_last_cycle_per_engine(
    test_df: pd.DataFrame,
    engine_id_col: str = "engine_id",
    cycle_col: str = "cycle",
) -> pd.DataFrame:
    """
    Extract the snapshot corresponding to the final observed cycle for each engine.
    This corresponds exactly to the time step where test ground truth RUL is evaluated.

    Args:
        test_df: Feature-engineered test DataFrame.
        engine_id_col: Column name identifying engines.
        cycle_col: Column name identifying cycles.

    Returns:
        DataFrame containing one row per engine at its max observed cycle.
    """
    idx_last = test_df.groupby(engine_id_col)[cycle_col].idxmax()
    return test_df.loc[idx_last].sort_values(by=engine_id_col).reset_index(drop=True)


def align_test_predictions_with_ground_truth(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    pred_col: str = "predicted_rul",
    gt_col: str = "rul",
    engine_id_col: str = "engine_id",
) -> pd.DataFrame:
    """
    Align model predictions with official C-MAPSS test RUL ground truth by engine_id.

    Args:
        predictions_df: DataFrame containing at least engine_id and pred_col.
        ground_truth_df: Official RUL DataFrame containing engine_id and gt_col.
        pred_col: Column name of predictions.
        gt_col: Column name of actual ground truth.
        engine_id_col: Column name for engine identifier.

    Returns:
        Merged DataFrame with columns [engine_id, actual_rul, predicted_rul, error].
    """
    merged = pd.merge(
        ground_truth_df[[engine_id_col, gt_col]],
        predictions_df[[engine_id_col, pred_col]],
        on=engine_id_col,
        how="inner",
    )

    if len(merged) != len(ground_truth_df):
        logger.warning(
            f"Test alignment mismatch: Ground truth has {len(ground_truth_df)} engines, "
            f"matched {len(merged)} engines."
        )

    merged = merged.rename(columns={gt_col: "actual_rul"})
    merged["error"] = merged[pred_col] - merged["actual_rul"]
    return merged.sort_values(by=engine_id_col).reset_index(drop=True)
