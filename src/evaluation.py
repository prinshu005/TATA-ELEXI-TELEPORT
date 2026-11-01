"""
Model evaluation module for predictive maintenance.
Computes standard regression metrics (MAE, RMSE, R2), NASA asymmetric prognostic score,
residual distributions, and stratified error analysis across lifecycle stages.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def compute_nasa_scoring(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the NASA PHM 2008 asymmetric prognostic score.

    Let d = y_pred - y_true:
    - If d < 0 (early prediction): Penalty is exp(-d / 13) - 1. Conservative; engine serviced early.
    - If d >= 0 (late prediction): Penalty is exp(d / 10) - 1. Hazardous; risks in-flight failure.

    Total Score S = sum(s_i)
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    early_mask = d < 0
    late_mask = d >= 0

    scores = np.zeros_like(d, dtype=float)
    scores[early_mask] = np.exp(-d[early_mask] / 13.0) - 1.0
    scores[late_mask] = np.exp(d[late_mask] / 10.0) - 1.0

    return float(np.sum(scores))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of RUL predictions against ground truth.

    Returns:
        Dictionary with MAE, RMSE, R2, NASA score, residuals, and stratified metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    nasa_score = compute_nasa_scoring(y_true, y_pred)

    residuals = y_pred - y_true
    over_predictions = int(np.sum(residuals > 0))
    under_predictions = int(np.sum(residuals < 0))
    exact_predictions = int(np.sum(residuals == 0))

    # Stratified breakdown by lifecycle stage
    tier_crit = y_true <= 30
    tier_mid = (y_true > 30) & (y_true <= 80)
    tier_high = y_true > 80

    stratified = {
        "critical_tier_rul_le_30": {
            "count": int(np.sum(tier_crit)),
            "mae": float(mean_absolute_error(y_true[tier_crit], y_pred[tier_crit]))
            if np.sum(tier_crit) > 0 else 0.0,
            "rmse": float(np.sqrt(mean_squared_error(y_true[tier_crit], y_pred[tier_crit])))
            if np.sum(tier_crit) > 0 else 0.0,
        },
        "mid_tier_rul_31_to_80": {
            "count": int(np.sum(tier_mid)),
            "mae": float(mean_absolute_error(y_true[tier_mid], y_pred[tier_mid]))
            if np.sum(tier_mid) > 0 else 0.0,
            "rmse": float(np.sqrt(mean_squared_error(y_true[tier_mid], y_pred[tier_mid])))
            if np.sum(tier_mid) > 0 else 0.0,
        },
        "high_tier_rul_gt_80": {
            "count": int(np.sum(tier_high)),
            "mae": float(mean_absolute_error(y_true[tier_high], y_pred[tier_high]))
            if np.sum(tier_high) > 0 else 0.0,
            "rmse": float(np.sqrt(mean_squared_error(y_true[tier_high], y_pred[tier_high])))
            if np.sum(tier_high) > 0 else 0.0,
        },
    }

    results = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nasa_score": nasa_score,
        "total_evaluated_engines": len(y_true),
        "over_prediction_count": over_predictions,
        "under_prediction_count": under_predictions,
        "exact_prediction_count": exact_predictions,
        "residual_stats": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "median": float(np.median(residuals)),
        },
        "stratified_metrics": stratified,
    }

    if engine_ids is not None:
        per_engine = pd.DataFrame({
            "engine_id": engine_ids,
            "actual_rul": y_true,
            "predicted_rul": y_pred,
            "error": residuals,
            "abs_error": np.abs(residuals),
        }).sort_values(by="abs_error", ascending=False)
        results["worst_5_engines"] = per_engine.head(5).to_dict(orient="records")
        results["best_5_engines"] = per_engine.tail(5).to_dict(orient="records")

    return results


def format_evaluation_markdown(
    title: str, metrics: Dict[str, Any], context: str = "Test Set Evaluation"
) -> str:
    """Format evaluation metrics into clean, publication-ready markdown."""
    md = f"""# {title}
**Context:** {context}  
**Total Evaluated Units:** {metrics['total_evaluated_engines']} engines  

## Key Prognostic Metrics
| Metric | Value | Unit | Definition |
| :--- | :--- | :--- | :--- |
| **MAE** | `{metrics['mae']:.2f}` | Cycles | Mean Absolute Error |
| **RMSE** | `{metrics['rmse']:.2f}` | Cycles | Root Mean Squared Error |
| **R^2 Score** | `{metrics['r2']:.4f}` | Dimensionless | Coefficient of Determination |
| **NASA Score** | `{metrics['nasa_score']:.2f}` | Penalty Points | Asymmetric PHM'08 Penalty Function |

## Residual Characteristics
- **Mean Residual (Bias):** `{metrics['residual_stats']['mean']:.2f}` cycles
- **Residual Std Dev:** `{metrics['residual_stats']['std']:.2f}` cycles
- **Over-predictions (Late):** `{metrics['over_prediction_count']}` engines (Penalized heavily)
- **Under-predictions (Early):** `{metrics['under_prediction_count']}` engines (Conservative)

## Stratified Lifecycle Performance
| Operating Regime | Engine Count | MAE (cycles) | RMSE (cycles) |
| :--- | :--- | :--- | :--- |
| **Critical Stage (RUL <= 30)** | {metrics['stratified_metrics']['critical_tier_rul_le_30']['count']} | `{metrics['stratified_metrics']['critical_tier_rul_le_30']['mae']:.2f}` | `{metrics['stratified_metrics']['critical_tier_rul_le_30']['rmse']:.2f}` |
| **Mid-Life Stage (30 < RUL <= 80)** | {metrics['stratified_metrics']['mid_tier_rul_31_to_80']['count']} | `{metrics['stratified_metrics']['mid_tier_rul_31_to_80']['mae']:.2f}` | `{metrics['stratified_metrics']['mid_tier_rul_31_to_80']['rmse']:.2f}` |
| **Healthy Stage (RUL > 80)** | {metrics['stratified_metrics']['high_tier_rul_gt_80']['count']} | `{metrics['stratified_metrics']['high_tier_rul_gt_80']['mae']:.2f}` | `{metrics['stratified_metrics']['high_tier_rul_gt_80']['rmse']:.2f}` |
"""
    return md
