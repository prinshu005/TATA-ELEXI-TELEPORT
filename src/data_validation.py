"""
Data validation module for C-MAPSS sensor telemetry.
Ensures data integrity, sequence monotonicity, schema conformance, and absence of non-finite values.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Structured report containing validation outcomes and data diagnostics."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary_stats": self.summary_stats,
        }


class DataValidator:
    """Comprehensive validator for run-to-failure engine sensor data."""

    def __init__(self, expected_columns: Optional[List[str]] = None):
        if expected_columns is None:
            self.expected_columns = [
                "engine_id",
                "cycle",
                "setting_1",
                "setting_2",
                "setting_3",
            ] + [f"sensor_{i}" for i in range(1, 22)]
        else:
            self.expected_columns = expected_columns

    def validate(self, df: pd.DataFrame, strict: bool = False) -> ValidationReport:
        """
        Validate input dataframe for schema, nulls, infinite values, and cycle monotonicity.

        Args:
            df: DataFrame to validate
            strict: If True, raises ValueError upon first error.

        Returns:
            ValidationReport with full diagnostic details.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(df, pd.DataFrame):
            errors.append("Input is not a pandas DataFrame.")
            if strict:
                raise ValueError(errors[-1])
            return ValidationReport(is_valid=False, errors=errors)

        if df.empty:
            errors.append("DataFrame is empty.")
            if strict:
                raise ValueError(errors[-1])
            return ValidationReport(is_valid=False, errors=errors)

        # 1. Schema check
        missing_cols = [c for c in self.expected_columns if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        extra_cols = [c for c in df.columns if c not in self.expected_columns]
        if extra_cols:
            warnings.append(f"Found extra columns not in standard schema: {extra_cols}")

        # 2. Check for missing or non-finite values
        numerical_cols = [c for c in self.expected_columns if c in df.columns]
        null_counts = df[numerical_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            errors.append(f"Columns with null values: {cols_with_nulls.to_dict()}")

        # Check for Infinite / -Infinite values
        inf_counts = np.isinf(df[numerical_cols].select_dtypes(include=[np.number])).sum()
        cols_with_inf = inf_counts[inf_counts > 0]
        if not cols_with_inf.empty:
            errors.append(f"Columns with Infinite values: {cols_with_inf.to_dict()}")

        # 3. Check engine_id and cycle logic
        if "engine_id" in df.columns and "cycle" in df.columns:
            # Check for non-positive cycles
            if (df["cycle"] <= 0).any():
                errors.append("Operating cycles must be strictly positive (> 0).")

            # Check for duplicates
            duplicates = df.duplicated(subset=["engine_id", "cycle"]).sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate (engine_id, cycle) records.")

            # Monotonicity check per engine
            non_monotonic_engines = []
            for engine_id, group in df.groupby("engine_id"):
                cycles = group["cycle"].values
                if not np.all(np.diff(cycles) > 0):
                    non_monotonic_engines.append(int(engine_id))

            if non_monotonic_engines:
                errors.append(
                    f"Cycles are not strictly ascending for engines: {non_monotonic_engines[:10]}"
                )

        # Summary statistics
        stats = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "num_engines": int(df["engine_id"].nunique()) if "engine_id" in df.columns else 0,
            "min_cycle": int(df["cycle"].min()) if "cycle" in df.columns else 0,
            "max_cycle": int(df["cycle"].max()) if "cycle" in df.columns else 0,
        }

        is_valid = len(errors) == 0
        if strict and not is_valid:
            raise ValueError(f"Data validation failed: {'; '.join(errors)}")

        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            summary_stats=stats,
        )


def validate_dataframe(df: pd.DataFrame, strict: bool = False) -> ValidationReport:
    """Helper function to perform validation."""
    validator = DataValidator()
    return validator.validate(df, strict=strict)
