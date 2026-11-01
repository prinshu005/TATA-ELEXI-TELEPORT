"""
Pytest fixtures providing lightweight synthetic telemetry datasets for unit tests.
Enables rapid, deterministic test execution without dependency on external files.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def synthetic_cmapss_df():
    """
    Creates a small, deterministic synthetic dataset mirroring C-MAPSS schema.
    Contains 3 engines with known run-to-failure cycle lengths:
    - Engine 1: 30 cycles
    - Engine 2: 50 cycles
    - Engine 3: 40 cycles
    """
    rows = []
    engine_specs = [(1, 30), (2, 50), (3, 40)]

    for engine_id, max_cycle in engine_specs:
        for c in range(1, max_cycle + 1):
            degradation = c / float(max_cycle)  # 0 to 1
            row = {
                "engine_id": engine_id,
                "cycle": c,
                "setting_1": 0.001 * (c % 5),
                "setting_2": 0.0002,
                "setting_3": 100.0,
            }
            # Active sensors that trend with degradation
            for s in range(1, 22):
                if s in [1, 5, 10, 16, 18, 19]:
                    row[f"sensor_{s}"] = 100.0  # Constant sensor
                elif s in [2, 3, 4, 7, 8, 11]:
                    # Increasing trend with cycle
                    row[f"sensor_{s}"] = 500.0 + degradation * 50.0 + (c % 2) * 0.1
                else:
                    # Decreasing trend with cycle
                    row[f"sensor_{s}"] = 700.0 - degradation * 40.0 - (c % 3) * 0.1
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_test_and_rul():
    """Synthetic test set with truncated cycles and known true ground truth RUL."""
    rows = []
    # Test Engine 1 observed up to cycle 20 (true RUL = 10)
    # Test Engine 2 observed up to cycle 35 (true RUL = 15)
    test_specs = [(1, 20, 10), (2, 35, 15)]

    for engine_id, max_c, true_rul in test_specs:
        for c in range(1, max_c + 1):
            row = {
                "engine_id": engine_id,
                "cycle": c,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
            }
            for s in range(1, 22):
                row[f"sensor_{s}"] = 500.0 + c * 0.5
            rows.append(row)

    test_df = pd.DataFrame(rows)
    rul_df = pd.DataFrame([{"engine_id": 1, "rul": 10}, {"engine_id": 2, "rul": 15}])
    return test_df, rul_df
