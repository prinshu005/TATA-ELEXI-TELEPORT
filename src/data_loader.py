"""
Data loader module for NASA C-MAPSS Turbofan Engine Degradation dataset.
Supports FD001, FD002, FD003, and FD004 subsets with robust validation.
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Standard 26-column schema for NASA C-MAPSS dataset
COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


class CMAPSSDataLoader:
    """Robust loader and parser for NASA C-MAPSS run-to-failure datasets."""

    def __init__(self, data_dir: str = "data/raw", fallback_dir: str = "CMAPSSData"):
        self.data_dir = data_dir
        self.fallback_dir = fallback_dir

    def _resolve_file_path(self, filename: str) -> str:
        """Resolve file path by checking data_dir then fallback_dir."""
        primary = os.path.join(self.data_dir, filename)
        if os.path.exists(primary):
            return primary
        fallback = os.path.join(self.fallback_dir, filename)
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(
            f"Dataset file '{filename}' was not found in '{self.data_dir}' or '{self.fallback_dir}'. "
            f"Please download the C-MAPSS dataset from the NASA Prognostics Center of Excellence "
            f"(https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) "
            f"and place the extracted files in '{self.data_dir}' or '{self.fallback_dir}'."
        )

    def load_subset(
        self, subset: str = "FD001"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training, testing, and true RUL ground truth datasets for the given subset.

        Args:
            subset: C-MAPSS dataset identifier (FD001, FD002, FD003, FD004)

        Returns:
            Tuple of (train_df, test_df, rul_df)
        """
        subset = subset.upper()
        train_path = self._resolve_file_path(f"train_{subset}.txt")
        test_path = self._resolve_file_path(f"test_{subset}.txt")
        rul_path = self._resolve_file_path(f"RUL_{subset}.txt")

        logger.info(f"Loading {subset} from {train_path}, {test_path}, {rul_path}")

        train_df = self._load_data_file(train_path, is_rul=False)
        test_df = self._load_data_file(test_path, is_rul=False)
        rul_df = self._load_data_file(rul_path, is_rul=True)

        # Assign engine_id mapping for RUL ground truth
        test_engines = sorted(test_df["engine_id"].unique())
        rul_df["engine_id"] = test_engines[: len(rul_df)]
        rul_df = rul_df[["engine_id", "rul"]]

        logger.info(
            f"Successfully loaded {subset}: Train shape {train_df.shape}, "
            f"Test shape {test_df.shape}, RUL engines {len(rul_df)}"
        )
        return train_df, test_df, rul_df

    def _load_data_file(self, filepath: str, is_rul: bool = False) -> pd.DataFrame:
        """
        Load a single space-delimited text file safely handling trailing spaces
        and headerless tables.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        if is_rul:
            df = pd.read_csv(filepath, sep=r"\s+", header=None)
            df.columns = ["rul"]
            df["rul"] = pd.to_numeric(df["rul"], errors="raise")
            return df

        df = pd.read_csv(filepath, sep=r"\s+", header=None)
        if df.shape[1] < 26:
            raise ValueError(
                f"Malformed data file {filepath}: Expected 26 columns, found {df.shape[1]}."
            )
        # In case of trailing whitespace creating extra NaN columns, select first 26
        df = df.iloc[:, :26]
        df.columns = COLUMN_NAMES

        # Ensure correct data types
        df["engine_id"] = df["engine_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Strict sorting by engine identifier and operating cycle
        df = df.sort_values(by=["engine_id", "cycle"]).reset_index(drop=True)
        return df


def load_cmapss_data(
    subset: str = "FD001", data_dir: str = "data/raw"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience functional wrapper for loading C-MAPSS subset."""
    loader = CMAPSSDataLoader(data_dir=data_dir)
    return loader.load_subset(subset)
