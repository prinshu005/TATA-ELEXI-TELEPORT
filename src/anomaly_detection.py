"""
Anomaly detection module using Isolation Forest.
Detects instantaneous deviations from learned healthy operational baselines,
normalizes anomaly scores, and categorizes observations into NORMAL, WARNING, or ANOMALOUS.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import logging

logger = logging.getLogger(__name__)


class FleetAnomalyDetector:
    """
    Unsupervised Isolation Forest anomaly detector for aircraft engine sensor streams.
    Learns normal operational boundaries and flags out-of-distribution observations.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        warning_threshold: float = 0.55,
        critical_threshold: float = 0.75,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.model: Optional[IsolationForest] = None
        self.feature_columns_: List[str] = []
        self.min_score_: float = 0.0
        self.max_score_: float = 1.0
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "FleetAnomalyDetector":
        """
        Fit Isolation Forest model on nominal operating observations.

        Args:
            X: Training DataFrame.
            feature_cols: Specific features to use for anomaly detection.
        """
        if feature_cols is not None:
            self.feature_columns_ = [c for c in feature_cols if c in X.columns]
        else:
            excluded = ["engine_id", "cycle", "rul", "raw_rul"]
            self.feature_columns_ = [c for c in X.columns if c not in excluded]

        X_train = X[self.feature_columns_].copy()

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train)

        # Compute training raw scores to establish normalization scale
        # Isolation Forest score_samples returns opposite of anomaly score (lower is more anomalous)
        raw_scores = -self.model.score_samples(X_train)
        self.min_score_ = float(np.percentile(raw_scores, 1))
        self.max_score_ = float(np.percentile(raw_scores, 99))
        if self.max_score_ == self.min_score_:
            self.max_score_ = self.min_score_ + 1e-5

        self.is_fitted_ = True
        logger.info(
            f"Fitted IsolationForest on {len(X_train)} samples across {len(self.feature_columns_)} features."
        )
        return self

    def predict_scores(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute continuous raw and normalized anomaly scores [0.0, 1.0].
        0.0 = perfectly nominal baseline; 1.0 = highly anomalous.
        """
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Anomaly detector is not fitted.")

        X_input = X[self.feature_columns_]
        raw_scores = -self.model.score_samples(X_input)

        # Normalize to [0, 1] range based on fitted boundaries
        normalized_scores = (raw_scores - self.min_score_) / (self.max_score_ - self.min_score_)
        normalized_scores = np.clip(normalized_scores, 0.0, 1.0)

        return raw_scores, normalized_scores

    def classify_status(self, normalized_scores: np.ndarray) -> List[str]:
        """Classify normalized anomaly scores into NORMAL, WARNING, or ANOMALOUS."""
        statuses = []
        for s in normalized_scores:
            if s >= self.critical_threshold:
                statuses.append("ANOMALOUS")
            elif s >= self.warning_threshold:
                statuses.append("WARNING")
            else:
                statuses.append("NORMAL")
        return statuses

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame and append anomaly metrics:
        'anomaly_raw_score', 'anomaly_score', and 'anomaly_status'.
        """
        df_out = df.copy()
        raw_scores, norm_scores = self.predict_scores(df_out)
        df_out["anomaly_raw_score"] = raw_scores
        df_out["anomaly_score"] = norm_scores
        df_out["anomaly_status"] = self.classify_status(norm_scores)
        return df_out

    def save(self, filepath: str) -> None:
        """Serialize detector to disk."""
        joblib.dump(self, filepath)
        logger.info(f"Saved anomaly detector to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FleetAnomalyDetector":
        """Load detector from disk."""
        return joblib.load(filepath)
