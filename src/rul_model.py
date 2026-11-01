"""
Remaining Useful Life (RUL) regression models.
Provides baseline linear model, Random Forest, and HistGradientBoosting regressors
with engine-grouped validation splitting and artifact serialization.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance
import joblib
import logging

logger = logging.getLogger(__name__)


def create_engine_grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    engine_id_col: str = "engine_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and validation sets strictly grouped by engine_id.
    Ensures zero engine boundary leakage between training and validation folds.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df[engine_id_col]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_engines = set(train_df[engine_id_col].unique())
    val_engines = set(val_df[engine_id_col].unique())
    assert len(train_engines.intersection(val_engines)) == 0, "Engine overlap detected!"

    logger.info(
        f"Group split created: Train engines={len(train_engines)} ({len(train_df)} rows), "
        f"Val engines={len(val_engines)} ({len(val_df)} rows)"
    )
    return train_df, val_df


class RULPredictor:
    """
    Unified predictor interface supporting Baseline Ridge, Random Forest,
    and HistGradientBoosting regression algorithms.
    """

    def __init__(
        self,
        model_type: str = "hist_gradient_boosting",
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model_params = model_params or {}

        self.model: Optional[Any] = None
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False
        self._init_model()

    def _init_model(self) -> None:
        """Instantiate underlying scikit-learn regressor."""
        if self.model_type in ["baseline", "ridge", "linear"]:
            alpha = self.model_params.get("alpha", 1.0)
            self.model = Ridge(alpha=alpha, random_state=self.random_state)
        elif self.model_type in ["random_forest", "rf"]:
            n_estimators = self.model_params.get("n_estimators", 100)
            max_depth = self.model_params.get("max_depth", 15)
            min_samples_split = self.model_params.get("min_samples_split", 5)
            min_samples_leaf = self.model_params.get("min_samples_leaf", 2)
            n_jobs = self.model_params.get("n_jobs", -1)
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=n_jobs,
            )
        elif self.model_type in ["hist_gradient_boosting", "hgb", "gradient_boosting"]:
            max_iter = self.model_params.get("max_iter", 150)
            max_depth = self.model_params.get("max_depth", 8)
            learning_rate = self.model_params.get("learning_rate", 0.05)
            min_samples_leaf = self.model_params.get("min_samples_leaf", 20)
            self.model = HistGradientBoostingRegressor(
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> "RULPredictor":
        """Fit regressor on training feature matrix and target RUL."""
        if feature_names is not None:
            self.feature_names_ = [f for f in feature_names if f in X.columns]
        else:
            excluded = ["engine_id", "cycle", "rul", "raw_rul"]
            self.feature_names_ = [c for c in X.columns if c not in excluded]

        X_train = X[self.feature_names_].values
        y_train = y.values

        self.model.fit(X_train, y_train)
        self.is_fitted_ = True
        logger.info(
            f"Fitted {self.model_type} regressor on {len(X_train)} samples "
            f"using {len(self.feature_names_)} features."
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate Remaining Useful Life point estimates (cycles).
        Enforces non-negativity constraint (RUL >= 0.0).
        """
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError(f"Model {self.model_type} is not fitted.")

        X_input = X[self.feature_names_].values
        preds = self.model.predict(X_input)
        return np.maximum(preds, 0.0)

    def get_feature_importances(self) -> Dict[str, float]:
        """Extract feature importances if supported by the model architecture."""
        if not self.is_fitted_:
            return {}

        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_names_, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            return dict(zip(self.feature_names_, np.abs(self.model.coef_)))
        return {}

    def save(self, filepath: str) -> None:
        """Serialize model artifact to disk."""
        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "model_params": self.model_params,
                "feature_names": self.feature_names_,
                "random_state": self.random_state,
                "is_fitted": self.is_fitted_,
            },
            filepath,
        )
        logger.info(f"Saved RUL model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "RULPredictor":
        """Load serialized model artifact from disk."""
        payload = joblib.load(filepath)
        predictor = cls(
            model_type=payload["model_type"],
            model_params=payload["model_params"],
            random_state=payload["random_state"],
        )
        predictor.model = payload["model"]
        predictor.feature_names_ = payload["feature_names"]
        predictor.is_fitted_ = payload["is_fitted"]
        return predictor
