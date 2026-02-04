"""
1. Linear Regression / XGBoost (ML baseline)
Tabular: X (n_samples, n_features), y (n_samples,).
"""

from typing import Any, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.models.forecasting.base import BaseForecaster


class LinearForecaster(BaseForecaster):
    """Linear regression. Fast baseline; assumes linear relationship."""

    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LinearForecaster":
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        y = np.asarray(y).ravel()
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.model.fit(Xs, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).ravel()


class XGBoostForecaster(BaseForecaster):
    """XGBoost regressor. Handles non-linearity and feature importance."""

    def __init__(self, **kwargs):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("pip install xgboost for XGBoostForecaster")
        self.model = xgb.XGBRegressor(**kwargs)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "XGBoostForecaster":
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        y = np.asarray(y).ravel()
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.model.fit(Xs, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).ravel()
