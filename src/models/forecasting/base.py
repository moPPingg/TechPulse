"""
Base interface for all forecasters. Same fit/predict contract for evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseForecaster(ABC):
    """Unified interface: fit(X, y), predict(X) -> 1d array."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseForecaster":
        """Fit on (X_train, y_train). X can be 2d (tabular) or 3d (samples, seq_len, features)."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict for X. Returns 1d array of length len(X)."""
        pass
