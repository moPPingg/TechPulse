"""
2. ARIMA
Univariate (target series only) or with exog. Uses statsmodels.
"""

from typing import Optional
import numpy as np
from src.models.forecasting.base import BaseForecaster

try:
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_ARIMA = True
except ImportError:
    _HAS_ARIMA = False


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA(p,d,q). Expects tabular X with shape (n, n_features); we use only the target series
    (last column = y history or we fit on y only). For unified API we fit on (X, y) but use only y
    for ARIMA; optional exog = X if provided.
    """

    def __init__(self, order: tuple = (2, 0, 2), **kwargs):
        if not _HAS_ARIMA:
            raise ImportError("pip install statsmodels for ARIMAForecaster")
        self.order = order
        self.kwargs = kwargs
        self.model = None
        self._last_y: Optional[np.ndarray] = None
        self._exog = None

    def fit(self, X: np.ndarray, y: np.ndarray, use_exog: bool = False, **kwargs) -> "ARIMAForecaster":
        y = np.asarray(y).ravel()
        self._last_y = y
        exog = None
        if use_exog:
            X = np.asarray(X)
            if X.ndim == 3:
                X = X.reshape(X.shape[0], -1)
            if X.shape[0] == len(y):
                exog = X
        self._exog = exog
        # statsmodels.tsa.arima.model.ARIMA.fit in recent versions does not accept `disp=0`;
        # we just pass through any extra kwargs from the caller.
        self.model = ARIMA(y, exog=exog, order=self.order, **self.kwargs).fit(**kwargs)
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        steps = len(X)
        exog = None
        if self._exog is not None and X is not None:
            X = np.asarray(X)
            if X.ndim == 3:
                X = X.reshape(X.shape[0], -1)
            exog = X[:steps] if len(X) >= steps else X
        f = self.model.get_forecast(steps=steps, exog=exog)
        pm = f.predicted_mean
        # In recent statsmodels, predicted_mean may already be a numpy array
        if hasattr(pm, "values"):
            return pm.values.ravel()
        return np.asarray(pm).ravel()
