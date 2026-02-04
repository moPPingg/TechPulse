"""
Unified forecasting models: same fit/predict interface for evaluation.
"""

from src.models.forecasting.base import BaseForecaster
from src.models.forecasting.baseline_ml import LinearForecaster, XGBoostForecaster
from src.models.forecasting.arima import ARIMAForecaster
from src.models.forecasting.lstm import LSTMForecaster
from src.models.forecasting.patchtst import PatchTSTForecaster
from src.models.forecasting.transformer import TransformerForecaster

__all__ = [
    "BaseForecaster",
    "LinearForecaster",
    "XGBoostForecaster",
    "ARIMAForecaster",
    "LSTMForecaster",
    "PatchTSTForecaster",
    "TransformerForecaster",
]
