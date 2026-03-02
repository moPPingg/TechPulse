import abc
import numpy as np
from typing import Any, Dict

class BaseModel(abc.ABC):
    """
    Base Model interface for all predictive architectures in the Green Dragon Trading System.
    All subclasses must implement fit() and predict_proba() to align with standard quant pipelines.
    """
    
    def __init__(self, **kwargs):
        self.model = None
        self.hyperparams = kwargs

    @abc.abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Train the model using chronological training splits and optional validation.
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Output calibrated probability (Action Score in [0,1]).
        For binary classification (Directional Prediction 0=Down, 1=Up).
        
        Returns:
            An array or tensor of probabilities for the positive class (Up / Long).
        """
        pass
        
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Outputs deterministic binary prediction based on probability threshold.
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
