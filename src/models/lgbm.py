import lightgbm as lgb
import numpy as np
from src.models.base import BaseModel

class LightGBMModel(BaseModel):
    """
    LightGBM model tailored for the Green Dragon Trading System.
    Note: As LightGBM is not natively sequential, the windowed time-series data
    must be flattened (samples, window_size * features) before fitting.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Default hyperparameters, can be overridden by kwargs
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'verbose': -1
        }
        self.params.update(kwargs)
        self.num_boost_round = self.params.pop('num_boost_round', 100)
        self.early_stopping_rounds = self.params.pop('early_stopping_rounds', 10)
        
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Flattens 3D time-series window (N, T, F) -> 2D (N, T*F)"""
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        
        X_train_flat = self._flatten(X_train)
        train_data = lgb.Dataset(X_train_flat, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            X_val_flat = self._flatten(X_val)
            valid_data = lgb.Dataset(X_val_flat, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
            
        callbacks = [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds)]
            
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        LightGBM objective 'binary' returns probability of the positive class directly.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
            
        X_flat = self._flatten(X)
        probs = self.model.predict(X_flat)
        return probs
