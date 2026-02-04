"""
5. Transformer (iTransformer-style: invert dimension so each variable is a token)
Input (samples, seq_len, n_features) -> treat n_features as sequence length for attention.
"""

from typing import Optional
import numpy as np
from src.models.forecasting.base import BaseForecaster

try:
    import torch
    import torch.nn as nn
    class _iTransformer(nn.Module):
        def __init__(self, seq_len: int, n_channels: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.seq_len = seq_len
            self.n_channels = n_channels
            self.input_proj = nn.Linear(seq_len, d_model)
            enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.head = nn.Linear(n_channels * d_model, 1)
        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.input_proj(x)
            x = self.transformer(x)
            x = x.reshape(x.size(0), -1)
            return self.head(x).squeeze(-1)
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    _iTransformer = None


class TransformerForecaster(BaseForecaster):
    """Transformer (iTransformer-style). Input (n_samples, seq_len, n_features)."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        if not _HAS_TORCH:
            raise ImportError("pip install torch for TransformerForecaster")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.scaler_x = None
        self.scaler_y = None
        self._seq_len = None
        self._n_channels = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TransformerForecaster":
        from sklearn.preprocessing import StandardScaler
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        n_samples, seq_len, n_feat = X.shape
        self._seq_len = seq_len
        self._n_channels = n_feat
        self.scaler_x = StandardScaler()
        X_flat = X.reshape(-1, n_feat)
        self.scaler_x.fit(X_flat)
        X = self.scaler_x.transform(X_flat).reshape(n_samples, seq_len, n_feat)
        self.scaler_y = StandardScaler()
        y = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        self.net = _iTransformer(seq_len, n_feat, self.d_model, self.n_heads, self.n_layers).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y).float(),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(self.batch_size, len(X)), shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        n_samples, seq_len, n_feat = X.shape
        X_flat = X.reshape(-1, n_feat)
        X = self.scaler_x.transform(X_flat).reshape(n_samples, seq_len, n_feat)
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(X).float().to(self.device))
        out = out.cpu().numpy().ravel()
        return self.scaler_y.inverse_transform(out.reshape(-1, 1)).ravel()
