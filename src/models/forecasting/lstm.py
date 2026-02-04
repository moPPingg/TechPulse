"""
3. LSTM
Sequence input (samples, seq_len, n_features). PyTorch.
"""

from typing import Optional
import numpy as np
from src.models.forecasting.base import BaseForecaster

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False


def _to_tensor(x: np.ndarray, device) -> "torch.Tensor":
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    return t.to(device)


def _LSTMNet(input_size: int, hidden_size: int, num_layers: int, dropout: float):
    if not _HAS_TORCH:
        raise RuntimeError("torch required")
    import torch.nn as nn
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.linear = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.linear(out).squeeze(-1)
    return _Net()


class LSTMForecaster(BaseForecaster):
    """LSTM for sequence-to-one regression. Input shape (n_samples, seq_len, n_features)."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 50,
        lr: float = 1e-2,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        if not _HAS_TORCH:
            raise ImportError("pip install torch for LSTMForecaster")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.scaler_x = None
        self.scaler_y = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LSTMForecaster":
        from sklearn.preprocessing import StandardScaler
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        n_samples, seq_len, n_feat = X.shape
        self.scaler_x = StandardScaler()
        X_flat = X.reshape(-1, n_feat)
        self.scaler_x.fit(X_flat)
        X = self.scaler_x.transform(X_flat).reshape(n_samples, seq_len, n_feat)
        self.scaler_y = StandardScaler()
        y = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        self.net = _LSTMNet(n_feat, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(_to_tensor(X, self.device), _to_tensor(y, self.device))
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(self.batch_size, len(X)), shuffle=True)
        import torch.nn as _nn
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = _nn.functional.mse_loss(self.net(xb), yb)
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
            out = self.net(_to_tensor(X, self.device))
        out = out.cpu().numpy().ravel()
        return self.scaler_y.inverse_transform(out.reshape(-1, 1)).ravel()
