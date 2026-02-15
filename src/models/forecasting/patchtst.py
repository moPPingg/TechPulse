"""
PatchTST: Patch-based Transformer for stock time-series prediction.

Architecture (intuitive):
  1. Patching: Split the input series (seq_len, n_channels) into non-overlapping
     patches of length patch_len. Each patch is a local snippet of time (e.g. 4 days).
     This reduces sequence length from seq_len to n_patches = seq_len // patch_len,
     so the transformer sees "patch tokens" instead of raw time steps.
  2. Patch embedding: Each patch (patch_len * n_channels values) is projected via
     a linear layer into a d_model-dimensional vector. Optionally we add positional
     embeddings so the model knows patch order.
  3. Transformer encoder: Standard stack of self-attention + FFN layers. The model
     attends over patch tokens to capture both local (within-patch) and global
     (across-patch) patterns.
  4. Head: We aggregate patch representations (mean-pool or last token) and project
     to a scalar for next-step return prediction (regression).

Dataset format: (n_samples, seq_len, n_features) — same as LSTM/Transformer.
Use src.evaluation.data.prepare_sequential() to build from train/val/test DataFrames.

Evaluation: fit/predict interface; use regression metrics (RMSE, MAE, R2) on test set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from src.models.forecasting.base import BaseForecaster

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Patch embedding
# -----------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """
    Turn a time series into a sequence of patch tokens.

    Input: (batch, seq_len, n_channels).
    We use non-overlapping patches of length patch_len. Effective sequence length
    becomes n_patches = seq_len // patch_len (we trim if needed).
    Each patch is flattened to (patch_len * n_channels) and projected to d_model.

    Intuition: Instead of feeding 60 time steps to the transformer, we feed
    e.g. 15 patches of 4 steps each. This cuts cost and often improves performance
    by letting the linear layer summarize local context per patch.
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        patch_len: int,
        d_model: int,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.n_channels = n_channels
        # Non-overlapping: n_patches = seq_len // patch_len; trim seq to n_patches * patch_len
        self.n_patches = seq_len // patch_len
        self.effective_seq_len = self.n_patches * patch_len
        self.patch_dim = patch_len * n_channels
        self.proj = nn.Linear(self.patch_dim, d_model)
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
            nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Use the most recent effective_seq_len steps (trim if needed).
        B, T, C = x.shape
        T_use = min(T, self.effective_seq_len)
        x = x[:, -T_use:, :]
        # Reshape into patches: (B, n_patches, patch_len, C) -> (B, n_patches, patch_len*C)
        x = x.reshape(B, self.n_patches, self.patch_len, C).reshape(B, self.n_patches, self.patch_dim)
        x = self.proj(x)  # (B, n_patches, d_model)
        if self.use_pos_embed:
            x = x + self.pos_embed
        return x


# -----------------------------------------------------------------------------
# 2. Full PatchTST model (embedding + transformer + head)
# -----------------------------------------------------------------------------


class PatchTST(nn.Module):
    """
    PatchTST: Patch embedding -> Transformer encoder -> regression head.

    - Patch embedding reduces (B, seq_len, n_channels) to (B, n_patches, d_model).
    - Transformer encoder applies self-attention over patch tokens.
    - Pooling: mean over patch dimension (default) or use last token.
    - Head: linear(d_model, 1) for scalar prediction.
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        patch_len: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        pool_mode: str = "mean",  # "mean" or "last"
        use_pos_embed: bool = True,
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or (d_model * 4)
        self.n_patches = seq_len // patch_len
        self.pool_mode = pool_mode
        self.patch_embed = PatchEmbedding(
            seq_len, n_channels, patch_len, d_model, use_pos_embed=use_pos_embed
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.patch_embed(x)  # (B, n_patches, d_model)
        x = self.transformer(x)  # (B, n_patches, d_model)
        if self.pool_mode == "mean":
            x = x.mean(dim=1)  # (B, d_model)
        else:
            x = x[:, -1, :]  # last token
        return self.head(x).squeeze(-1)  # (B,)


# -----------------------------------------------------------------------------
# 3. Training pipeline
# -----------------------------------------------------------------------------


def train_patchtst(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    *,
    device: Union[str, torch.device] = "cpu",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train PatchTST with MSE loss. Optional validation and early stopping.
    Returns history: train_loss, val_loss (if val provided).
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required")
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_train, dtype=np.float32)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.asarray(X_val, dtype=np.float32)),
                torch.from_numpy(np.asarray(y_val, dtype=np.float32)),
            ),
            batch_size=min(batch_size, len(X_val)),
            shuffle=False,
        )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history["train_loss"].append(epoch_loss / len(train_loader))

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item()
            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)
            if verbose and (epoch + 1) % 5 == 0:
                logger.info("Epoch %d train_loss=%.6f val_loss=%.6f", epoch + 1, history["train_loss"][-1], val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                break
        elif verbose and (epoch + 1) % 10 == 0:
            logger.info("Epoch %d train_loss=%.6f", epoch + 1, history["train_loss"][-1])

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_patchtst(
    model: nn.Module,
    X: np.ndarray,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Run model in eval mode and return 1d predictions."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch required")
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device).eval()
    out_list = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.asarray(X[start : start + batch_size], dtype=np.float32)).to(device)
            out_list.append(model(batch).cpu().numpy())
    return np.concatenate(out_list, axis=0).ravel()


# -----------------------------------------------------------------------------
# 4. Dataset formatting (helpers)
# -----------------------------------------------------------------------------


def format_sequential_for_patchtst(
    X: np.ndarray,
    seq_len: Optional[int] = None,
    patch_len: Optional[int] = None,
) -> np.ndarray:
    """
    Ensure X has shape (n_samples, seq_len, n_features) and optionally trim
    seq_len so it is divisible by patch_len (required for non-overlapping patches).

    If X is 2d (n_samples, features), we do not add sequence dimension; caller
    must provide 3d input for PatchTST. If seq_len is set and X has more steps,
    we take the last seq_len steps. If patch_len is set, we trim seq_len to
    (seq_len // patch_len) * patch_len.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X  # caller must expand to 3d
    if X.ndim != 3:
        raise ValueError("X must be (n_samples, seq_len, n_features)")
    n, T, C = X.shape
    if seq_len is not None and T > seq_len:
        X = X[:, -seq_len:, :]
        T = seq_len
    if patch_len is not None and T % patch_len != 0:
        T_trim = (T // patch_len) * patch_len
        if T_trim > 0:
            X = X[:, -T_trim:, :]
    return X


# -----------------------------------------------------------------------------
# 5. Evaluation framework
# -----------------------------------------------------------------------------


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Regression metrics for time-series forecast (e.g. next-step return).
    Returns dict with rmse, mae, mape (avoid div-by-zero), r2.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    # MAPE: mean(|pred - true| / |true|), skip near-zero true
    mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs(diff[mask]) / np.abs(y_true[mask]))) if mask.any() else 0.0
    # R2
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# -----------------------------------------------------------------------------
# 6. BaseForecaster-compatible class (fit/predict)
# -----------------------------------------------------------------------------


class PatchTSTForecaster(BaseForecaster):
    """
    PatchTST for next-step return prediction. Input shape (n_samples, seq_len, n_features).

    Dataset: Use src.evaluation.data.prepare_sequential(train_df, val_df, test_df, seq_len)
    to get X_train, y_train, X_val, y_val, X_test, y_test. Then fit(X_train, y_train)
    and predict(X_test). Optionally pass X_val, y_val in fit via kwargs for early stopping.
    """

    def __init__(
        self,
        patch_len: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        pool_mode: str = "mean",
        use_pos_embed: bool = True,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
    ):
        if not _HAS_TORCH:
            raise ImportError("pip install torch for PatchTSTForecaster")
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.pool_mode = pool_mode
        self.use_pos_embed = use_pos_embed
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.scaler_x = None
        self.scaler_y = None
        self._seq_len = None
        self._n_channels = None
        self._history: Dict[str, Any] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "PatchTSTForecaster":
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        n_samples, seq_len, n_feat = X.shape
        # Trim seq_len to multiple of patch_len
        seq_len = (seq_len // self.patch_len) * self.patch_len
        if seq_len <= 0:
            raise ValueError("seq_len must be >= patch_len")
        X = X[:, -seq_len:, :]

        self._seq_len = seq_len
        self._n_channels = n_feat
        self.scaler_x = StandardScaler()
        X_flat = X.reshape(-1, n_feat)
        self.scaler_x.fit(X_flat)
        X = self.scaler_x.transform(X_flat).reshape(n_samples, seq_len, n_feat)
        self.scaler_y = StandardScaler()
        y = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        self.net = PatchTST(
            seq_len,
            n_feat,
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            pool_mode=self.pool_mode,
            use_pos_embed=self.use_pos_embed,
        ).to(self.device)

        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val = np.asarray(X_val, dtype=np.float32)
            if X_val.ndim == 2:
                X_val = X_val[:, :, np.newaxis]
            X_val = X_val[:, -seq_len:, :]
            X_val_flat = X_val.reshape(-1, n_feat)
            X_val = self.scaler_x.transform(X_val_flat).reshape(X_val.shape[0], seq_len, n_feat)
            y_val = np.asarray(y_val, dtype=np.float32).ravel()
            y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()

        self._history = train_patchtst(
            self.net,
            X, y,
            X_val=X_val if (X_val is not None and len(X_val) > 0) else None,
            y_val=y_val if (y_val is not None and len(y_val) > 0) else None,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            early_stopping_patience=self.early_stopping_patience,
            verbose=True,
        )
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        seq_len = self._seq_len
        n_feat = self._n_channels
        X = X[:, -seq_len:, :]
        X_flat = X.reshape(-1, n_feat)
        X = self.scaler_x.transform(X_flat).reshape(X.shape[0], seq_len, n_feat)
        self.net.eval()
        pred = predict_patchtst(self.net, X, device=self.device, batch_size=self.batch_size)
        return self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
