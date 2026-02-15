"""
iTransformer-based time-series model for stock trend classification.

Design:
- Inverted embedding: Unlike standard transformers that treat each time step as a token,
  iTransformer treats each *variable* (feature) as a token. Input (B, seq_len, C) is
  transposed to (B, C, seq_len); each channel's time series is then projected to
  d_model with Linear(seq_len, d_model). So we get C tokens of dimension d_model,
  and the transformer attends over *variables* to capture cross-series dependencies.
- Model: InvertedEmbedding -> Transformer encoder -> mean-pool -> Linear(d_model, num_classes).
- Dataset: Same sliding-window + trend labels as LSTM trend (build_sliding_windows_from_splits).
- Training: Cross-entropy, validation, early stopping.
- Evaluation: Same metrics as LSTM/LightGBM (accuracy, F1, ROC-AUC) for comparison.

Reference: iTransformer (Liu et al.): Inverted Transformers are effective for
multivariate time series forecasting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

# Reuse data and evaluation from LSTM trend and LightGBM
from src.models.lightgbm_trend import (
    DEFAULT_RETURN_COL,
    load_price_features,
    load_news_daily_features,
    merge_multimodal,
    get_feature_columns_for_trend,
    time_aware_split,
    continuous_to_trend_labels,
    evaluate_trend,
)
from src.models.lstm_trend import (
    build_sliding_windows_from_splits,
    scale_sequences,
    SlidingWindowDataset,
)


# -----------------------------------------------------------------------------
# 1. Inverted embedding
# -----------------------------------------------------------------------------


class InvertedEmbedding(nn.Module):
    """
    iTransformer-style embedding: each variable (channel) becomes one token.

    Input: (batch, seq_len, n_channels). We transpose to (batch, n_channels, seq_len)
    so that each channel has a 1d time series of length seq_len. Each series is
    projected to d_model via Linear(seq_len, d_model). Output: (batch, n_channels, d_model).

    Intuition: The transformer then attends over *variables*, not time steps. So it
    learns relationships between features (e.g. volume and return) across the window,
    with the temporal pattern of each variable already encoded in its embedding.
    """

    def __init__(self, seq_len: int, n_channels: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.proj = nn.Linear(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)  # (B, C, T)
        return self.proj(x)   # (B, C, d_model)


# -----------------------------------------------------------------------------
# 2. iTransformer trend classifier
# -----------------------------------------------------------------------------


class iTransformerTrendClassifier(nn.Module):
    """
    iTransformer for trend classification: invert -> transformer over variables -> pool -> head.

    - InvertedEmbedding: (B, seq_len, C) -> (B, C, d_model); C tokens.
    - Transformer encoder: self-attention over the C variable-tokens.
    - Pool: mean over tokens -> (B, d_model).
    - Head: Linear(d_model, num_classes) for logits.
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or (d_model * 4)
        self.embed = InvertedEmbedding(seq_len, n_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.embed(x)           # (B, C, d_model)
        x = self.transformer(x)     # (B, C, d_model)
        x = x.mean(dim=1)          # (B, d_model)
        return self.head(x)         # (B, num_classes)


# -----------------------------------------------------------------------------
# 3. Training loop
# -----------------------------------------------------------------------------


def train_itransformer_trend(
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
) -> Dict[str, Any]:
    """
    Train iTransformer trend classifier with cross-entropy. Optional validation and early stopping.
    Returns history: train_loss, val_loss, val_accuracy.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required")
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        SlidingWindowDataset(X_train, y_train),
        batch_size=min(batch_size, len(X_train)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_loader = DataLoader(
            SlidingWindowDataset(X_val, y_val),
            batch_size=min(batch_size, len(X_val)),
            shuffle=False,
        )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}
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
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item()
                    correct += (logits.argmax(dim=1) == yb).sum().item()
                    total += yb.size(0)
            val_loss /= len(val_loader)
            val_acc = correct / total if total else 0.0
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch + 1, history["train_loss"][-1], val_loss, val_acc,
                )
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
            logger.info("Epoch %d train_loss=%.4f", epoch + 1, history["train_loss"][-1])

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# -----------------------------------------------------------------------------
# 4. Prediction and evaluation
# -----------------------------------------------------------------------------


def predict_itransformer(
    model: nn.Module,
    X: np.ndarray,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return class predictions and class probabilities (for ROC-AUC)."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch required")
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device).eval()
    probs_list = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.asarray(X[start : start + batch_size], dtype=np.float32)).to(device)
            logits = model(batch)
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# -----------------------------------------------------------------------------
# 5. Full pipeline class
# -----------------------------------------------------------------------------


class iTransformerTrendPipeline:
    """
    End-to-end iTransformer trend classification pipeline.

    Same data and splits as LSTM trend: sliding windows + trend labels,
    time-aware train/val/test, scale on train only.
    """

    def __init__(
        self,
        symbol: str,
        features_dir: Union[str, Path],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        include_news: bool = False,
        return_col: str = DEFAULT_RETURN_COL,
        trend_threshold_up: float = 0.0,
        trend_threshold_down: float = -0.0,
        n_classes: int = 2,
        seq_len: int = 20,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        purge_gap: int = 0,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
    ):
        self.symbol = symbol.upper()
        self.features_dir = Path(features_dir)
        self.from_date = from_date
        self.to_date = to_date
        self.include_news = include_news
        self.return_col = return_col
        self.trend_threshold_up = trend_threshold_up
        self.trend_threshold_down = trend_threshold_down
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.purge_gap = purge_gap
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")

        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.X_train = self.y_train = self.X_val = self.y_val = self.X_test = self.y_test = None
        self.scaler: Optional[Any] = None
        self.model: Optional[nn.Module] = None
        self.metrics: Dict[str, Any] = {}
        self.history: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """Load price features and optionally merge news."""
        self.df = load_price_features(
            self.symbol,
            self.features_dir,
            from_date=self.from_date,
            to_date=self.to_date,
        )
        if self.include_news and len(self.df) > 0:
            from_d = self.df["date"].min()
            to_d = self.df["date"].max()
            news_df = load_news_daily_features(self.symbol, from_d, to_d)
            self.df = merge_multimodal(self.df, news_df, news_prefix="news_")
        self.feature_cols = get_feature_columns_for_trend(self.df, exclude=[self.return_col])
        if not self.feature_cols:
            raise ValueError("No feature columns found")
        logger.info(
            "iTransformer: loaded %d rows, %d features for %s",
            len(self.df), len(self.feature_cols), self.symbol,
        )
        return self.df

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-aware train/val/test split (same as LSTM/LightGBM)."""
        if self.df is None:
            self.load_data()
        return time_aware_split(
            self.df,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            purge_gap=self.purge_gap,
        )

    def build_sequences(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Build sliding-window X, y and scale on train (same as LSTM trend)."""
        (self.X_train, self.y_train,
         self.X_val, self.y_val,
         self.X_test, self.y_test) = build_sliding_windows_from_splits(
            train_df, val_df, test_df,
            self.feature_cols,
            return_col=self.return_col,
            seq_len=self.seq_len,
            threshold_up=self.trend_threshold_up,
            threshold_down=self.trend_threshold_down,
            n_classes=self.n_classes,
        )
        self.X_train, self.X_val, self.X_test, self.scaler = scale_sequences(
            self.X_train, self.X_val, self.X_test, fit_on_train=True
        )
        logger.info(
            "iTransformer: sequences train=%d val=%d test=%d",
            len(self.X_train), len(self.X_val), len(self.X_test),
        )

    def train(self) -> Dict[str, Any]:
        """Build model and run training loop."""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call load_data(), split(), then build_sequences() first")
        _, seq_len, n_feat = self.X_train.shape
        self.model = iTransformerTrendClassifier(
            seq_len=seq_len,
            n_channels=n_feat,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            num_classes=self.n_classes,
        )
        self.history = train_itransformer_trend(
            self.model,
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            early_stopping_patience=self.early_stopping_patience,
            verbose=True,
        )
        return self.history

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on test set (same metrics as LSTM/LightGBM for comparison)."""
        if self.model is None or self.X_test is None or self.y_test is None:
            raise RuntimeError("Train model first")
        y_pred, y_prob = predict_itransformer(
            self.model, self.X_test, device=self.device, batch_size=self.batch_size
        )
        self.metrics = evaluate_trend(
            self.y_test, y_pred, y_prob=y_prob, n_classes=self.n_classes
        )
        logger.info(
            "iTransformer test metrics: %s",
            {k: v for k, v in self.metrics.items() if k != "confusion_matrix"},
        )
        return self.metrics

    def run(self) -> Dict[str, Any]:
        """Full pipeline: load -> split -> sequences -> train -> evaluate."""
        self.load_data()
        train_df, val_df, test_df = self.split()
        self.build_sequences(train_df, val_df, test_df)
        self.train()
        self.evaluate()
        return self.metrics
