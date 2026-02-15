"""
LSTM-based time-series model for stock trend classification.

Design:
- Sliding window: each sample is (seq_len, n_features) with label = next-day trend
  (aligned with LightGBM: predict return at t+1, then binarize).
- PyTorch Dataset + DataLoader; LSTM classifier (sequence -> logits).
- Training: cross-entropy, validation monitoring, optional early stopping.
- Evaluation: same metrics as LightGBM (accuracy, F1, ROC-AUC, confusion matrix)
  for direct comparison.
- No shuffle across time: train/val/test are split by prediction date.

Upstream: same feature DataFrames as lightgbm_trend (load_price_features, etc.).
Downstream: metrics and optional model checkpoint for inference.
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# Reuse same target/feature conventions as LightGBM pipeline
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


# -----------------------------------------------------------------------------
# 1. Sliding-window dataset construction
# -----------------------------------------------------------------------------


def build_sliding_windows(
    full_df: pd.DataFrame,
    feature_cols: List[str],
    return_col: str = DEFAULT_RETURN_COL,
    seq_len: int = 20,
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
    threshold_up: float = 0.0,
    threshold_down: float = -0.0,
    n_classes: int = 2,
    fill_missing: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y) for train/val/test from one ordered DataFrame using sliding windows.

    - Each sample: X = window of shape (seq_len, n_features), last row = most recent day.
    - Target: next-day return at t+1, converted to trend label (same as LightGBM).
    - Indices: window ending at row i (0-based) predicts return at i+1.
      Train: i in [seq_len-1, n_train-1), val: [n_train-1, n_train+n_val-1), test: [n_train+n_val-1, n_full-1).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (all numpy).
        X_* shapes: (n_samples, seq_len, n_features); y_*: (n_samples,) int labels.
    """
    n_full = len(full_df)
    if n_full < seq_len + 1:
        raise ValueError(f"Need at least seq_len+1 rows (got {n_full}, seq_len={seq_len})")

    M = full_df[feature_cols].copy()
    M = M.fillna(fill_missing)
    M = np.nan_to_num(M.values.astype(np.float64), nan=fill_missing, posinf=0.0, neginf=0.0)
    M = np.clip(M, -1e10, 1e10)

    next_return = full_df[return_col].shift(-1).values.astype(np.float64)
    next_return = np.nan_to_num(next_return, nan=0.0)

    # Build all valid windows: window ending at i predicts return at i+1
    X_list, y_cont_list = [], []
    for i in range(seq_len - 1, n_full - 1):
        X_list.append(M[i - seq_len + 1 : i + 1])  # (seq_len, n_feat)
        y_cont_list.append(next_return[i + 1])
    X_all = np.stack(X_list, axis=0)
    y_cont = np.array(y_cont_list, dtype=np.float64)
    y_all = continuous_to_trend_labels(y_cont, threshold_up, threshold_down, n_classes)

    # Split by prediction date: sample k has window ending at row (seq_len-1+k), predicts return at (seq_len+k).
    # Train: prediction indices [seq_len, n_train) -> k in [0, n_train - seq_len).
    # Val: [n_train, n_train+n_val) -> k in [n_train - seq_len, n_train - seq_len + n_val).
    # Test: [n_train+n_val, n_full) -> k in [n_train - seq_len + n_val, n_full - seq_len).
    train_end = max(0, n_train - seq_len)
    val_end = train_end + n_val
    test_end = n_full - seq_len

    X_train = X_all[:train_end]
    y_train = y_all[:train_end]
    X_val = X_all[train_end:val_end]
    y_val = y_all[train_end:val_end]
    X_test = X_all[val_end:test_end]
    y_test = y_all[val_end:test_end]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_sliding_windows_from_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    return_col: str = DEFAULT_RETURN_COL,
    seq_len: int = 20,
    threshold_up: float = 0.0,
    threshold_down: float = -0.0,
    n_classes: int = 2,
    fill_missing: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window datasets from already split train/val/test DataFrames.

    Concatenates in order (train, val, test) so that val samples can use train history
    and test samples can use train+val history (no future leakage).
    """
    full = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    return build_sliding_windows(
        full,
        feature_cols=feature_cols,
        return_col=return_col,
        seq_len=seq_len,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        threshold_up=threshold_up,
        threshold_down=threshold_down,
        n_classes=n_classes,
        fill_missing=fill_missing,
    )


# -----------------------------------------------------------------------------
# 2. PyTorch Dataset and scaling
# -----------------------------------------------------------------------------


class SlidingWindowDataset(Dataset):
    """
    PyTorch Dataset for (X, y) where X is (seq_len, n_features) and y is int class.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def scale_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    fit_on_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Scale sequence data: fit StandardScaler on flattened train windows, then transform all.
    Preserves shape (n_samples, seq_len, n_features).
    """
    n_samples_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler() if fit_on_train else None
    if fit_on_train and n_samples_train > 0:
        scaler = StandardScaler()
        X_flat = X_train.reshape(-1, n_feat)
        scaler.fit(X_flat)
    if scaler is None:
        return X_train, X_val, X_test, None
    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    if len(X_val) > 0:
        X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    if len(X_test) > 0:
        X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)
    return X_train, X_val, X_test, scaler


# -----------------------------------------------------------------------------
# 3. LSTM classifier model
# -----------------------------------------------------------------------------


class LSTMTrendClassifier(nn.Module):
    """
    LSTM that consumes a sequence (seq_len, n_features) and outputs class logits.
    Uses last hidden state -> linear -> num_classes.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # (batch, hidden_size)
        return self.linear(out)  # (batch, num_classes)


# -----------------------------------------------------------------------------
# 4. Training loop
# -----------------------------------------------------------------------------


def train_lstm_trend(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    *,
    device: Union[str, torch.device] = "cpu",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-2,
    early_stopping_patience: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train LSTM classifier with cross-entropy. Optional validation and early stopping.
    Returns history dict with train_loss, val_loss, val_accuracy per epoch.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required. Install with: pip install torch")
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = SlidingWindowDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
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
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history["train_loss"].append(epoch_loss / len(train_loader))

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item()
                    pred = logits.argmax(dim=1)
                    correct += (pred == yb).sum().item()
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
# 5. Evaluation and prediction
# -----------------------------------------------------------------------------


def predict_lstm(
    model: nn.Module,
    X: np.ndarray,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return class predictions and class probabilities (for ROC-AUC).
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required")
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()
    probs_list = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.asarray(X[start : start + batch_size], dtype=np.float32)).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# -----------------------------------------------------------------------------
# 6. Full pipeline class
# -----------------------------------------------------------------------------


class LSTMTrendPipeline:
    """
    End-to-end LSTM trend classification pipeline with sliding windows.

    Steps:
    1. Load price (and optional news) features.
    2. Time-aware train/val/test split (same as LightGBM).
    3. Build sliding-window sequences and trend labels.
    4. Scale on train only.
    5. Train LSTM with early stopping.
    6. Evaluate on test (same metrics as LightGBM for comparison).
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
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-2,
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")

        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.X_train = self.y_train = self.X_val = self.y_val = self.X_test = self.y_test = None
        self.scaler: Optional[StandardScaler] = None
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
        logger.info("LSTM: loaded %d rows, %d features for %s", len(self.df), len(self.feature_cols), self.symbol)
        return self.df

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-aware train/val/test split (same as LightGBM)."""
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
        """Build sliding-window X, y and scale on train."""
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
            "LSTM: sequences train=%d val=%d test=%d",
            len(self.X_train), len(self.X_val), len(self.X_test),
        )

    def train(self) -> Dict[str, Any]:
        """Build model and run training loop."""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call load_data(), split(), then build_sequences() first")
        _, seq_len, n_feat = self.X_train.shape
        self.model = LSTMTrendClassifier(
            input_size=n_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.n_classes,
            dropout=self.dropout,
        )
        self.history = train_lstm_trend(
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
        """Evaluate on test set (same metrics as LightGBM for comparison)."""
        if self.model is None or self.X_test is None or self.y_test is None:
            raise RuntimeError("Train model first")
        y_pred, y_prob = predict_lstm(
            self.model, self.X_test, device=self.device, batch_size=self.batch_size
        )
        self.metrics = evaluate_trend(
            self.y_test, y_pred, y_prob=y_prob, n_classes=self.n_classes
        )
        logger.info("LSTM test metrics: %s", {k: v for k, v in self.metrics.items() if k != "confusion_matrix"})
        return self.metrics

    def run(self) -> Dict[str, Any]:
        """Full pipeline: load -> split -> sequences -> train -> evaluate."""
        self.load_data()
        train_df, val_df, test_df = self.split()
        self.build_sequences(train_df, val_df, test_df)
        self.train()
        self.evaluate()
        return self.metrics
