"""
Retrain LSTM with 7 features (OHLCV + ls_binary + ls_strength)
then save new best_lstm_model.pt

Run from project root:
    python scripts/retrain_lstm_7feat.py
"""
import os, sys, glob, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sklearn.preprocessing import RobustScaler
from src.data.time_series import create_windows, chronological_split
from src.models.lstm import LSTMModel

WINDOW_SIZE  = 20
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
EPOCHS       = 10
BATCH_SIZE   = 256
LR           = 5e-4
THRESHOLD    = 0.635
FEATURE_COLS = ["open", "high", "low", "close", "volume", "ls_binary", "ls_strength"]
MODEL_PATH   = os.path.join(ROOT, "models", "best_lstm_model.pt")
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)


def build_dataset():
    raw_files = glob.glob(os.path.join(ROOT, "data/raw/*.csv"))
    smc = pd.read_csv(os.path.join(ROOT, "data/processed/smc_features.csv"))
    smc["date"] = pd.to_datetime(smc["date"])

    X_tr_l, y_tr_l = [], []
    X_va_l, y_va_l = [], []
    X_te_l, y_te_l = [], []

    for f in raw_files:
        ticker = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = ticker

        # Merge SMC features
        smc_t = smc[smc["symbol"] == ticker] if "symbol" in smc.columns else smc
        df = pd.merge(df, smc_t[["date", "symbol", "ls_binary", "ls_strength"]],
                      on=["date", "symbol"], how="left")
        df["ls_binary"]   = df["ls_binary"].fillna(0)
        df["ls_strength"] = df["ls_strength"].fillna(0)

        df = df.sort_values("date").copy()
        df["forward_close"] = df["close"].shift(-1)
        df["target"] = (df["forward_close"] > df["close"]).astype(int)

        # Return-based scaling
        scaler = RobustScaler()
        df["open"]   = df["open"].pct_change()
        df["high"]   = df["high"].pct_change()
        df["low"]    = df["low"].pct_change()
        df["close"]  = df["close"].pct_change()
        df["volume"] = np.log1p(df["volume"]).diff()
        df.dropna(inplace=True)
        if len(df) < WINDOW_SIZE + 10:
            continue
        df[["open","high","low","close","volume"]] = scaler.fit_transform(
            df[["open","high","low","close","volume"]]
        )

        avail = [c for c in FEATURE_COLS if c in df.columns]
        X, y, dates = create_windows(df, avail, "target", WINDOW_SIZE, 1)
        if len(X) == 0:
            continue

        X_tv, y_tv, d_tv, X_te, y_te, _ = chronological_split(X, y, dates, train_ratio=0.8, purge_gap=1)
        X_tr, y_tr, _,    X_va, y_va, _ = chronological_split(X_tv, y_tv, d_tv, train_ratio=0.8, purge_gap=1)

        X_tr_l.append(X_tr); y_tr_l.append(y_tr)
        X_va_l.append(X_va); y_va_l.append(y_va)
        X_te_l.append(X_te); y_te_l.append(y_te)

    return (
        np.nan_to_num(np.concatenate(X_tr_l)),
        np.concatenate(y_tr_l),
        np.nan_to_num(np.concatenate(X_va_l)),
        np.concatenate(y_va_l),
        np.nan_to_num(np.concatenate(X_te_l)),
        np.concatenate(y_te_l),
        len(avail),
    )


def train(model, X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    opt  = optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCELoss()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={avg:.5f}")
    return model


def eval_sharpe(model, X_val, y_val, threshold):
    from src.backtest.financial_metrics import compute_all_financial_metrics
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    positions = (preds > threshold).astype(float)
    m = compute_all_financial_metrics(positions, y_val)
    return m["sharpe_ratio"]


def main():
    print("Loading and building 7-feature dataset …")
    X_tr, y_tr, X_va, y_va, X_te, y_te, num_features = build_dataset()
    print(f"Features: {num_features} | Train: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")

    # Train on Train+Val for production model
    X_full = np.concatenate([X_tr, X_va])
    y_full = np.concatenate([y_tr, y_va])

    print(f"\nTraining LSTM (input_size={num_features}, hidden={HIDDEN_SIZE}) for {EPOCHS} epochs …")
    lstm = LSTMModel(input_size=num_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    lstm = train(lstm, X_full, y_full)

    # Quick test evaluation
    sharpe = eval_sharpe(lstm, X_te, y_te, THRESHOLD)
    print(f"\nTest Sharpe @ threshold={THRESHOLD}: {sharpe:.4f}")

    # Save model
    torch.save(lstm.state_dict(), MODEL_PATH)
    print(f"\nSaved → {MODEL_PATH}")
    print(f"input_size={num_features} | Update api.py LSTMModel(input_size={num_features})")


if __name__ == "__main__":
    main()
