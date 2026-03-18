"""
Script to generate the OOS returns for each model to be used by the bootstrap script.
Saves results to data/oos_returns/{model}_returns.csv
"""

import os, sys, glob, warnings
import pandas as pd
import numpy as np
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
from src.models.lgbm import LightGBMModel
from src.models.lstm import LSTMModel
from src.models.patchtst import PatchTSTModel
from src.models.itransformer import iTransformerModel
from src.backtest.backtest import VectorizedBacktester

# ─── constants ───────────────────────────────────────────────────────────────
TRANSACTION_COST = 0.0025
OPTUNA_THRESHOLD = 0.635
WINDOW_SIZE = 20
EPOCHS_DL = 5
BATCH_SIZE = 256
LR = 5e-4
RETURNS_DIR = os.path.join(ROOT, "data/oos_returns")
os.makedirs(RETURNS_DIR, exist_ok=True)

FEATURE_COLS = ["open", "high", "low", "close", "volume", "ls_binary", "ls_strength"]

def load_and_prepare():
    print("Loading raw CSVs and SMC features …")
    raw_files = glob.glob(os.path.join(ROOT, "data/raw/*.csv"))
    dfs = []
    for f in raw_files:
        df = pd.read_csv(f)
        if "symbol" not in df.columns:
            df["symbol"] = os.path.basename(f).split(".")[0]
        dfs.append(df)
    master = pd.concat(dfs, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])

    smc_path = os.path.join(ROOT, "data/processed/smc_features.csv")
    smc = pd.read_csv(smc_path)
    smc["date"] = pd.to_datetime(smc["date"])

    merged = pd.merge(master, smc, on=["symbol", "date"], how="inner")
    merged.sort_values(["symbol", "date"], inplace=True)
    return merged

def build_dataset(df: pd.DataFrame):
    X_tr_l, y_tr_l = [], []
    X_va_l, y_va_l = [], []
    X_te_l, y_te_l = [], []
    price_te_l = []

    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("date").copy()
        grp["forward_close"] = grp["close"].shift(-1)
        grp["target"] = (grp["forward_close"] > grp["close"]).astype(int)
        grp["close_orig"] = grp["close"]

        scaler = RobustScaler()
        grp["open"]   = grp["open"].pct_change()
        grp["high"]   = grp["high"].pct_change()
        grp["low"]    = grp["low"].pct_change()
        grp["close"]  = grp["close"].pct_change()
        grp["volume"] = np.log1p(grp["volume"]).diff()
        grp.dropna(inplace=True)
        if len(grp) < WINDOW_SIZE + 10:
            continue
        grp[["open","high","low","close","volume"]] = scaler.fit_transform(
            grp[["open","high","low","close","volume"]]
        )

        avail_cols = [c for c in FEATURE_COLS if c in grp.columns]
        X, y, dates = create_windows(grp, avail_cols, "target", WINDOW_SIZE, 1)
        if len(X) == 0:
            continue

        X_tv, y_tv, d_tv, X_te, y_te, d_te = chronological_split(X, y, dates, train_ratio=0.8, purge_gap=1)
        X_tr, y_tr, d_tr, X_va, y_va, d_va = chronological_split(X_tv, y_tv, d_tv, train_ratio=0.8, purge_gap=1)

        X_tr_l.append(X_tr);  y_tr_l.append(y_tr)
        X_va_l.append(X_va);  y_va_l.append(y_va)
        X_te_l.append(X_te);  y_te_l.append(y_te)

        tg = grp[grp["date"].isin(d_te)][["symbol","date","close_orig","target"]]
        price_te_l.append(tg)

    return (
        np.nan_to_num(np.concatenate(X_tr_l)),
        np.concatenate(y_tr_l),
        np.nan_to_num(np.concatenate(X_va_l)),
        np.concatenate(y_va_l),
        np.nan_to_num(np.concatenate(X_te_l)),
        np.concatenate(y_te_l),
        pd.concat(price_te_l),
        len(avail_cols),
    )

def train_dl(model, X, y, epochs=EPOCHS_DL, batch_size=BATCH_SIZE, lr=LR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size, shuffle=True
    )
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()
    return model

def infer_dl(model, X):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()

def export_returns(name, preds, price_df, threshold):
    bt = VectorizedBacktester(transaction_cost=TRANSACTION_COST)
    bt_df = bt.run_backtest(
        df=price_df, action_scores=preds,
        price_col="close_orig", date_col="date",
        threshold_long=threshold,
    )
    
    # Save the 'strat_return_net' as 'daily_return'
    out_df = bt_df[['date', 'strat_return_net']].rename(columns={'strat_return_net': 'daily_return'})
    out_path = os.path.join(RETURNS_DIR, f"{name.lower()}_returns.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Exported {name} returns to {out_path}")

def main():
    df = load_and_prepare()
    print("Building dataset …")
    (X_train, y_train, X_val, y_val, X_test, y_test, test_price_df, num_features) = build_dataset(df)

    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    print("\nTraining models and exporting returns …")

    # 1. LightGBM
    print("Processing LightGBM …")
    lgbm = LightGBMModel(num_boost_round=200)
    lgbm.fit(X_full, y_full, X_test, y_test)
    lgbm_preds = lgbm.predict_proba(X_test)
    export_returns("lightgbm", lgbm_preds, test_price_df, OPTUNA_THRESHOLD)

    # 2. LSTM
    print("Processing LSTM …")
    lstm = train_dl(LSTMModel(input_size=num_features, hidden_size=64, num_layers=2), X_full, y_full)
    lstm_preds = infer_dl(lstm, X_test)
    export_returns("lstm", lstm_preds, test_price_df, OPTUNA_THRESHOLD)

    # 3. PatchTST
    print("Processing PatchTST …")
    try:
        ptst = train_dl(PatchTSTModel(seq_len=WINDOW_SIZE, num_features=num_features), X_full, y_full)
        ptst_preds = infer_dl(ptst, X_test)
    except Exception as e:
        print(f"  PatchTST failed ({e}), skipping export")
        ptst_preds = None
    if ptst_preds is not None:
        export_returns("patchtst", ptst_preds, test_price_df, OPTUNA_THRESHOLD)

    # 4. iTransformer
    print("Processing iTransformer …")
    try:
        itr = train_dl(iTransformerModel(seq_len=WINDOW_SIZE, num_features=num_features), X_full, y_full)
        itr_preds = infer_dl(itr, X_test)
    except Exception as e:
        print(f"  iTransformer failed ({e}), skipping export")
        itr_preds = None
    if itr_preds is not None:
        export_returns("itransformer", itr_preds, test_price_df, OPTUNA_THRESHOLD)

    print("\n✅ All OOS returns exported successfully to data/oos_returns/")

if __name__ == "__main__":
    main()
