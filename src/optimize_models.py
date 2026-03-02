import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna

# Ensure reproducible stochastic behavior for DL models
torch.manual_seed(42)
np.random.seed(42)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.time_series import create_windows, chronological_split
from src.models.lgbm import LightGBMModel
from src.models.lstm import LSTMModel
from src.models.patchtst import PatchTSTModel
from src.models.itransformer import iTransformerModel
from src.backtest.backtest import VectorizedBacktester

# Supress optuna terminal clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_merge_data() -> pd.DataFrame:
    raw_files = glob.glob('d:/TechPulse/data/raw/*.csv')
    dfs = []
    for f in raw_files:
        df = pd.read_csv(f)
        if 'symbol' not in df.columns:
            df['symbol'] = os.path.basename(f).split('.')[0]
        dfs.append(df)
    master_df = pd.concat(dfs, ignore_index=True)
    master_df['date'] = pd.to_datetime(master_df['date'])
    
    smc_file = 'd:/TechPulse/data/processed/smc_features.csv'
    smc_df = pd.read_csv(smc_file)
    smc_df['date'] = pd.to_datetime(smc_df['date'])
    
    merged_df = pd.merge(master_df, smc_df, on=['symbol', 'date'], how='inner')
    merged_df.sort_values(['symbol', 'date'], inplace=True)
    return merged_df

def prepare_dataset_optuna(df: pd.DataFrame, window_size=20, horizon=1):
    X_train_list, y_train_list, d_train_list = [], [], []
    X_val_list, y_val_list, d_val_list = [], [], []
    X_test_list, y_test_list, d_test_list = [], [], []
    price_val_list, price_test_list = [], []
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'ls_binary', 'ls_strength']
    
    print("Preparing train, validation, and test sliding windows per symbol...")
    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('date').copy()
        group['forward_close'] = group['close'].shift(-horizon)
        group['target'] = (group['forward_close'] > group['close']).astype(int)
        group['close_orig'] = group['close']
        
        # Apply robust standardization to manage outliers while preserving relative distribution
        # Prices are still percentage returns, volume is diff log
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        group['open'] = group['open'].pct_change()
        group['high'] = group['high'].pct_change()
        group['low'] = group['low'].pct_change()
        group['close_ret'] = group['close'].pct_change()
        group['close'] = group['close_ret']
        group['volume'] = np.log1p(group['volume']).diff()
        
        group = group.dropna()
        if len(group) > 0:
             group[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(group[['open', 'high', 'low', 'close', 'volume']])
        
        X, y, dates = create_windows(group, feature_cols, 'target', window_size, horizon)
        if len(X) == 0: continue
        
        # 80/20 Test Split
        X_tr_v, y_tr_v, d_tr_v, X_te, y_te, d_te = chronological_split(X, y, dates, train_ratio=0.8, purge_gap=horizon)
        # 80/20 Val Split from remaining memory (approx 64% train / 16% val / 20% test overall)
        X_tr, y_tr, d_tr, X_val, y_val, d_val = chronological_split(X_tr_v, y_tr_v, d_tr_v, train_ratio=0.8, purge_gap=horizon)
        
        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        d_train_list.append(d_tr)
        
        X_val_list.append(X_val)
        y_val_list.append(y_val)
        d_val_list.append(d_val)
        
        X_test_list.append(X_te)
        y_test_list.append(y_te)
        d_test_list.append(d_te)
        
        val_group = group[group['date'].isin(d_val)].copy()
        price_val_list.append(val_group[['symbol', 'date', 'close_orig', 'target']])
        
        test_group = group[group['date'].isin(d_te)].copy()
        price_test_list.append(test_group[['symbol', 'date', 'close_orig', 'target']])

    X_train = np.nan_to_num(np.concatenate(X_train_list), posinf=0.0, neginf=0.0)
    y_train = np.concatenate(y_train_list)
    X_val = np.nan_to_num(np.concatenate(X_val_list), posinf=0.0, neginf=0.0)
    y_val = np.concatenate(y_val_list)
    X_test = np.nan_to_num(np.concatenate(X_test_list), posinf=0.0, neginf=0.0)
    y_test = np.concatenate(y_test_list)
    dates_test = np.concatenate(d_test_list)
    
    val_price_df = pd.concat(price_val_list)
    test_price_df = pd.concat(price_test_list)
    
    return X_train, y_train, X_val, y_val, val_price_df, X_test, y_test, test_price_df, len(feature_cols)

def train_pytorch_model(model, X_train, y_train, epochs=3, batch_size=256, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
    return model

def infer_pytorch_model(model, X):
    device = next(model.parameters()).device
    model.eval()
    dataset = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(dataset).cpu().numpy().flatten()
    return preds

def main():
    print("Loading data...")
    df = load_and_merge_data()
    window_size = 20
    print("Preprocessing data for strictly partitioned Optuna hyper-tuning...")
    (X_train, y_train, X_val, y_val, val_price_df, 
     X_test, y_test, test_price_df, num_features) = prepare_dataset_optuna(df, window_size)
    
    print(f"X_train shape: {X_train.shape} | X_val shape: {X_val.shape} | X_test shape: {X_test.shape}")
    
    bt_val = VectorizedBacktester(transaction_cost=0.0025)
    bt_test = VectorizedBacktester(transaction_cost=0.0025)

    def evaluate_study(preds, price_df, gt, threshold):
        bt_df = bt_val.run_backtest(df=price_df, action_scores=preds, ground_truth=gt, 
                                    price_col='close_orig', date_col='date', threshold_long=threshold)
        metrics = bt_val.compute_metrics(bt_df)
        return metrics['Sharpe Ratio']

    # --- 1. LIGHTGBM OPTUNA ---
    def objective_lgb(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 12)
        }
        threshold = trial.suggest_float('threshold', 0.55, 0.90)
        
        lgb_model = LightGBMModel(num_boost_round=100, **params)
        lgb_model.fit(X_train, y_train, X_val, y_val)
        preds = lgb_model.predict_proba(X_val)
        return evaluate_study(preds, val_price_df, y_val, threshold)

    print("\n[LightGBM] Starting Optuna Sweep (30 Trials)...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=30)
    print(f"  Best Val Sharpe: {study_lgb.best_value:.4f}")

    # --- 2. LSTM OPTUNA ---
    def objective_lstm(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        batch_size = trial.suggest_categorical('batch_size', [128, 256])
        threshold = trial.suggest_float('threshold', 0.55, 0.90)

        lstm = LSTMModel(input_size=num_features, hidden_size=hidden_size, num_layers=2)
        model = train_pytorch_model(lstm, X_train, y_train, epochs=5, batch_size=batch_size, lr=lr)
        preds = infer_pytorch_model(model, X_val)
        return evaluate_study(preds, val_price_df, y_val, threshold)

    print("\n[LSTM] Starting Optuna Sweep (20 Trials)...")
    study_lstm = optuna.create_study(direction='maximize')
    study_lstm.optimize(objective_lstm, n_trials=20)
    print(f"  Best Val Sharpe: {study_lstm.best_value:.4f}")

    # --- 3. PatchTST OPTUNA ---
    def objective_patchtst(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256])
        threshold = trial.suggest_float('threshold', 0.55, 0.90)
        
        try:
            ptst = PatchTSTModel(seq_len=window_size, num_features=num_features)
            model = train_pytorch_model(ptst, X_train, y_train, epochs=5, batch_size=batch_size, lr=lr)
            preds = infer_pytorch_model(model, X_val)
            return evaluate_study(preds, val_price_df, y_val, threshold)
        except:
            return -999.0

    print("\n[PatchTST] Starting Optuna Sweep (20 Trials)...")
    study_patch = optuna.create_study(direction='maximize')
    study_patch.optimize(objective_patchtst, n_trials=20)
    print(f"  Best Val Sharpe: {study_patch.best_value:.4f}")

    # --- 4. iTransformer OPTUNA ---
    def objective_itransformer(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256])
        threshold = trial.suggest_float('threshold', 0.55, 0.90)
        
        try:
            itransf = iTransformerModel(seq_len=window_size, num_features=num_features)
            model = train_pytorch_model(itransf, X_train, y_train, epochs=5, batch_size=batch_size, lr=lr)
            preds = infer_pytorch_model(model, X_val)
            return evaluate_study(preds, val_price_df, y_val, threshold)
        except:
            return -999.0

    print("\n[iTransformer] Starting Optuna Sweep (20 Trials)...")
    study_itrans = optuna.create_study(direction='maximize')
    study_itrans.optimize(objective_itransformer, n_trials=20)
    print(f"  Best Val Sharpe: {study_itrans.best_value:.4f}")

    # --- FINAL EVALUATION (TEST SET) ---
    print("\nInstantiating Best Models & Retraining on Train+Val dataset...")
    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    best_models = {
        'LightGBM': study_lgb.best_params,
        'LSTM': study_lstm.best_params,
        'PatchTST': study_patch.best_params,
        'iTransformer': study_itrans.best_params
    }
    
    print("\nPrinting Best Parameters & Dynamic Thresholds:")
    for m, p in best_models.items():
        params_str = ", ".join([f"{k}: {v}" for k, v in p.items() if k != 'threshold'])
        print(f"  {m}: Threshold = {p['threshold']:.3f} | Params = {{{params_str}}}")

    results = {}

    # Retrain LGBM
    p_lgb = study_lgb.best_params
    lgb_full = LightGBMModel(num_boost_round=100, **{k: v for k, v in p_lgb.items() if k != 'threshold'})
    # Cannot do early stopping nicely on Full without split, so we just run 100 rounds
    lgb_full.fit(X_train_full, y_train_full, X_test, y_test)
    results['LightGBM'] = (lgb_full.predict_proba(X_test), p_lgb['threshold'])

    # Retrain LSTM
    p_lstm = study_lstm.best_params
    lstm_full = LSTMModel(input_size=num_features, hidden_size=p_lstm['hidden_size'], num_layers=2)
    m_lstm = train_pytorch_model(lstm_full, X_train_full, y_train_full, epochs=5, batch_size=p_lstm['batch_size'], lr=p_lstm['lr'])
    results['LSTM'] = (infer_pytorch_model(m_lstm, X_test), p_lstm['threshold'])

    # Retrain PatchTST
    if study_patch.best_value > -900:
        p_pt = study_patch.best_params
        pt_full = PatchTSTModel(seq_len=window_size, num_features=num_features)
        m_pt = train_pytorch_model(pt_full, X_train_full, y_train_full, epochs=5, batch_size=p_pt['batch_size'], lr=p_pt['lr'])
        results['PatchTST'] = (infer_pytorch_model(m_pt, X_test), p_pt['threshold'])

    # Retrain iTransformer
    if study_itrans.best_value > -900:
        p_it = study_itrans.best_params
        it_full = iTransformerModel(seq_len=window_size, num_features=num_features)
        m_it = train_pytorch_model(it_full, X_train_full, y_train_full, epochs=5, batch_size=p_it['batch_size'], lr=p_it['lr'])
        results['iTransformer'] = (infer_pytorch_model(m_it, X_test), p_it['threshold'])

    metrics_records = []
    
    for model_name, (preds, threshold) in results.items():
        bt_df = bt_test.run_backtest(
            df=test_price_df, 
            action_scores=preds, 
            ground_truth=test_price_df['target'].values,
            price_col='close_orig', 
            date_col='date',
            threshold_long=threshold
        )
        
        regime_metrics = bt_test.evaluate_by_regime(bt_df)
        
        for regime, stats in regime_metrics.items():
            metrics_records.append({
                'Model': model_name,
                'Regime': regime,
                'Sharpe Ratio': round(stats['Sharpe Ratio'], 2),
                'Max Drawdown (%)': round(stats['Max Drawdown'], 2)
            })

    benchmark_df = pd.DataFrame(metrics_records)
    benchmark_table = benchmark_df.pivot(index='Model', columns='Regime', values=['Sharpe Ratio', 'Max Drawdown (%)'])
    benchmark_table.columns = [f"{col[1]} | {col[0]}" for col in benchmark_table.columns]
    
    print("\n================================= TUNED BENCHMARK TABLE =================================")
    print(benchmark_table.to_string())
    print("=========================================================================================")
    benchmark_table.to_csv('d:/TechPulse/optuna_benchmark_table.csv')

if __name__ == "__main__":
    main()
