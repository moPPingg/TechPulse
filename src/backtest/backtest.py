import pandas as pd
import numpy as np

class VectorizedBacktester:
    """
    Vectorized Backtesting Engine for the Green Dragon Trading System.
    Evaluates Action Scores (probabilities) into Long/Short positions and 
    computes rigorous financial metrics including transaction costs.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.0025,  # 0.15% commission + 0.1% slippage
                 trading_days_per_year: int = 252,
                 risk_free_rate: float = 0.0):
        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.annual_factor = trading_days_per_year
        self.rf = risk_free_rate

    def generate_signals(self, 
                         action_scores: np.ndarray, 
                         threshold_long: float = 0.6) -> np.ndarray:
        """
        Converts Action Scores ∈ [0,1] into positions {1, 0}.
        Strictly Long-Only for VN spot constraints.
        > threshold_long -> Long (1)
        Otherwise -> Flat (0)
        """
        positions = np.zeros_like(action_scores, dtype=float)
        positions[action_scores > threshold_long] = 1.0
        return positions

    def run_backtest(self, 
                     df: pd.DataFrame, 
                     action_scores: np.ndarray,
                     ground_truth: np.ndarray = None,
                     price_col: str = 'close',
                     date_col: str = 'date',
                     threshold_long: float = 0.6) -> pd.DataFrame:
        """
        Executes the vectorized backtest on historical price data.
        Calculates dynamic 20-day historical volatility and 252-day rolling percentiles to classify Regimes.
        """
        if len(df) != len(action_scores):
            raise ValueError("Length of price dataframe and action_scores must match.")
            
        bt_df = df[[date_col, price_col]].copy()
        if date_col != 'date':
            bt_df.rename(columns={date_col: 'date'}, inplace=True)
        bt_df['date'] = pd.to_datetime(bt_df['date'])
        
        # Calculate daily log returns and volatility, grouped by symbol if available
        if 'symbol' in bt_df.columns:
            bt_df['log_ret'] = bt_df.groupby('symbol')[price_col].apply(lambda x: np.log(x / x.shift(1))).reset_index(level=0, drop=True)
            bt_df['volatility_20d'] = bt_df.groupby('symbol')['log_ret'].transform(lambda x: x.rolling(20).std() * np.sqrt(self.annual_factor))
            
            def rank_pct(x):
                if len(x) < 2: return 0.5
                return pd.Series(x).rank(pct=True).iloc[-1]
                
            bt_df['vol_percentile'] = bt_df.groupby('symbol')['volatility_20d'].transform(
                lambda x: x.rolling(252, min_periods=min(252, max(len(x)//4, 20))).apply(rank_pct)
            ).bfill().fillna(0.5)
            
            bt_df['asset_return'] = bt_df.groupby('symbol')[price_col].pct_change().shift(-1).fillna(0.0)
            
            # Trade mask must not leak across symbols
            bt_df['action_score'] = action_scores
            bt_df['position'] = self.generate_signals(action_scores, threshold_long=threshold_long)
            bt_df['trade_mask'] = bt_df.groupby('symbol')['position'].diff().fillna(bt_df['position']) != 0
            
        else:
            bt_df['log_ret'] = np.log(bt_df[price_col] / bt_df[price_col].shift(1))
            bt_df['volatility_20d'] = bt_df['log_ret'].rolling(window=20).std() * np.sqrt(self.annual_factor)
            
            min_p = min(252, max(len(bt_df) // 4, 20))
            def rank_pct(x):
                if len(x) < 2: return 0.5
                return pd.Series(x).rank(pct=True).iloc[-1]
                
            bt_df['vol_percentile'] = bt_df['volatility_20d'].rolling(window=252, min_periods=min_p).apply(rank_pct)
            bt_df['vol_percentile'] = bt_df['vol_percentile'].bfill().fillna(0.5)
            bt_df['asset_return'] = bt_df[price_col].pct_change().shift(-1).fillna(0.0)
            
            bt_df['action_score'] = action_scores
            bt_df['position'] = self.generate_signals(action_scores, threshold_long=threshold_long)
            bt_df['trade_mask'] = bt_df['position'].diff().fillna(bt_df['position']) != 0
        
        # Classify Regimes
        bt_df['regime'] = 'Regime 2 (Normal)'
        bt_df.loc[bt_df['vol_percentile'] < 0.4, 'regime'] = 'Regime 1 (Stable)'
        bt_df.loc[bt_df['vol_percentile'] > 0.8, 'regime'] = 'Regime 3 (Extreme)'

        if ground_truth is not None:
            bt_df['ground_truth'] = ground_truth
            
        bt_df['strat_return'] = (bt_df['position'] * bt_df['asset_return'])
        
        position_change = bt_df['position'].diff().fillna(bt_df['position']).abs()
        bt_df['tc_penalty'] = position_change * self.tc
        bt_df['strat_return_net'] = bt_df['strat_return'] - bt_df['tc_penalty']
        
        bt_df['cum_return'] = (1 + bt_df['strat_return_net']).cumprod()
        bt_df['portfolio_value'] = self.initial_capital * bt_df['cum_return']
        
        bt_df['cum_max'] = bt_df['cum_return'].cummax()
        bt_df['drawdown'] = (bt_df['cum_return'] - bt_df['cum_max']) / bt_df['cum_max']
        
        return bt_df

    def compute_metrics(self, bt_df: pd.DataFrame) -> dict:
        """
        Computes summary statistics (Sharpe, MDD, CumRet) for a given dataframe subset.
        """
        returns = bt_df['strat_return_net']
        
        cum_ret = (1 + returns).prod() - 1.0 if len(returns) > 0 else 0.0
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        if std_ret > 0:
            sharpe_ratio = ((mean_ret - self.rf) / std_ret) * np.sqrt(self.annual_factor)
        else:
            sharpe_ratio = 0.0
            
        if 'cum_return' in bt_df.columns:
            # Recalculate drawdown internally if evaluated on a slice
            sub_cum = (1 + returns).cumprod()
            sub_max = sub_cum.cummax()
            dd = (sub_cum - sub_max) / sub_max
            max_drawdown = dd.min()
        else:
            max_drawdown = 0.0
            
        num_trades = bt_df['trade_mask'].sum() if 'trade_mask' in bt_df.columns else 0
        
        res = {
            "Cumulative Return": float(cum_ret),
            "Sharpe Ratio": float(sharpe_ratio),
            "Max Drawdown": float(max_drawdown) * 100.0,
        }
        
        # Calculate F1 Score if ground_truth available
        if 'ground_truth' in bt_df.columns and 'position' in bt_df.columns:
            from sklearn.metrics import f1_score
            y_pred_bin = (bt_df['position'] >= 0.5).astype(int)
            y_true_bin = bt_df['ground_truth'].astype(int)
            try:
                f1 = f1_score(y_true_bin, y_pred_bin)
            except Exception:
                f1 = 0.0
            res["F1-Score"] = float(f1)
            
        return res
        
    def evaluate_by_regime(self, bt_df: pd.DataFrame) -> dict:
        """
        Computes the Sharpe Ratio, Max Drawdown, and F1-Score globally and entirely isolated by Regime clusters.
        """
        results = {}
        # Overall
        results['Overall'] = self.compute_metrics(bt_df)
        
        # By Regime
        for regime in sorted(bt_df['regime'].unique()):
            sub_df = bt_df[bt_df['regime'] == regime].copy()
            if len(sub_df) > 0:
                results[regime] = self.compute_metrics(sub_df)
                
        return results
