# Baseline Models cho Time Series
## ARIMA, GARCH và Linear Models - Nền tảng để so sánh

---

## Mục lục

1. [Tại sao Baseline quan trọng trong Research?](#1-tại-sao-baseline-quan-trọng-trong-research)
2. [So sánh Baseline vs ML vs DL công bằng](#2-so-sánh-baseline-vs-ml-vs-dl-công-bằng)
3. [Rolling vs Expanding Baselines](#3-rolling-vs-expanding-baselines)
4. [Linear Regression](#4-linear-regression)
5. [ARIMA Models](#5-arima-models)
6. [GARCH Models](#6-garch-models)
7. [Naive Forecasting](#7-naive-forecasting)
8. [Baselines cho Classification](#8-baselines-cho-classification)
9. [Statistical Significance Testing](#9-statistical-significance-testing)
10. [Bài tập thực hành](#10-bài-tập-thực-hành)

---

## 1. TẠI SAO BASELINE QUAN TRỌNG TRONG RESEARCH?

### 1.1. Baseline trong Scientific Method

**Baseline không chỉ là "model đơn giản":**
```
Baseline = Hypothesis về performance tối thiểu mà model phức tạp phải beat

Không có baseline → Không có cách đo lường "improvement"
                  → Không biết model có thực sự "học" được gì không
                  → Kết quả không có ý nghĩa khoa học
```

### 1.2. Tại sao Baseline critical?

**1. Prevent Overfitting to Complexity:**
```
Tình huống thực tế:
- Team A: "LSTM đạt MAE = 2.5!"
- Team B: "Nhưng Naive baseline MAE = 2.3..."
- Team A: "..."

→ LSTM của Team A không học được gì cả!
→ Chỉ memorize training data
```

**2. Establish Lower Bound:**
```
Baseline đặt "floor" cho performance:

Naive baseline:     MAE = 5.0    (floor)
Linear Regression:  MAE = 4.2    (baseline)
XGBoost:           MAE = 3.5    (+19% vs LR)
LSTM:              MAE = 3.0    (+29% vs LR)

→ Mỗi bước improvement có reference rõ ràng
```

**3. Detect Data Leakage:**
```
Dấu hiệu data leakage:
- Model phức tạp beat baseline quá nhiều (>50%)
- Test performance tốt hơn validation
- Kết quả "too good to be true"

Baseline giúp detect:
- Naive MAE = 5.0
- Model MAE = 0.5  ← Suspicious! 10x better?
→ Check for lookahead bias!
```

**4. Scientific Reproducibility:**
```
Paper không có baseline → Không reproducible
Reviewer sẽ reject vì:
- "So với gì?"
- "Improvement có significant không?"
- "Có thể chỉ là random variance"
```

### 1.3. Hierarchy of Baselines

```
Level 0: Trivial Baselines (MUST BEAT)
├── Naive (predict last value)
├── Historical Mean
└── Random prediction

Level 1: Simple Models (SHOULD BEAT)
├── Moving Average
├── Exponential Smoothing
└── Linear Regression

Level 2: Statistical Models (STANDARD COMPARISON)
├── ARIMA/SARIMA
├── GARCH (for volatility)
└── VAR (for multivariate)

Level 3: Machine Learning (FAIR COMPARISON)
├── XGBoost/LightGBM
├── Random Forest
└── Ridge/Lasso Regression

Level 4: Deep Learning (YOUR MODEL)
├── LSTM/GRU
├── Transformer
└── Novel architectures

RULE: Model ở Level N phải beat Level N-1
```

### 1.4. Research-Grade Checklist

```
□ Implement ít nhất 3 baselines (different levels)
□ Report metrics cho TẤT CẢ baselines, không chỉ best
□ Statistical significance tests
□ Multiple random seeds
□ Cross-validation hoặc walk-forward
□ Ablation study (what contributes to improvement?)
```

---

## 2. SO SÁNH BASELINE VS ML VS DL CÔNG BẰNG

### 2.1. Unfair Comparison (Phổ biến nhưng SAI)

**Các lỗi thường gặp:**

**1. Data Leakage cho DL, không cho Baseline:**
```python
# SAI: Baseline không được scale đúng
# Baseline: dùng raw data
baseline_pred = naive_forecast(raw_data)

# DL: được scale với toàn bộ data
scaler.fit(all_data)  # ← Leak!
dl_pred = lstm.predict(scaled_data)

# → Comparison không fair!
```

**2. Hyperparameter Tuning không đều:**
```python
# SAI: DL được tune kỹ, baseline mặc định
# Baseline: mặc định
arima = ARIMA(order=(1,1,1))  # No tuning

# DL: tune 100 configurations
lstm = tune_hyperparameters(lstm, 100_configs)  # Heavy tuning

# → Comparison không fair!
```

**3. Different Test Sets:**
```python
# SAI: Test sets khác nhau
baseline_mse = evaluate(baseline, test_set_1)
dl_mse = evaluate(dl_model, test_set_2)  # Different!

# → Comparison không fair!
```

### 2.2. Fair Comparison Protocol

**Protocol đúng:**

```python
def fair_comparison_protocol(data, models_config):
    """
    Fair comparison between Baseline, ML, and DL models
    """
    results = []
    
    # 1. SAME data split for all models
    train, val, test = time_series_split(data, ratios=[0.7, 0.15, 0.15])
    
    # 2. SAME preprocessing pipeline
    scaler = StandardScaler()
    scaler.fit(train)  # Fit ONLY on train
    
    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    
    # 3. SAME hyperparameter tuning budget (optional but fair)
    tuning_budget = 50  # Same for all
    
    # 4. Train and evaluate EACH model
    for model_name, model_class in models_config.items():
        print(f"\n=== {model_name} ===")
        
        # Tune (if applicable)
        if hasattr(model_class, 'tune'):
            model = model_class.tune(train_scaled, val_scaled, budget=tuning_budget)
        else:
            model = model_class()
        
        # Train
        model.fit(train_scaled)
        
        # Predict on SAME test set
        pred = model.predict(test_scaled)
        
        # Inverse transform
        pred_original = scaler.inverse_transform(pred)
        test_original = scaler.inverse_transform(test_scaled)
        
        # Evaluate with SAME metrics
        metrics = compute_metrics(test_original, pred_original)
        metrics['model'] = model_name
        results.append(metrics)
    
    # 5. Statistical significance tests
    results_df = pd.DataFrame(results)
    significance = statistical_tests(results)
    
    return results_df, significance
```

### 2.3. Comparison Table Template

**Research-grade comparison table:**

```python
def create_comparison_table(results):
    """
    Create publication-ready comparison table
    """
    table = pd.DataFrame(results)
    
    # Add improvement column
    baseline_mae = table[table['model'] == 'Naive']['MAE'].values[0]
    table['Improvement vs Naive'] = (baseline_mae - table['MAE']) / baseline_mae * 100
    
    # Add rank
    table['Rank'] = table['MAE'].rank()
    
    # Format
    table = table.round({
        'MAE': 4, 
        'RMSE': 4, 
        'MAPE': 2,
        'Improvement vs Naive': 1
    })
    
    return table

# Example output:
"""
| Model            | MAE    | RMSE   | MAPE   | Improvement vs Naive | Rank |
|------------------|--------|--------|--------|---------------------|------|
| Naive            | 0.0523 | 0.0687 | 5.23%  | 0.0%                | 5    |
| Moving Average   | 0.0498 | 0.0654 | 4.98%  | 4.8%                | 4    |
| ARIMA(1,1,1)     | 0.0456 | 0.0598 | 4.56%  | 12.8%               | 3    |
| XGBoost          | 0.0412 | 0.0542 | 4.12%  | 21.2%               | 2    |
| LSTM             | 0.0389 | 0.0512 | 3.89%  | 25.6%               | 1    |
"""
```

### 2.4. What to Report in Paper

```
Table 1: Comparison of forecasting models on FPT stock (2020-2024)

Model           | MAE (±std)      | RMSE (±std)     | p-value vs LSTM
----------------|-----------------|-----------------|----------------
Naive           | 5.23 (±0.42)    | 6.87 (±0.51)    | <0.001***
Moving Average  | 4.98 (±0.38)    | 6.54 (±0.47)    | <0.001***
ARIMA(1,1,1)    | 4.56 (±0.35)    | 5.98 (±0.44)    | 0.003**
XGBoost         | 4.12 (±0.31)    | 5.42 (±0.40)    | 0.045*
LSTM (ours)     | 3.89 (±0.29)    | 5.12 (±0.38)    | -

Notes:
- Results averaged over 5-fold time series cross-validation
- *** p<0.001, ** p<0.01, * p<0.05 (Diebold-Mariano test)
- All models use same train/test split and preprocessing
```

---

## 3. ROLLING VS EXPANDING BASELINES

### 3.1. Static Baseline (Bad Practice)

```python
# SAI: Train baseline 1 lần, dùng mãi
baseline = ARIMA(train_data, order=(1,1,1))
baseline.fit()

# Predict for all test period
all_predictions = baseline.forecast(len(test_data))

# Vấn đề:
# - Model không được update với new data
# - Performance degrade over time
# - Không realistic
```

### 3.2. Rolling Baseline (Realistic)

**Ý tưởng:** Retrain model với fixed window size, slide theo thời gian.

```
Time: [1][2][3][4][5][6][7][8][9][10][11][12]

Window 1: [████████] → Predict [9]
Window 2:   [████████] → Predict [10]
Window 3:     [████████] → Predict [11]
Window 4:       [████████] → Predict [12]

Train window cố định, slide theo thời gian
```

**Code:**
```python
def rolling_baseline(data, model_class, window_size=252, **model_params):
    """
    Rolling baseline with fixed window
    
    Args:
        data: Full time series
        model_class: Model class (e.g., ARIMA, LinearRegression)
        window_size: Training window size (252 = 1 year trading days)
    
    Returns:
        predictions, actuals, metrics_per_step
    """
    predictions = []
    actuals = []
    
    for i in range(window_size, len(data) - 1):
        # Training window
        train = data[i - window_size:i]
        
        # Actual next value
        actual = data[i + 1]
        
        # Train model
        model = model_class(**model_params)
        model.fit(train)
        
        # Predict 1 step ahead
        pred = model.forecast(steps=1)[0]
        
        predictions.append(pred)
        actuals.append(actual)
    
    return np.array(predictions), np.array(actuals)

# Usage
predictions, actuals = rolling_baseline(
    data=df['close'],
    model_class=ARIMA,
    window_size=252,
    order=(1, 1, 1)
)

mae = mean_absolute_error(actuals, predictions)
print(f"Rolling ARIMA MAE: {mae:.4f}")
```

### 3.3. Expanding Baseline

**Ý tưởng:** Training window tăng dần, dùng tất cả data quá khứ.

```
Time: [1][2][3][4][5][6][7][8][9][10][11][12]

Window 1: [████████]     → Predict [9]
Window 2: [█████████]    → Predict [10]
Window 3: [██████████]   → Predict [11]
Window 4: [███████████]  → Predict [12]

Train window tăng dần
```

**Code:**
```python
def expanding_baseline(data, model_class, min_train_size=252, **model_params):
    """
    Expanding baseline with growing window
    
    Args:
        data: Full time series
        model_class: Model class
        min_train_size: Minimum training samples
    
    Returns:
        predictions, actuals
    """
    predictions = []
    actuals = []
    
    for i in range(min_train_size, len(data) - 1):
        # Training window: from start to current
        train = data[:i]
        
        # Actual next value
        actual = data[i + 1]
        
        # Train model
        model = model_class(**model_params)
        model.fit(train)
        
        # Predict 1 step ahead
        pred = model.forecast(steps=1)[0]
        
        predictions.append(pred)
        actuals.append(actual)
    
    return np.array(predictions), np.array(actuals)
```

### 3.4. So sánh Rolling vs Expanding

| Aspect | Rolling | Expanding |
|--------|---------|-----------|
| **Training size** | Cố định | Tăng dần |
| **Adapt to regime** | Tốt (quên data cũ) | Kém (remember all) |
| **Data efficiency** | Bỏ data cũ | Dùng tất cả |
| **Computation** | Nhẹ hơn | Nặng dần |
| **Use case** | Regime changes | Stable patterns |

**Recommendation:**
```
Rolling:   Khi market có regime changes (bull → bear)
Expanding: Khi pattern ổn định, cần nhiều data

Best practice: Test CẢ HAI và report cả hai
```

### 3.5. Complete Rolling Evaluation

```python
def complete_rolling_evaluation(data, models_config, window_size=252):
    """
    Evaluate multiple models with rolling window
    """
    results = {name: {'preds': [], 'actuals': []} for name in models_config}
    
    for i in range(window_size, len(data) - 1):
        train = data[i - window_size:i]
        actual = data[i + 1]
        
        for name, model_class in models_config.items():
            model = model_class()
            model.fit(train)
            pred = model.predict(1)[0]
            
            results[name]['preds'].append(pred)
            results[name]['actuals'].append(actual)
    
    # Compute metrics
    metrics = []
    for name in models_config:
        preds = np.array(results[name]['preds'])
        actuals = np.array(results[name]['actuals'])
        
        metrics.append({
            'Model': name,
            'MAE': mean_absolute_error(actuals, preds),
            'RMSE': np.sqrt(mean_squared_error(actuals, preds)),
            'N_predictions': len(preds)
        })
    
    return pd.DataFrame(metrics)
```

---

## 4. LINEAR REGRESSION

### 4.1. Linear Regression cho Time Series

**Công thức:**
```
price_tomorrow = w1×close_today + w2×ma20 + w3×rsi + w4×macd + b
```

**Code:**
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# Chuẩn bị data
feature_cols = ['close', 'ma_20', 'rsi_14', 'macd', 'volatility_20']
X = df[feature_cols]
y = df['close'].shift(-1)  # Target: giá ngày mai

# Drop NaN
data = pd.concat([X, y.rename('target')], axis=1).dropna()
X = data[feature_cols]
y = data['target']

# Train/test split (theo thời gian)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Linear Regression MAE: {mae:.4f}")
print(f"Linear Regression RMSE: {rmse:.4f}")
```

### 4.2. Regularized Versions

```python
# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1 regularization - sparse)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Compare coefficients
print("Feature Importance:")
for feat, lr_coef, ridge_coef, lasso_coef in zip(
    feature_cols, model.coef_, ridge.coef_, lasso.coef_
):
    print(f"  {feat}: LR={lr_coef:.4f}, Ridge={ridge_coef:.4f}, Lasso={lasso_coef:.4f}")
```

---

## 5. ARIMA MODELS

### 5.1. ARIMA(p, d, q)

**Công thức:**
```
ARIMA = AR(p) + I(d) + MA(q)

- p: AutoRegressive order (dùng p lags của y)
- d: Differencing order (làm stationary)
- q: Moving Average order (dùng q lags của error)
```

**Code:**
```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

# 1. Check stationarity
result = adfuller(df['close'].dropna())
print(f"ADF p-value: {result[1]:.4f}")
print(f"Stationary: {result[1] < 0.05}")

# 2. Auto ARIMA để tìm p, d, q
auto_model = auto_arima(
    df['close'],
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,
    seasonal=False,
    trace=True,
    suppress_warnings=True
)
print(f"Best order: {auto_model.order}")

# 3. Train ARIMA
train = df['close'][:split_idx]
test = df['close'][split_idx:]

model = ARIMA(train, order=auto_model.order)
fitted = model.fit()
print(fitted.summary())

# 4. Forecast
forecast = fitted.forecast(steps=len(test))
mae = mean_absolute_error(test, forecast)
print(f"ARIMA MAE: {mae:.4f}")
```

---

## 6. GARCH MODELS

### 6.1. GARCH cho Volatility

**GARCH dự đoán VOLATILITY, không dự đoán price.**

**Code:**
```python
from arch import arch_model

# Returns (%)
returns = df['close'].pct_change().dropna() * 100

# Train GARCH(1,1)
model = arch_model(returns, vol='Garch', p=1, q=1)
fitted = model.fit(disp='off')
print(fitted.summary())

# Forecast volatility
forecast = fitted.forecast(horizon=30)
predicted_vol = np.sqrt(forecast.variance.values[-1, :])

print(f"30-day volatility forecast: {predicted_vol}")
```

---

## 7. NAIVE FORECASTING

### 7.1. Naive Baselines

```python
def naive_last_value(train, test):
    """Predict = last value"""
    return np.full(len(test), train.iloc[-1])

def naive_mean(train, test):
    """Predict = historical mean"""
    return np.full(len(test), train.mean())

def naive_drift(train, test):
    """Predict = last value + average change"""
    avg_change = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    return train.iloc[-1] + avg_change * np.arange(1, len(test) + 1)

def seasonal_naive(train, test, period=5):
    """Predict = value from same period last cycle"""
    forecast = []
    combined = pd.concat([train, test])
    for i in range(len(train), len(combined)):
        forecast.append(combined.iloc[i - period])
    return np.array(forecast)

# Evaluate all naive methods
for name, func in [
    ('Naive (Last)', naive_last_value),
    ('Naive (Mean)', naive_mean),
    ('Naive (Drift)', naive_drift),
]:
    pred = func(train, test)
    mae = mean_absolute_error(test, pred)
    print(f"{name}: MAE = {mae:.4f}")
```

---

## 8. BASELINES CHO CLASSIFICATION

### 8.1. Tại sao cần Classification Baselines?

**Classification tasks trong trading:**
- Dự đoán Up/Down
- Dự đoán Strong Up/Weak Up/Flat/Weak Down/Strong Down
- Trading signals (Buy/Hold/Sell)

### 8.2. Classification Baselines

**1. Random Baseline:**
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def random_baseline(y_train, y_test):
    """Random prediction based on class distribution"""
    classes, counts = np.unique(y_train, return_counts=True)
    probs = counts / len(y_train)
    predictions = np.random.choice(classes, size=len(y_test), p=probs)
    return predictions

# Usage
y_pred_random = random_baseline(y_train, y_test)
print(f"Random Accuracy: {accuracy_score(y_test, y_pred_random):.4f}")
```

**2. Majority Class Baseline:**
```python
def majority_baseline(y_train, y_test):
    """Always predict the most frequent class"""
    majority_class = y_train.value_counts().idxmax()
    return np.full(len(y_test), majority_class)

y_pred_majority = majority_baseline(y_train, y_test)
print(f"Majority Accuracy: {accuracy_score(y_test, y_pred_majority):.4f}")
```

**3. Stratified Random Baseline:**
```python
from sklearn.dummy import DummyClassifier

# Stratified random
dummy_stratified = DummyClassifier(strategy='stratified')
dummy_stratified.fit(X_train, y_train)
y_pred_stratified = dummy_stratified.predict(X_test)
print(f"Stratified Accuracy: {accuracy_score(y_test, y_pred_stratified):.4f}")
```

**4. Prior Day Momentum Baseline:**
```python
def momentum_baseline(returns_series):
    """
    Predict: If yesterday was up, today will be up
    """
    yesterday = returns_series.shift(1)
    predictions = (yesterday > 0).astype(int)  # 1=Up, 0=Down
    return predictions

y_pred_momentum = momentum_baseline(df['return'])
actuals = (df['return'] > 0).astype(int)
print(f"Momentum Accuracy: {accuracy_score(actuals[1:], y_pred_momentum[1:]):.4f}")
```

**5. Mean-Reversion Baseline:**
```python
def mean_reversion_baseline(returns_series):
    """
    Predict: If yesterday was up, today will be down (contrarian)
    """
    yesterday = returns_series.shift(1)
    predictions = (yesterday < 0).astype(int)  # Opposite of momentum
    return predictions

y_pred_mr = mean_reversion_baseline(df['return'])
print(f"Mean-Reversion Accuracy: {accuracy_score(actuals[1:], y_pred_mr[1:]):.4f}")
```

### 8.3. Complete Classification Baseline Comparison

```python
def classification_baseline_comparison(X_train, X_test, y_train, y_test):
    """
    Compare all classification baselines
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    
    results = []
    
    # 1. Random
    dummy_random = DummyClassifier(strategy='uniform')
    dummy_random.fit(X_train, y_train)
    y_pred = dummy_random.predict(X_test)
    results.append({
        'Model': 'Random',
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='weighted')
    })
    
    # 2. Majority
    dummy_majority = DummyClassifier(strategy='most_frequent')
    dummy_majority.fit(X_train, y_train)
    y_pred = dummy_majority.predict(X_test)
    results.append({
        'Model': 'Majority Class',
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='weighted')
    })
    
    # 3. Stratified
    dummy_strat = DummyClassifier(strategy='stratified')
    dummy_strat.fit(X_train, y_train)
    y_pred = dummy_strat.predict(X_test)
    results.append({
        'Model': 'Stratified Random',
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='weighted')
    })
    
    # 4. Logistic Regression (simple model baseline)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='weighted')
    })
    
    return pd.DataFrame(results).sort_values('F1', ascending=False)

# Usage
results = classification_baseline_comparison(X_train, X_test, y_train, y_test)
print(results)
```

### 8.4. Important: Imbalanced Classes

```python
# Trong trading, classes thường imbalanced
# Ví dụ: 52% Up, 48% Down (slightly biased)

# Metric quan trọng:
# - Accuracy có thể misleading (majority baseline đạt 52%)
# - F1-score weighted hoặc macro
# - ROC-AUC

print("\nClass Distribution:")
print(y_train.value_counts(normalize=True))

# Nếu imbalanced, majority baseline sẽ có accuracy cao
# Nhưng F1 và AUC sẽ thấp
```

---

## 9. STATISTICAL SIGNIFICANCE TESTING

### 9.1. Tại sao cần Statistical Tests?

**Vấn đề:**
```
Model A: MAE = 4.52
Model B: MAE = 4.48

Q: Model B có thực sự tốt hơn không?
A: Không chắc! Có thể chỉ là random variance.

→ Cần statistical test để xác nhận
```

### 9.2. Diebold-Mariano Test

**Diebold-Mariano Test:**
- So sánh predictive accuracy của 2 models
- H0: Hai models có accuracy bằng nhau
- H1: Model B tốt hơn (hoặc khác) Model A

```python
from scipy import stats

def diebold_mariano_test(actual, pred1, pred2, h=1, power=2):
    """
    Diebold-Mariano test for comparing two forecasts
    
    Args:
        actual: Actual values
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        h: Forecast horizon
        power: Power for loss function (1=MAE, 2=MSE)
    
    Returns:
        DM statistic, p-value
    """
    # Loss differences
    e1 = actual - pred1
    e2 = actual - pred2
    
    if power == 1:
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1**power - e2**power
    
    # Mean and variance of d
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    
    # Adjust for autocorrelation in forecast errors
    T = len(d)
    
    # Autocovariance adjustment for h > 1
    if h > 1:
        autocov = 0
        for i in range(1, h):
            autocov += np.cov(d[:-i], d[i:])[0, 1]
        var_d = var_d + 2 * autocov / T
    
    # DM statistic
    dm_stat = mean_d / np.sqrt(var_d / T)
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

# Usage
dm_stat, p_value = diebold_mariano_test(
    actual=y_test.values,
    pred1=baseline_predictions,
    pred2=lstm_predictions
)

print(f"DM Statistic: {dm_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ LSTM significantly different from baseline (p < 0.05)")
else:
    print("→ No significant difference (p >= 0.05)")
```

### 9.3. Paired t-test for Errors

```python
def paired_ttest_errors(actual, pred1, pred2):
    """
    Paired t-test on absolute errors
    """
    errors1 = np.abs(actual - pred1)
    errors2 = np.abs(actual - pred2)
    
    t_stat, p_value = stats.ttest_rel(errors1, errors2)
    
    return t_stat, p_value

t_stat, p_value = paired_ttest_errors(y_test.values, baseline_pred, model_pred)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
```

### 9.4. Bootstrap Confidence Intervals

```python
def bootstrap_confidence_interval(actual, predictions, metric_func, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap confidence interval for a metric
    """
    n = len(actual)
    metrics = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.randint(0, n, n)
        sample_actual = actual[idx]
        sample_pred = predictions[idx]
        
        # Compute metric
        metric = metric_func(sample_actual, sample_pred)
        metrics.append(metric)
    
    # Confidence interval
    alpha = 1 - ci
    lower = np.percentile(metrics, alpha/2 * 100)
    upper = np.percentile(metrics, (1 - alpha/2) * 100)
    
    return np.mean(metrics), lower, upper

# Usage
mean_mae, lower, upper = bootstrap_confidence_interval(
    y_test.values,
    model_predictions,
    mean_absolute_error
)

print(f"MAE: {mean_mae:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
```

### 9.5. Multiple Comparisons Correction

**Khi so sánh nhiều models:**
```python
from statsmodels.stats.multitest import multipletests

def compare_multiple_models(actual, predictions_dict, baseline_key='Naive'):
    """
    Compare multiple models with Bonferroni correction
    """
    baseline_pred = predictions_dict[baseline_key]
    
    p_values = []
    model_names = []
    
    for name, pred in predictions_dict.items():
        if name == baseline_key:
            continue
        
        _, p_value = diebold_mariano_test(actual, baseline_pred, pred)
        p_values.append(p_value)
        model_names.append(name)
    
    # Bonferroni correction
    reject, corrected_p, _, _ = multipletests(p_values, method='bonferroni')
    
    results = pd.DataFrame({
        'Model': model_names,
        'P-value (raw)': p_values,
        'P-value (corrected)': corrected_p,
        'Significant': reject
    })
    
    return results

# Usage
results = compare_multiple_models(
    actual=y_test.values,
    predictions_dict={
        'Naive': naive_pred,
        'ARIMA': arima_pred,
        'XGBoost': xgb_pred,
        'LSTM': lstm_pred
    }
)
print(results)
```

### 9.6. Complete Significance Report

```python
def full_significance_report(actual, predictions_dict, baseline_key='Naive'):
    """
    Generate complete significance report
    """
    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE REPORT")
    print("=" * 60)
    
    baseline_pred = predictions_dict[baseline_key]
    baseline_mae = mean_absolute_error(actual, baseline_pred)
    
    print(f"\nBaseline ({baseline_key}): MAE = {baseline_mae:.4f}")
    print("\nPairwise Comparisons with Baseline:")
    print("-" * 60)
    
    for name, pred in predictions_dict.items():
        if name == baseline_key:
            continue
        
        mae = mean_absolute_error(actual, pred)
        improvement = (baseline_mae - mae) / baseline_mae * 100
        
        dm_stat, p_value = diebold_mariano_test(actual, baseline_pred, pred)
        
        # Bootstrap CI
        mean_mae, lower, upper = bootstrap_confidence_interval(
            actual, pred, mean_absolute_error
        )
        
        print(f"\n{name}:")
        print(f"  MAE: {mae:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
        print(f"  Improvement vs {baseline_key}: {improvement:.1f}%")
        print(f"  DM Test: stat={dm_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.001:
            print("  → Highly significant (p < 0.001) ***")
        elif p_value < 0.01:
            print("  → Very significant (p < 0.01) **")
        elif p_value < 0.05:
            print("  → Significant (p < 0.05) *")
        else:
            print("  → Not significant (p >= 0.05)")

# Usage
full_significance_report(y_test.values, all_predictions)
```

---

## 10. BÀI TẬP THỰC HÀNH

### Bài tập 1: Complete Baseline Pipeline

**Yêu cầu:**
1. Implement 5 baselines (Naive, MA, ARIMA, Linear, XGBoost)
2. Rolling evaluation (window=252)
3. Statistical significance tests
4. Report với confidence intervals

### Bài tập 2: Classification Baselines

**Yêu cầu:**
1. Tạo binary target (Up/Down)
2. Implement 4 classification baselines
3. So sánh với Logistic Regression
4. Report Accuracy, F1, AUC

### Bài tập 3: Fair Comparison ML vs DL

**Yêu cầu:**
1. Train XGBoost và LSTM
2. SAME preprocessing cho cả hai
3. SAME hyperparameter tuning budget
4. Statistical significance tests

---

## Kiểm tra hiểu bài

- [ ] Giải thích được tại sao baseline critical
- [ ] Implement được fair comparison protocol
- [ ] Phân biệt được rolling vs expanding baselines
- [ ] Implement được classification baselines
- [ ] Thực hiện được Diebold-Mariano test
- [ ] Hiểu được multiple comparison correction

---

## Tài liệu tham khảo

**Papers:**
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy"
- Makridakis et al. (2020). "M4 Competition: Results, Findings, and Conclusions"

**Books:**
- "Forecasting: Principles and Practice" - Rob Hyndman

---

## Bước tiếp theo

Sau khi hoàn thành:
- `02_ML_MODELS.md` - XGBoost, LightGBM, Random Forest
- `03_LSTM_GRU.md` - Deep Learning cho Time Series
