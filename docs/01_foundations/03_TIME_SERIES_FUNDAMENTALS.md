# Time Series Fundamentals
## Hiểu đặc thù của dữ liệu chuỗi thời gian

---

## Mục lục

1. [Time Series là gì?](#1-time-series-là-gì)
2. [Autocorrelation, ACF, PACF](#2-autocorrelation-acf-pacf)
3. [Lag Features vs Rolling Features](#3-lag-features-vs-rolling-features)
4. [Train-Test Split cho Time Series](#4-train-test-split-cho-time-series)
5. [Walk-Forward Validation (Research-Grade)](#5-walk-forward-validation-research-grade)
6. [Lookahead Bias](#6-lookahead-bias)
7. [Stationarity](#7-stationarity)
8. [Differencing và Transformations](#8-differencing-và-transformations)
9. [Mean-Reversion vs Momentum](#9-mean-reversion-vs-momentum)
10. [Forecasting Metrics](#10-forecasting-metrics)
11. [Multi-Step Forecasting](#11-multi-step-forecasting)
12. [Bài tập thực hành](#12-bài-tập-thực-hành)

---

## 1. TIME SERIES LÀ GÌ?

### Định nghĩa

**Time Series = Dữ liệu được thu thập theo thứ tự thời gian**

Mỗi điểm dữ liệu gắn với một thời điểm cụ thể, và thứ tự này quan trọng.

### Ví dụ

**Là Time Series:**
- Giá cổ phiếu hàng ngày
- Nhiệt độ hàng giờ
- Doanh số bán hàng hàng tháng

**Không phải Time Series:**
- Chiều cao của học sinh trong lớp
- Giá nhà ở các quận khác nhau

### Đặc điểm quan trọng

**1. Thứ tự thời gian (Temporal Ordering):**
```
KHÔNG THỂ shuffle data!

Sai:  [2024-01-05, 2024-01-01, 2024-01-03]
Đúng: [2024-01-01, 2024-01-02, 2024-01-03]
```

**2. Phụ thuộc thời gian (Temporal Dependence):**
```
Giá hôm nay phụ thuộc vào giá hôm qua
→ Cần features từ quá khứ (lagged features)
→ Cần models có "memory" (LSTM, ARIMA)
```

**3. Trend và Seasonality:**
```
Trend: Xu hướng dài hạn (tăng/giảm)
Seasonality: Pattern lặp lại theo chu kỳ (tuần, tháng, năm)
```

---

## 2. AUTOCORRELATION, ACF, PACF

### 2.1. Autocorrelation là gì?

**Định nghĩa:** Tương quan của chuỗi với chính nó ở các thời điểm khác nhau (lags).

**Ý tưởng:**
```
Autocorrelation lag-1: Correlation giữa y(t) và y(t-1)
Autocorrelation lag-5: Correlation giữa y(t) và y(t-5)
```

### Ví dụ trực quan

**Positive Autocorrelation (Momentum):**
```
Nếu hôm nay tăng → Ngày mai có xu hướng tăng

Day 1: +2%
Day 2: +1.8%  ← Cùng chiều
Day 3: +1.5%  ← Cùng chiều
Day 4: +1.2%  ← Cùng chiều

→ Chuỗi có momentum, trend-following strategy có thể hiệu quả
```

**Negative Autocorrelation (Mean-Reversion):**
```
Nếu hôm nay tăng → Ngày mai có xu hướng giảm

Day 1: +2%
Day 2: -1.5%  ← Ngược chiều
Day 3: +1.8%  ← Ngược chiều
Day 4: -1.2%  ← Ngược chiều

→ Chuỗi mean-revert, contrarian strategy có thể hiệu quả
```

**Zero Autocorrelation (Random Walk):**
```
Hôm nay tăng/giảm không ảnh hưởng ngày mai

Day 1: +2%
Day 2: -0.5%  ← Ngẫu nhiên
Day 3: +1.2%  ← Ngẫu nhiên
Day 4: +0.3%  ← Ngẫu nhiên

→ Gần như không dự đoán được, efficient market
```

### 2.2. ACF (Autocorrelation Function)

**ACF(k) = Correlation giữa y(t) và y(t-k)**

Bao gồm cả tương quan trực tiếp và gián tiếp.

```python
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Plot ACF
fig, ax = plt.subplots(figsize=(12, 5))
plot_acf(returns.dropna(), lags=20, ax=ax)
ax.set_title('Autocorrelation Function (ACF)')
ax.set_xlabel('Lag')
ax.set_ylabel('Correlation')
plt.show()
```

**Cách đọc ACF:**
```
         Confidence interval (vùng xanh)
              ↓
    │  ████████ lag 1 (significant - vượt ra ngoài)
    │  ███      lag 2
    │  ██       lag 3
    │  █        lag 4
    │  ─        lag 5 (không significant - trong vùng xanh)
    └─────────────────────

Bar vượt ra ngoài vùng xanh → Autocorrelation có ý nghĩa thống kê
```

### 2.3. PACF (Partial Autocorrelation Function)

**PACF(k) = Correlation trực tiếp giữa y(t) và y(t-k), loại bỏ ảnh hưởng của các lags trung gian.**

**Sự khác biệt:**
```
ACF(3):  Correlation y(t) với y(t-3)
         Bao gồm: y(t) → y(t-1) → y(t-2) → y(t-3) (gián tiếp)
         
PACF(3): Correlation y(t) với y(t-3) trực tiếp
         Loại bỏ ảnh hưởng của y(t-1), y(t-2)
```

```python
from statsmodels.graphics.tsaplots import plot_pacf

# Plot PACF
fig, ax = plt.subplots(figsize=(12, 5))
plot_pacf(returns.dropna(), lags=20, ax=ax, method='ywm')
ax.set_title('Partial Autocorrelation Function (PACF)')
plt.show()
```

### 2.4. Dùng ACF/PACF để chọn Model

| Process | ACF | PACF | Ví dụ |
|---------|-----|------|-------|
| **AR(p)** | Decay dần | Cuts off sau lag p | AR(2): PACF significant ở lag 1,2, sau đó = 0 |
| **MA(q)** | Cuts off sau lag q | Decay dần | MA(2): ACF significant ở lag 1,2, sau đó = 0 |
| **ARMA** | Decay dần | Decay dần | Cả hai đều decay |

**Ví dụ thực tế:**
```python
# ACF: Significant tại lag 1, cuts off
# PACF: Decay dần
# → Gợi ý: MA(1) hoặc ARIMA(0,d,1)

# ACF: Decay dần
# PACF: Significant tại lag 1, 2, cuts off
# → Gợi ý: AR(2) hoặc ARIMA(2,d,0)
```

### 2.5. Ý nghĩa trong Trading

```
High positive autocorrelation (lag 1-5):
→ Momentum strategy có thể hiệu quả
→ Trend-following models

High negative autocorrelation:
→ Mean-reversion strategy có thể hiệu quả
→ Contrarian models

Near-zero autocorrelation:
→ Thị trường gần efficient
→ Khó dự đoán từ giá quá khứ
→ Cần thêm features khác (volume, sentiment)
```

---

## 3. LAG FEATURES VS ROLLING FEATURES

### 3.1. Lag Features

**Định nghĩa:** Giá trị của biến ở các thời điểm trước đó.

**Ví dụ:**
```
t   | close | close_lag1 | close_lag5
----|-------|------------|------------
1   | 100   | NaN        | NaN
2   | 102   | 100        | NaN
3   | 105   | 102        | NaN
4   | 103   | 105        | NaN
5   | 108   | 103        | NaN
6   | 110   | 108        | 100
7   | 112   | 110        | 102
```

**Code:**
```python
# Tạo lag features
df['close_lag1'] = df['close'].shift(1)   # Giá hôm qua
df['close_lag5'] = df['close'].shift(5)   # Giá 5 ngày trước
df['return_lag1'] = df['return'].shift(1) # Return hôm qua

# Multiple lags
for lag in [1, 2, 3, 5, 10, 20]:
    df[f'close_lag{lag}'] = df['close'].shift(lag)
```

**Khi nào dùng:**
- Khi muốn biết giá trị cụ thể tại thời điểm quá khứ
- AR models (AutoRegressive)
- Simple baseline predictions

### 3.2. Rolling Features

**Định nghĩa:** Thống kê tính trên một window di chuyển.

**Ví dụ:**
```
t   | close | ma_3 (rolling mean) | std_3 (rolling std)
----|-------|---------------------|---------------------
1   | 100   | NaN                 | NaN
2   | 102   | NaN                 | NaN
3   | 105   | 102.33              | 2.52
4   | 103   | 103.33              | 1.53
5   | 108   | 105.33              | 2.52
6   | 110   | 107.00              | 3.61
```

**Code:**
```python
# Rolling statistics
df['ma_20'] = df['close'].rolling(window=20).mean()      # Moving average 20 ngày
df['std_20'] = df['close'].rolling(window=20).std()      # Rolling std
df['min_20'] = df['close'].rolling(window=20).min()      # Rolling min
df['max_20'] = df['close'].rolling(window=20).max()      # Rolling max

# Rolling với shift (quan trọng để tránh lookahead)
df['ma_20'] = df['close'].rolling(window=20).mean().shift(1)  # Chỉ dùng data đến hôm qua
```

**Khi nào dùng:**
- Smoothing, trend detection
- Volatility estimation
- Support/Resistance levels
- Technical indicators (RSI, MACD, Bollinger Bands)

### 3.3. So sánh

| Aspect | Lag Features | Rolling Features |
|--------|--------------|------------------|
| **Ý nghĩa** | Giá trị cụ thể tại t-k | Thống kê trên window |
| **Thông tin** | Point-in-time | Aggregated |
| **Use case** | AR models, discrete signals | Trend, volatility |
| **Memory** | Ít tính toán | Cần tính toán window |
| **Ví dụ** | close_lag5 = 100 | ma_5 = 103.4 |

### 3.4. Kết hợp cả hai

```python
# Feature engineering thực tế: kết hợp cả lag và rolling

# Lag features
df['return_lag1'] = df['return'].shift(1)
df['return_lag5'] = df['return'].shift(5)

# Rolling features (shift để tránh lookahead)
df['ma_20'] = df['close'].rolling(20).mean().shift(1)
df['volatility_20'] = df['return'].rolling(20).std().shift(1)

# Derived features
df['price_vs_ma20'] = df['close'].shift(1) / df['ma_20'] - 1  # % distance from MA
df['momentum_5'] = df['close'].shift(1) / df['close'].shift(6) - 1  # 5-day momentum
```

---

## 4. TRAIN-TEST SPLIT CHO TIME SERIES

### 4.1. Tại sao không được Random Split?

```
Random Split (SAI cho time series):

Data:    [Jan][Feb][Mar][Apr][May][Jun][Jul][Aug][Sep]
Train:   [Jan][   ][Mar][   ][May][   ][Jul][   ][Sep]
Test:    [   ][Feb][   ][Apr][   ][Jun][   ][Aug][   ]

→ Model được train trên May → Test trên Apr
→ Dùng tương lai dự đoán quá khứ!
→ LOOKAHEAD BIAS
```

### 4.2. Sequential Split (Simple)

```
Sequential Split (ĐÚNG):

Data:    [Jan][Feb][Mar][Apr][May][Jun][Jul][Aug][Sep]
Train:   [Jan][Feb][Mar][Apr][May][Jun][Jul]
Test:                                      [Aug][Sep]

→ Luôn train trên quá khứ, test trên tương lai
```

**Code:**
```python
# Simple train-test split
split_date = '2023-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

# Hoặc theo tỷ lệ
split_idx = int(len(df) * 0.8)
train = df[:split_idx]
test = df[split_idx:]
```

### 4.3. Walk-Forward Validation

**Tại sao cần Walk-Forward?**
```
Single split: Chỉ test trên 1 period
→ Nếu period đó đặc biệt (COVID, crash) → Kết quả không đại diện

Walk-forward: Test trên nhiều periods
→ Robust hơn, đánh giá thực tế hơn
```

**Expanding Window:**
```
Fold 1: [████████] [test]
Fold 2: [██████████] [test]
Fold 3: [████████████] [test]
Fold 4: [██████████████] [test]

Train window tăng dần, dùng tất cả data quá khứ
```

**Rolling Window:**
```
Fold 1: [████] [test]
Fold 2:   [████] [test]
Fold 3:     [████] [test]
Fold 4:       [████] [test]

Train window cố định, slide theo thời gian
```

**Code - Expanding Window:**
```python
from sklearn.model_selection import TimeSeriesSplit

# TimeSeriesSplit = Expanding window
tscv = TimeSeriesSplit(n_splits=5)

scores = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Fold {fold+1}:")
    print(f"  Train: {len(train_idx)} samples ({df.iloc[train_idx[0]]['date']} to {df.iloc[train_idx[-1]]['date']})")
    print(f"  Test:  {len(test_idx)} samples ({df.iloc[test_idx[0]]['date']} to {df.iloc[test_idx[-1]]['date']})")
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"\nAverage score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

**Code - Rolling Window:**
```python
def rolling_window_cv(df, train_size=252, test_size=30, step=30):
    """
    Rolling window cross-validation
    
    Args:
        train_size: 252 trading days = 1 năm
        test_size: 30 days = 1 tháng
        step: 30 days giữa các folds
    """
    results = []
    start = 0
    
    while start + train_size + test_size <= len(df):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        
        # Train and evaluate
        # ...
        
        start += step
    
    return results
```

### 4.4. So sánh các Strategies

| Strategy | Ưu điểm | Nhược điểm | Khi nào dùng |
|----------|---------|------------|--------------|
| **Simple Split** | Đơn giản | Không robust | Quick evaluation |
| **Expanding Window** | Dùng tất cả data | Train size tăng dần | Default choice |
| **Rolling Window** | Train size cố định | Bỏ data cũ | Regime changes |

---

## 5. WALK-FORWARD VALIDATION (RESEARCH-GRADE)

### 5.1. Tại sao cần Walk-Forward Validation?

**Vấn đề với Single Split:**
```
Data: 2015 ──────────────────────────── 2024
           [████████████ Train ████████████][Test]
                                          2023-2024

Chỉ test trên 1 period (2023-2024)
→ Nếu period đó đặc biệt (bull run / crash) → Kết quả không đại diện
→ Model có thể overfit vào period đó
→ KHÔNG BIẾT model hoạt động thế nào ở các thời điểm khác
```

**Walk-Forward giải quyết:**
```
Fold 1: [████ Train ████][Val] 2015-2017 | 2018
Fold 2: [██████ Train ██████][Val] 2015-2018 | 2019
Fold 3: [████████ Train ████████][Val] 2015-2019 | 2020 (COVID!)
Fold 4: [██████████ Train ██████████][Val] 2015-2020 | 2021
Fold 5: [████████████ Train ████████████][Val] 2015-2021 | 2022

→ Test trên NHIỀU periods khác nhau
→ Đánh giá model trong nhiều market conditions
→ Kết quả robust và đáng tin cậy hơn
```

### 5.2. Expanding Window Validation

**Định nghĩa:** Train window tăng dần, sử dụng TẤT CẢ data quá khứ.

```
Fold 1: [████████████████] [test]
Fold 2: [██████████████████] [test]
Fold 3: [████████████████████] [test]
Fold 4: [██████████████████████] [test]
Fold 5: [████████████████████████] [test]

→ Mỗi fold dùng nhiều data hơn fold trước
→ Giả định: Data cũ vẫn hữu ích
```

**Implementation đầy đủ:**

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Tuple, Dict, Any

class ExpandingWindowCV:
    """
    Expanding Window Cross-Validation cho Time Series
    
    Đặc điểm:
    - Train size tăng dần qua mỗi fold
    - Test size cố định
    - Có thể có gap (embargo) giữa train và test
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: int = 252,  # 1 năm trading
                 gap: int = 0,          # Embargo period
                 min_train_size: int = 504):  # Minimum 2 năm train
        """
        Args:
            n_splits: Số folds
            test_size: Kích thước test set (số observations)
            gap: Khoảng cách giữa train và test (để tránh leakage)
            min_train_size: Kích thước tối thiểu của train set
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.min_train_size = min_train_size
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices cho mỗi fold
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        
        # Tính kích thước cần thiết
        total_test_size = self.n_splits * self.test_size
        available_for_last_train = n_samples - self.test_size - self.gap
        
        if available_for_last_train < self.min_train_size:
            raise ValueError(f"Not enough data. Need at least {self.min_train_size + self.test_size + self.gap} samples")
        
        splits = []
        
        for fold in range(self.n_splits):
            # Test end position (từ cuối data ngược lại)
            test_end = n_samples - fold * self.test_size
            test_start = test_end - self.test_size
            
            # Train end (có gap)
            train_end = test_start - self.gap
            train_start = 0  # Expanding: luôn bắt đầu từ đầu
            
            if train_end - train_start < self.min_train_size:
                continue
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        # Reverse để fold 1 là earliest
        return splits[::-1]
    
    def get_fold_info(self, X: pd.DataFrame, dates: pd.Series = None) -> pd.DataFrame:
        """
        Get detailed info về mỗi fold
        """
        splits = self.split(X)
        info = []
        
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            fold_info = {
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_start_idx': train_idx[0],
                'train_end_idx': train_idx[-1],
                'test_start_idx': test_idx[0],
                'test_end_idx': test_idx[-1],
            }
            
            if dates is not None:
                fold_info.update({
                    'train_start_date': dates.iloc[train_idx[0]],
                    'train_end_date': dates.iloc[train_idx[-1]],
                    'test_start_date': dates.iloc[test_idx[0]],
                    'test_end_date': dates.iloc[test_idx[-1]],
                })
            
            info.append(fold_info)
        
        return pd.DataFrame(info)


# Usage
cv = ExpandingWindowCV(n_splits=5, test_size=252, gap=5, min_train_size=504)

# Xem thông tin các folds
fold_info = cv.get_fold_info(X, df['date'])
print(fold_info.to_string(index=False))

# Cross-validation loop
results = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'fold': fold,
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'MAE': mae,
        'RMSE': rmse
    })
    
    print(f"Fold {fold}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

results_df = pd.DataFrame(results)
print(f"\n=== SUMMARY ===")
print(f"Mean MAE: {results_df['MAE'].mean():.4f} ± {results_df['MAE'].std():.4f}")
print(f"Mean RMSE: {results_df['RMSE'].mean():.4f} ± {results_df['RMSE'].std():.4f}")
```

### 5.3. Rolling Window Validation (Sliding Window)

**Định nghĩa:** Train window CỐ ĐỊNH, slide theo thời gian.

```
Fold 1: [████████] [test]
Fold 2:   [████████] [test]
Fold 3:     [████████] [test]
Fold 4:       [████████] [test]
Fold 5:         [████████] [test]

→ Train size luôn bằng nhau
→ Data cũ bị DROP
→ Phù hợp khi market dynamics thay đổi (regime changes)
```

**Khi nào dùng Rolling thay vì Expanding?**

| Scenario | Nên dùng | Lý do |
|----------|----------|-------|
| Market ổn định | Expanding | Nhiều data = better |
| Regime changes | Rolling | Data cũ có thể misleading |
| Concept drift | Rolling | Model cần adapt |
| Research paper | Expanding | Standard trong academia |
| Trading system | Rolling | Practical, adaptive |

**Implementation:**

```python
class RollingWindowCV:
    """
    Rolling Window Cross-Validation cho Time Series
    
    Đặc điểm:
    - Train size CỐ ĐỊNH
    - Window slides theo thời gian
    - Data cũ bị drop
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 train_size: int = 504,   # 2 năm
                 test_size: int = 63,     # 3 tháng
                 step_size: int = 63,     # Step giữa các folds
                 gap: int = 0):           # Embargo
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.gap = gap
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        splits = []
        
        for fold in range(self.n_splits):
            # Train start position
            train_start = fold * self.step_size
            train_end = train_start + self.train_size
            
            # Test positions (với gap)
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits


# Usage
cv = RollingWindowCV(
    n_splits=10,
    train_size=504,   # 2 năm
    test_size=63,     # 3 tháng
    step_size=63,     # Roll mỗi 3 tháng
    gap=5             # 5 ngày embargo
)

for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
    print(f"Fold {fold}: Train {len(train_idx)} samples, Test {len(test_idx)} samples")
```

### 5.4. Purged Cross-Validation (Chống Leakage)

**Vấn đề:** Trong financial data, observation tại t có thể overlap với t+1, t+2, ...

```
Ví dụ: 5-day return
Day 1: return[1:6]   = [d1, d2, d3, d4, d5]
Day 2: return[2:7]   = [d2, d3, d4, d5, d6]
Day 3: return[3:8]   = [d3, d4, d5, d6, d7]

→ Day 1 và Day 2 share d2, d3, d4, d5!
→ Nếu Day 1 trong train, Day 2 trong test → LEAKAGE!
```

**Purged CV: Loại bỏ overlap**

```python
class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation
    
    Loại bỏ samples trong train set mà overlap với test set
    Thêm embargo period sau train để tránh leakage
    
    Reference: "Advances in Financial Machine Learning" - Marcos López de Prado
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 purge_gap: int = 5,     # Loại bỏ samples overlap
                 embargo_pct: float = 0.01):  # % data làm embargo
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(self, X: pd.DataFrame, 
              t1: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Args:
            X: Features DataFrame
            t1: Series với end time của mỗi observation
                Nếu None, assume no overlap
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Tính embargo size
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Chia thành n_splits theo thời gian
        test_size = n_samples // self.n_splits
        
        splits = []
        
        for fold in range(self.n_splits):
            test_start = fold * test_size
            test_end = min((fold + 1) * test_size, n_samples)
            
            test_indices = indices[test_start:test_end]
            
            # Train indices: tất cả trừ test + purge + embargo
            train_mask = np.ones(n_samples, dtype=bool)
            
            # Remove test
            train_mask[test_start:test_end] = False
            
            # Purge: Remove samples trước test có thể overlap
            purge_start = max(0, test_start - self.purge_gap)
            train_mask[purge_start:test_start] = False
            
            # Embargo: Remove samples ngay sau test
            embargo_end = min(n_samples, test_end + embargo_size)
            train_mask[test_end:embargo_end] = False
            
            # Chỉ lấy train TRƯỚC test (time series constraint)
            train_mask[test_end:] = False
            
            train_indices = indices[train_mask]
            
            if len(train_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits


# Usage với overlapping returns
cv = PurgedKFoldCV(n_splits=5, purge_gap=5, embargo_pct=0.01)

for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
    print(f"Fold {fold}:")
    print(f"  Train: {len(train_idx)} samples [{train_idx[0]} to {train_idx[-1]}]")
    print(f"  Test:  {len(test_idx)} samples [{test_idx[0]} to {test_idx[-1]}]")
    print(f"  Gap between train and test: {test_idx[0] - train_idx[-1] - 1} samples")
```

### 5.5. Combinatorial Purged Cross-Validation (CPCV)

**Ý tưởng:** Tạo nhiều train/test combinations hơn standard CV.

```
Standard 5-fold: 5 combinations
CPCV với n=5, k=2: C(5,2) = 10 combinations

→ Nhiều evaluation points hơn
→ Giảm variance của performance estimate
```

```python
from itertools import combinations

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation
    
    Tạo tất cả combinations có thể của test folds
    Mỗi combination có k folds làm test, n-k folds làm train
    
    Reference: López de Prado (2018)
    """
    
    def __init__(self,
                 n_groups: int = 6,      # Số groups chia data
                 n_test_groups: int = 2,  # Số groups làm test mỗi lần
                 purge_gap: int = 5,
                 embargo_pct: float = 0.01):
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Chia data thành n_groups
        group_size = n_samples // self.n_groups
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_groups - 1 else n_samples
            groups.append(indices[start:end])
        
        # Tạo tất cả combinations
        test_combinations = list(combinations(range(self.n_groups), self.n_test_groups))
        
        embargo_size = int(n_samples * self.embargo_pct)
        splits = []
        
        for test_group_indices in test_combinations:
            # Test indices
            test_indices = np.concatenate([groups[i] for i in test_group_indices])
            test_indices.sort()
            
            # Train mask
            train_mask = np.ones(n_samples, dtype=bool)
            
            # Remove test groups
            train_mask[test_indices] = False
            
            # Purge và embargo cho mỗi test group
            for group_idx in test_group_indices:
                group = groups[group_idx]
                test_start = group[0]
                test_end = group[-1]
                
                # Purge before
                purge_start = max(0, test_start - self.purge_gap)
                train_mask[purge_start:test_start] = False
                
                # Embargo after
                embargo_end = min(n_samples, test_end + embargo_size + 1)
                train_mask[test_end + 1:embargo_end] = False
            
            train_indices = indices[train_mask]
            
            # Chỉ giữ train TRƯỚC test (earliest test group)
            earliest_test = min(test_indices)
            train_indices = train_indices[train_indices < earliest_test]
            
            if len(train_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits

# Usage
cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
n_splits = len(cv.split(X))
print(f"Number of train/test combinations: {n_splits}")  # C(6,2) = 15
```

### 5.6. Aggregation Metrics across Folds

**Vấn đề:** Làm sao tổng hợp kết quả từ nhiều folds?

```python
def aggregate_cv_results(fold_results: List[Dict]) -> Dict:
    """
    Aggregate metrics từ multiple folds
    
    Args:
        fold_results: List of dicts với metrics per fold
    
    Returns:
        Dict với aggregated metrics
    """
    df = pd.DataFrame(fold_results)
    
    metrics = ['MAE', 'RMSE', 'MAPE', 'DirectionalAccuracy']
    agg_results = {}
    
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            
            agg_results[f'{metric}_mean'] = values.mean()
            agg_results[f'{metric}_std'] = values.std()
            agg_results[f'{metric}_min'] = values.min()
            agg_results[f'{metric}_max'] = values.max()
            agg_results[f'{metric}_median'] = values.median()
            
            # Confidence interval (95%)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            agg_results[f'{metric}_ci_95'] = (ci_lower, ci_upper)
    
    # Stability metrics
    if 'MAE' in df.columns:
        agg_results['stability_coefficient'] = df['MAE'].std() / df['MAE'].mean()
        agg_results['worst_fold_ratio'] = df['MAE'].max() / df['MAE'].min()
    
    return agg_results


def print_cv_summary(agg_results: Dict):
    """Pretty print CV summary"""
    print("\n" + "="*60)
    print("WALK-FORWARD CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    for metric in ['MAE', 'RMSE']:
        if f'{metric}_mean' in agg_results:
            mean = agg_results[f'{metric}_mean']
            std = agg_results[f'{metric}_std']
            ci = agg_results.get(f'{metric}_ci_95', (None, None))
            
            print(f"\n{metric}:")
            print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}")
            print(f"  Range: [{agg_results[f'{metric}_min']:.4f}, {agg_results[f'{metric}_max']:.4f}]")
            if ci[0]:
                print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    if 'stability_coefficient' in agg_results:
        print(f"\nStability Metrics:")
        print(f"  Coefficient of Variation: {agg_results['stability_coefficient']:.4f}")
        print(f"  Worst/Best Fold Ratio: {agg_results['worst_fold_ratio']:.2f}")
        
        if agg_results['stability_coefficient'] < 0.2:
            print("  → Model STABLE across folds")
        else:
            print("  → Model UNSTABLE - high variance across folds")

# Usage
agg = aggregate_cv_results(results)
print_cv_summary(agg)
```

### 5.7. Best Practices cho Walk-Forward Validation

**1. Chọn test_size phù hợp:**
```python
# Quarterly evaluation
test_size = 63  # ~3 tháng trading days

# Annual evaluation  
test_size = 252  # 1 năm

# Monthly evaluation (nhiều folds hơn)
test_size = 21  # 1 tháng
```

**2. Luôn dùng embargo/purge:**
```python
# Minimum embargo = horizon của prediction
# Nếu predict 5-day return → embargo >= 5

cv = ExpandingWindowCV(
    n_splits=5,
    test_size=252,
    gap=5  # Embargo 5 days
)
```

**3. Report đầy đủ metrics:**
```python
# Không chỉ report mean, mà cả:
# - Standard deviation (stability)
# - Min/Max (worst/best case)
# - Confidence interval

print(f"MAE: {mean:.4f} ± {std:.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
```

**4. Visualize performance qua thời gian:**
```python
import matplotlib.pyplot as plt

def plot_cv_performance(fold_results: List[Dict], dates: List):
    """Visualize CV performance across folds"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    folds = [r['fold'] for r in fold_results]
    maes = [r['MAE'] for r in fold_results]
    
    # MAE per fold
    axes[0].bar(folds, maes)
    axes[0].axhline(np.mean(maes), color='red', linestyle='--', label=f'Mean: {np.mean(maes):.4f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('MAE across Walk-Forward Folds')
    axes[0].legend()
    
    # Cumulative performance
    cumulative_mae = np.cumsum(maes) / np.arange(1, len(maes) + 1)
    axes[1].plot(folds, cumulative_mae, marker='o')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Cumulative MAE')
    axes[1].set_title('Cumulative Average MAE')
    
    plt.tight_layout()
    plt.show()
```

---

## 6. LOOKAHEAD BIAS

### 5.1. Lookahead Bias là gì?

**Định nghĩa:** Model "nhìn thấy" thông tin từ tương lai trong quá trình training hoặc feature engineering.

**Hậu quả:** Backtest performance tốt nhưng live trading thất bại hoàn toàn.

### 5.2. Các dạng Lookahead Bias

**1. Feature Engineering Leak:**
```python
# SAI: Rolling mean với center=True
df['ma_20'] = df['close'].rolling(20, center=True).mean()
# center=True dùng 10 ngày trước VÀ 10 ngày sau!

# ĐÚNG: Chỉ dùng data quá khứ
df['ma_20'] = df['close'].rolling(20).mean().shift(1)
```

**2. Scaling/Normalization Leak:**
```python
# SAI: Fit scaler trên toàn bộ data
scaler = StandardScaler()
df['close_scaled'] = scaler.fit_transform(df[['close']])
# mean, std tính từ cả test data!

# ĐÚNG: Chỉ fit trên train
scaler.fit(train[['close']])
train['close_scaled'] = scaler.transform(train[['close']])
test['close_scaled'] = scaler.transform(test[['close']])
```

**3. Target Encoding Leak:**
```python
# SAI: Target encoding dùng toàn bộ data
sector_returns = df.groupby('sector')['return'].mean()
df['sector_encoded'] = df['sector'].map(sector_returns)
# Bao gồm returns từ tương lai!

# ĐÚNG: Chỉ dùng train data
sector_returns = train.groupby('sector')['return'].mean()
train['sector_encoded'] = train['sector'].map(sector_returns)
test['sector_encoded'] = test['sector'].map(sector_returns)
```

**4. Hyperparameter Tuning Leak:**
```python
# SAI: Tune hyperparameters trên test set
for lr in [0.001, 0.01, 0.1]:
    model.fit(X_train, y_train)
    test_score = evaluate(X_test, y_test)  # Leak!
    if test_score > best:
        best_lr = lr

# ĐÚNG: Dùng validation set
for lr in [0.001, 0.01, 0.1]:
    model.fit(X_train, y_train)
    val_score = evaluate(X_val, y_val)  # Val, không phải Test
    if val_score > best:
        best_lr = lr

# Test chỉ dùng 1 lần cuối cùng
final_score = evaluate(X_test, y_test)
```

**5. Point-in-Time Data:**
```python
# SAI: Dùng financial data không point-in-time
# Ví dụ: EPS Q1/2023 công bố tháng 4/2023
# Nhưng dùng cho prediction tháng 1/2023!

# ĐÚNG: Chỉ dùng data đã available tại thời điểm đó
# EPS Q4/2022 (công bố tháng 1/2023) cho prediction tháng 2/2023
```

### 5.3. Checklist tránh Lookahead Bias

```
□ Không dùng center=True trong rolling
□ Shift tất cả features ít nhất 1 period
□ Fit scaler/encoder chỉ trên train
□ Không tune hyperparameters trên test
□ Point-in-time data cho fundamentals
□ Split data theo thời gian, không random
□ Validation set riêng với test set
```

### 5.4. Test để phát hiện Lookahead

```python
def check_for_lookahead(df, feature_col, target_col):
    """
    Kiểm tra xem feature có lookahead không
    
    Nếu correlation quá cao (>0.9) → Có thể có lookahead!
    """
    corr = df[feature_col].corr(df[target_col])
    
    if abs(corr) > 0.9:
        print(f"WARNING: {feature_col} has correlation {corr:.3f} with target")
        print("This may indicate lookahead bias!")
    
    return corr

# Test
for col in feature_columns:
    check_for_lookahead(df, col, 'return_1d')
```

---

## 7. STATIONARITY

### 6.1. Stationarity là gì?

**Định nghĩa:** Tính chất thống kê không đổi theo thời gian.

**Stationary series có:**
- Mean (trung bình) không đổi
- Variance (phương sai) không đổi
- Autocovariance chỉ phụ thuộc lag, không phụ thuộc thời điểm

### 6.2. Tại sao Stationarity quan trọng?

**Về mặt toán học:**
```
Model ARIMA giả định: E[y(t)] = μ (constant)

Nếu non-stationary:
- E[y(t)] = f(t) (thay đổi theo t)
- Model parameters không ổn định
- Predictions không reliable

Ví dụ:
- Train: 2015-2020, mean price = 50
- Test: 2021-2023, mean price = 100
- Model trained với mean=50 sẽ sai hoàn toàn!
```

**Về mặt thực tế:**
```
Non-stationary (Price):
2015: Mean = 30K
2020: Mean = 70K
2024: Mean = 100K
→ Không thể dùng past patterns để predict future

Stationary (Returns):
2015: Mean ≈ 0.1%
2020: Mean ≈ 0.08%
2024: Mean ≈ 0.12%
→ Past patterns có thể giúp predict future
```

### 6.3. Kiểm tra Stationarity

**Visual Test:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price (non-stationary)
axes[0, 0].plot(df['close'])
axes[0, 0].set_title('Price (Non-Stationary)')

# Returns (stationary)
axes[0, 1].plot(df['return'])
axes[0, 1].set_title('Returns (Stationary)')

# Rolling statistics for price
rolling_mean = df['close'].rolling(252).mean()
rolling_std = df['close'].rolling(252).std()
axes[1, 0].plot(df['close'], label='Price')
axes[1, 0].plot(rolling_mean, label='Rolling Mean')
axes[1, 0].legend()
axes[1, 0].set_title('Price with Rolling Mean (increasing)')

# Rolling statistics for returns
rolling_mean_ret = df['return'].rolling(252).mean()
rolling_std_ret = df['return'].rolling(252).std()
axes[1, 1].plot(rolling_mean_ret, label='Rolling Mean')
axes[1, 1].plot(rolling_std_ret, label='Rolling Std')
axes[1, 1].legend()
axes[1, 1].set_title('Returns Rolling Stats (stable)')

plt.tight_layout()
plt.show()
```

**ADF Test (Augmented Dickey-Fuller):**
```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, name=''):
    """
    ADF Test:
    H0: Series is non-stationary (has unit root)
    H1: Series is stationary
    
    p-value < 0.05 → Reject H0 → Stationary
    p-value > 0.05 → Accept H0 → Non-stationary
    """
    result = adfuller(series.dropna())
    
    print(f"=== {name} ===")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print("→ STATIONARY (p < 0.05)")
    else:
        print("→ NON-STATIONARY (p > 0.05)")
    
    return result[1] < 0.05

# Test
check_stationarity(df['close'], 'Price')
check_stationarity(df['return'], 'Returns')
```

**KPSS Test (alternative):**
```python
from statsmodels.tsa.stattools import kpss

def kpss_test(series, name=''):
    """
    KPSS Test:
    H0: Series is stationary
    H1: Series is non-stationary
    
    p-value < 0.05 → Reject H0 → Non-stationary
    p-value > 0.05 → Accept H0 → Stationary
    
    (Ngược với ADF!)
    """
    result = kpss(series.dropna())
    
    print(f"=== {name} ===")
    print(f"KPSS Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    
    if result[1] < 0.05:
        print("→ NON-STATIONARY")
    else:
        print("→ STATIONARY")
```

---

## 8. DIFFERENCING VÀ TRANSFORMATIONS

### 7.1. First Differencing

**Ý tưởng:** Chuyển từ levels sang changes.

```
y'(t) = y(t) - y(t-1)

Price:   [100, 102, 105, 103, 108]
Diff:    [NaN,  2,   3,  -2,   5]
```

**Code:**
```python
# First difference
df['close_diff'] = df['close'].diff()

# Check stationarity after differencing
check_stationarity(df['close_diff'], 'Price after 1st diff')
```

**Khi nào cần:**
- Price levels (almost always need differencing)
- Series có trend

### 7.2. Second Differencing

**Khi first differencing không đủ:**
```
y''(t) = y'(t) - y'(t-1)
       = [y(t) - y(t-1)] - [y(t-1) - y(t-2)]
       = y(t) - 2*y(t-1) + y(t-2)
```

```python
# Second difference
df['close_diff2'] = df['close'].diff().diff()

# Thường không cần diff 2 lần cho stock prices
# Returns (first diff of log prices) thường đã stationary
```

### 7.3. Log Transform

**Tại sao dùng Log?**
```
1. Giảm heteroskedasticity (variance thay đổi theo level)
2. Chuyển multiplicative relationships → additive
3. Log returns có interpretation tốt hơn

Price 100 → 110: +10% 
Price 1000 → 1100: +10%

Với log:
log(110) - log(100) ≈ 0.095
log(1100) - log(1000) ≈ 0.095
→ Comparable!
```

```python
import numpy as np

# Log transform
df['log_price'] = np.log(df['close'])

# Log returns (preferred in finance)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
# Equivalent to: np.log(df['close']).diff()
```

### 7.4. Các Transformations khác

**Box-Cox Transform:**
```python
from scipy.stats import boxcox

# Tự động tìm lambda tối ưu
df['close_boxcox'], lambda_param = boxcox(df['close'])
print(f"Optimal lambda: {lambda_param:.4f}")
```

**Seasonal Differencing:**
```python
# Remove yearly seasonality (252 trading days)
df['close_seasonal_diff'] = df['close'] - df['close'].shift(252)
```

### 7.5. Workflow để đạt Stationarity

```python
def make_stationary(series, max_diff=2):
    """
    Tự động differencing cho đến khi stationary
    """
    d = 0
    current = series.copy()
    
    while d < max_diff:
        result = adfuller(current.dropna())
        if result[1] < 0.05:
            print(f"Stationary after {d} differencing(s)")
            return current, d
        
        current = current.diff()
        d += 1
    
    print(f"Warning: Not stationary after {max_diff} differencing(s)")
    return current, d

# Usage
stationary_series, d = make_stationary(df['close'])
print(f"d = {d} for ARIMA(p, {d}, q)")
```

---

## 9. MEAN-REVERSION VS MOMENTUM

### 8.1. Mean-Reversion

**Định nghĩa:** Giá có xu hướng quay về mean sau khi di chuyển xa.

```
Price
  │
  │     ╱╲
  │    ╱  ╲    ╱╲
  │───╱────╲──╱──╲───── Mean
  │          ╲╱
  │
  └────────────────────→ Time

Giá cao hơn mean → Sẽ giảm
Giá thấp hơn mean → Sẽ tăng
```

**Đặc điểm:**
- Negative autocorrelation
- Half-life: Thời gian để giá quay về mean
- Bollinger Bands, RSI overbought/oversold

**Trading Strategy:**
```python
# Mean-reversion strategy
df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()

df['signal'] = 0
df.loc[df['z_score'] > 2, 'signal'] = -1   # Sell when too high
df.loc[df['z_score'] < -2, 'signal'] = 1   # Buy when too low
```

### 8.2. Momentum

**Định nghĩa:** Giá có xu hướng tiếp tục theo hướng đang di chuyển.

```
Price
  │
  │            ╱╱
  │          ╱╱
  │        ╱╱
  │      ╱╱
  │    ╱╱
  │  ╱╱
  └────────────────────→ Time

Giá đang tăng → Tiếp tục tăng
Giá đang giảm → Tiếp tục giảm
```

**Đặc điểm:**
- Positive autocorrelation
- Trend-following
- Moving average crossover, breakout

**Trading Strategy:**
```python
# Momentum strategy
df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

df['signal'] = 0
df.loc[df['momentum_20'] > 0.05, 'signal'] = 1   # Buy when momentum up
df.loc[df['momentum_20'] < -0.05, 'signal'] = -1  # Sell when momentum down
```

### 8.3. Regime Detection

**Thực tế:** Market thay đổi giữa các regimes.

```
2019-2020: Mean-reversion regime (range-bound)
2020-2021: Momentum regime (strong bull)
2022: Mean-reversion regime (consolidation)
2023: Momentum regime (recovery)
```

**Code để detect regime:**
```python
def detect_regime(returns, window=60):
    """
    Detect market regime based on autocorrelation
    
    Positive autocorr → Momentum regime
    Negative autocorr → Mean-reversion regime
    """
    # Rolling autocorrelation lag-1
    def rolling_autocorr(x):
        return x.autocorr(lag=1)
    
    autocorr = returns.rolling(window).apply(rolling_autocorr)
    
    regime = pd.Series(index=returns.index, dtype='object')
    regime[autocorr > 0.1] = 'Momentum'
    regime[autocorr < -0.1] = 'Mean-Reversion'
    regime[(autocorr >= -0.1) & (autocorr <= 0.1)] = 'Random'
    
    return regime, autocorr

regime, autocorr = detect_regime(df['return'])
print(regime.value_counts())
```

### 8.4. So sánh Strategies

| Aspect | Mean-Reversion | Momentum |
|--------|----------------|----------|
| **Autocorrelation** | Negative | Positive |
| **Strategy** | Buy low, sell high | Buy winners, sell losers |
| **Indicators** | RSI, Bollinger | MA crossover, Breakout |
| **Time horizon** | Short-term | Medium to long-term |
| **Risk** | Gap risk, trend | Reversal risk |

---

## 10. FORECASTING METRICS

### 9.1. MAE (Mean Absolute Error)

**Công thức:**
```
MAE = (1/n) × Σ|y_true - y_pred|
```

**Ví dụ:**
```
y_true = [100, 105, 102]
y_pred = [98,  107, 100]
error  = [2,   2,   2]
MAE = (2 + 2 + 2) / 3 = 2
```

**Đặc điểm:**
- Đơn vị giống với y
- Không phạt nặng outliers
- Dễ interpret: "Trung bình sai 2 đơn vị"

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")
```

### 9.2. RMSE (Root Mean Squared Error)

**Công thức:**
```
RMSE = √[(1/n) × Σ(y_true - y_pred)²]
```

**Ví dụ:**
```
y_true = [100, 105, 102]
y_pred = [98,  107, 100]
error² = [4,   4,   4]
MSE = (4 + 4 + 4) / 3 = 4
RMSE = √4 = 2
```

**Đặc điểm:**
- Đơn vị giống với y
- Phạt nặng outliers (vì bình phương)
- Sensitive to large errors

```python
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.2f}")
```

### 9.3. MAPE (Mean Absolute Percentage Error)

**Công thức:**
```
MAPE = (1/n) × Σ|((y_true - y_pred) / y_true)| × 100%
```

**Ví dụ:**
```
y_true = [100, 105, 102]
y_pred = [98,  107, 100]
error% = [2%,  1.9%, 2%]
MAPE = (2 + 1.9 + 2) / 3 = 1.97%
```

**Đặc điểm:**
- Scale-independent (%)
- Dễ so sánh giữa các series khác nhau
- Vấn đề: Undefined khi y_true = 0

```python
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAPE: {mape(y_true, y_pred):.2f}%")
```

### 9.4. So sánh Metrics

| Metric | Ưu điểm | Nhược điểm | Khi nào dùng |
|--------|---------|------------|--------------|
| **MAE** | Robust to outliers, dễ hiểu | Không phạt nặng lỗi lớn | Default choice |
| **RMSE** | Phạt lỗi lớn | Sensitive to outliers | Khi lỗi lớn quan trọng |
| **MAPE** | Scale-free | Undefined when y=0, asymmetric | So sánh nhiều series |

### 9.5. Metrics cho Trading

**Ngoài accuracy, cần xét:**
```python
# Directional Accuracy
correct_direction = np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])
directional_accuracy = correct_direction.mean()
print(f"Directional Accuracy: {directional_accuracy:.2%}")

# Sharpe Ratio của strategy
returns = calculate_returns(y_pred)
sharpe = returns.mean() / returns.std() * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe:.2f}")
```

---

## 11. MULTI-STEP FORECASTING

### 10.1. One-Step vs Multi-Step

**One-Step Forecast:**
```
Input:  [t-10, t-9, ..., t-1, t]
Output: t+1

Dự đoán: Ngày mai duy nhất
```

**Multi-Step Forecast:**
```
Input:  [t-10, t-9, ..., t-1, t]
Output: [t+1, t+2, t+3, t+4, t+5]

Dự đoán: 5 ngày tiếp theo
```

### 10.2. Recursive Strategy

**Ý tưởng:** Train 1 model, dùng predictions làm input cho step tiếp theo.

```
Step 1: f([t-10,...,t]) → ŷ(t+1)
Step 2: f([t-9,...,t, ŷ(t+1)]) → ŷ(t+2)
Step 3: f([t-8,...,ŷ(t+1), ŷ(t+2)]) → ŷ(t+3)
...
```

**Code:**
```python
def recursive_forecast(model, X_last, steps=5):
    """
    Recursive multi-step forecasting
    
    Args:
        model: Trained model
        X_last: Last known feature vector
        steps: Number of steps to forecast
    """
    forecasts = []
    X_current = X_last.copy()
    
    for step in range(steps):
        # Predict next step
        y_pred = model.predict(X_current.reshape(1, -1))[0]
        forecasts.append(y_pred)
        
        # Shift features and add prediction
        X_current = np.roll(X_current, -1)
        X_current[-1] = y_pred
    
    return forecasts

# Usage
last_features = X_test.iloc[-1].values
forecasts = recursive_forecast(model, last_features, steps=5)
```

**Ưu điểm:**
- Chỉ cần train 1 model
- Có thể forecast arbitrary horizon

**Nhược điểm:**
- Error accumulation (lỗi tích lũy)
- Horizon càng xa, error càng lớn

### 10.3. Direct Strategy

**Ý tưởng:** Train model riêng cho mỗi horizon.

```
Model 1: f₁([t-10,...,t]) → ŷ(t+1)
Model 2: f₂([t-10,...,t]) → ŷ(t+2)
Model 3: f₃([t-10,...,t]) → ŷ(t+3)
...
```

**Code:**
```python
def direct_forecast(X_train, y_series, X_test, max_horizon=5):
    """
    Direct multi-step forecasting
    
    Train separate model for each horizon
    """
    from sklearn.linear_model import Ridge
    
    models = {}
    forecasts = {}
    
    for h in range(1, max_horizon + 1):
        # Create target for horizon h
        y_h = y_series.shift(-h)
        
        # Align and remove NaN
        valid_idx = ~y_h.isna()
        X_train_h = X_train[valid_idx]
        y_train_h = y_h[valid_idx]
        
        # Train model for horizon h
        model = Ridge()
        model.fit(X_train_h, y_train_h)
        models[h] = model
        
        # Forecast
        forecasts[h] = model.predict(X_test)[-1]
        
        print(f"Horizon {h}: Forecast = {forecasts[h]:.2f}")
    
    return models, forecasts

models, forecasts = direct_forecast(X_train, df['close'], X_test, max_horizon=5)
```

**Ưu điểm:**
- Mỗi horizon được optimize riêng
- Không có error accumulation

**Nhược điểm:**
- Cần train nhiều models
- Không leverage relationships giữa các horizons

### 10.4. Multi-Output Strategy

**Ý tưởng:** Train 1 model với multiple outputs.

```
Model: f([t-10,...,t]) → [ŷ(t+1), ŷ(t+2), ŷ(t+3), ŷ(t+4), ŷ(t+5)]
```

**Code với Neural Network:**
```python
import torch
import torch.nn as nn

class MultiStepModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_steps):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_steps)  # Multi-output
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output: [batch, n_steps]
        return x

# Training
model = MultiStepModel(input_dim=10, hidden_dim=64, n_steps=5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# y_train shape: [batch, 5] - 5 days forecast
```

### 10.5. So sánh Strategies

| Strategy | Số models | Error accumulation | Complexity | Khi nào dùng |
|----------|-----------|-------------------|------------|--------------|
| **Recursive** | 1 | Có | Thấp | Prototype, short horizon |
| **Direct** | H models | Không | Trung bình | Production, accuracy matters |
| **Multi-Output** | 1 | Không | Cao | Deep learning, nhiều data |

### 10.6. Best Practices

```python
# 1. Start với Recursive (simple baseline)
recursive_forecasts = recursive_forecast(model, X_last, steps=5)

# 2. Compare với Direct
direct_forecasts = direct_forecast(X_train, y, X_test, max_horizon=5)

# 3. Evaluate per horizon
for h in range(1, 6):
    print(f"Horizon {h}:")
    print(f"  Recursive MAE: {mae(y_true[h], recursive_forecasts[h-1]):.4f}")
    print(f"  Direct MAE: {mae(y_true[h], direct_forecasts[h]):.4f}")

# 4. Use Direct for production if accuracy > speed
```

---

## 12. BÀI TẬP THỰC HÀNH

### Bài tập 1: Phân tích Autocorrelation

**Yêu cầu:**
1. Load data FPT
2. Tính returns
3. Vẽ ACF và PACF
4. Xác định: Market có momentum hay mean-reversion?

```python
# Gợi ý
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load data
df = pd.read_csv('data/features/vn30/FPT.csv')
df['return'] = df['close'].pct_change()

# Plot ACF và PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['return'].dropna(), lags=20, ax=axes[0])
plot_pacf(df['return'].dropna(), lags=20, ax=axes[1])
plt.show()

# Phân tích và kết luận
```

### Bài tập 2: Kiểm tra Stationarity

**Yêu cầu:**
1. Test stationarity cho Price và Returns
2. Nếu non-stationary, transform để đạt stationarity
3. Xác định d cho ARIMA

```python
# Gợi ý
from statsmodels.tsa.stattools import adfuller

# Test price
result_price = adfuller(df['close'].dropna())
print(f"Price p-value: {result_price[1]:.4f}")

# Test returns
result_returns = adfuller(df['return'].dropna())
print(f"Returns p-value: {result_returns[1]:.4f}")

# Kết luận d = ?
```

### Bài tập 3: Walk-Forward Validation

**Yêu cầu:**
1. Implement expanding window với 5 folds
2. Train Linear Regression trên mỗi fold
3. Tính average MAE và std

```python
# Gợi ý
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

tscv = TimeSeriesSplit(n_splits=5)
mae_scores = []

for train_idx, test_idx in tscv.split(X):
    # Train and evaluate
    # TODO: Implement
    pass

print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
```

### Bài tập 4: Multi-Step Forecasting

**Yêu cầu:**
1. Implement Recursive forecasting (5 days)
2. Implement Direct forecasting (5 days)
3. So sánh MAE per horizon

```python
# Gợi ý
# Recursive
recursive_preds = recursive_forecast(model, X_last, steps=5)

# Direct
for h in range(1, 6):
    y_h = df['close'].shift(-h)
    # Train model_h
    # Predict

# Compare
```

### Bài tập 5: Lookahead Bias Detection

**Yêu cầu:**
1. Tạo feature có lookahead bias (có ý)
2. Kiểm tra correlation bất thường
3. Fix lookahead bias

```python
# Gợi ý
# SAI
df['ma_20_wrong'] = df['close'].rolling(20, center=True).mean()

# ĐÚNG  
df['ma_20_correct'] = df['close'].rolling(20).mean().shift(1)

# So sánh correlation với target
```

---

## Kiểm tra hiểu bài

**Phần Autocorrelation:**
- [ ] Giải thích được ACF vs PACF
- [ ] Đọc được ACF/PACF để chọn AR/MA
- [ ] Hiểu ý nghĩa của autocorrelation trong trading

**Phần Features:**
- [ ] Phân biệt được lag vs rolling features
- [ ] Biết khi nào dùng loại nào

**Phần Validation:**
- [ ] Implement được walk-forward validation
- [ ] Hiểu tại sao không được random split

**Phần Lookahead:**
- [ ] Liệt kê được các dạng lookahead bias
- [ ] Biết cách phát hiện và tránh

**Phần Stationarity:**
- [ ] Giải thích được tại sao stationarity quan trọng
- [ ] Kiểm tra và transform được non-stationary series

**Phần Regimes:**
- [ ] Phân biệt được mean-reversion vs momentum
- [ ] Hiểu regime changes

**Phần Metrics:**
- [ ] Tính được MAE, RMSE, MAPE
- [ ] Biết khi nào dùng metric nào

**Phần Multi-Step:**
- [ ] Implement được recursive và direct strategies
- [ ] So sánh và chọn strategy phù hợp

---

## Tài liệu tham khảo

**Books:**
- "Forecasting: Principles and Practice" (3rd ed) - Rob Hyndman & George Athanasopoulos (FREE: https://otexts.com/fpp3/)
- "Time Series Analysis and Its Applications" - Shumway & Stoffer

**Python Libraries:**
- `statsmodels`: ARIMA, ACF/PACF, ADF test
- `pmdarima`: Auto ARIMA
- `sktime`: Sklearn-compatible time series

---

## Bước tiếp theo

Sau khi hoàn thành bài này, sang:
- `02_modeling/01_BASELINE_MODELS.md` - ARIMA, GARCH
- `02_modeling/02_ML_MODELS.md` - XGBoost, LightGBM
- `02_modeling/03_LSTM_GRU.md` - Deep Learning cho Time Series
