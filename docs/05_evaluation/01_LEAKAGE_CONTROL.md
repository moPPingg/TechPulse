# Leakage Control cho Time Series & Event Data
## Kiểm soát rò rỉ thông tin trong Financial ML

---

## Mục lục

1. [Tổng quan về Data Leakage](#1-tổng-quan-về-data-leakage)
2. [Feature Leakage](#2-feature-leakage)
3. [Target Leakage](#3-target-leakage)
4. [Event/News Future Contamination](#4-eventnews-future-contamination)
5. [Purged Split và Embargo](#5-purged-split-và-embargo)
6. [Leakage Detection và Testing](#6-leakage-detection-và-testing)
7. [Implementation Checklist](#7-implementation-checklist)
8. [Bài tập thực hành](#8-bài-tập-thực-hành)

---

## 1. TỔNG QUAN VỀ DATA LEAKAGE

### 1.1. Định nghĩa

**Data Leakage = Model sử dụng thông tin KHÔNG CÓ SẴN tại thời điểm prediction**

```
Timeline:
Past ──────────────────── Now ──────────────────── Future
     [Available info]      │     [NOT available]
                           │
                     Prediction point

Leakage = Dùng info từ Future để predict
```

### 1.2. Tại sao Leakage nguy hiểm?

**Hậu quả:**
```
Backtest với leakage:
- Accuracy: 95%
- Sharpe: 5.0
- "Wow! Amazing model!"

Live trading (no leakage):
- Accuracy: 48%
- Sharpe: -0.5
- "Mất hết tiền..."

→ Leakage tạo ra ẢOTƯỞNG về performance
→ Model KHÔNG hoạt động trong thực tế
```

### 1.3. Các loại Leakage chính

| Loại | Mô tả | Ví dụ |
|------|-------|-------|
| **Feature Leakage** | Feature chứa thông tin từ tương lai | MA tính với center=True |
| **Target Leakage** | Target được tính từ future data | Return_5d bao gồm ngày mai |
| **Train-Test Leakage** | Test data leak vào train | Scaler fit trên toàn bộ data |
| **Temporal Leakage** | Dùng future info trong feature | Event xảy ra ngày mai |
| **Point-in-Time Leakage** | Data revision/update leak | EPS revised sau khi đã dùng |

---

## 2. FEATURE LEAKAGE

### 2.1. Rolling Window Leakage

**Vấn đề: center=True**

```python
# SAI: Rolling với center=True
df['ma_20'] = df['close'].rolling(20, center=True).mean()

# center=True nghĩa là:
# ma_20[t] = mean(close[t-10] đến close[t+9])
#                              ↑
#                     FUTURE DATA!
```

**Minh họa:**
```
Thời điểm t=15:

center=False (đúng):
ma[15] = mean(close[0:15])  ← Chỉ dùng past
         [0,1,2,...,14,15]

center=True (sai):
ma[15] = mean(close[5:25])  ← Dùng cả future!
         [...,14,15,16,...,24]
                  ↑
              LEAKAGE!
```

**Fix:**
```python
# ĐÚNG: Rolling chỉ dùng past data
df['ma_20'] = df['close'].rolling(20, center=False).mean()

# Thêm shift để chắc chắn (recommended)
df['ma_20'] = df['close'].rolling(20).mean().shift(1)
```

### 2.2. Expanding Window Leakage

```python
# SAI: Expanding statistics trên toàn bộ data
df['cumulative_mean'] = df['close'].expanding().mean()
# Vấn đề: expanding() vẫn include current observation
# Nếu dùng cumulative_mean để predict close → LEAKAGE

# ĐÚNG: Shift expanding statistics
df['cumulative_mean'] = df['close'].expanding().mean().shift(1)
```

### 2.3. Scaling/Normalization Leakage

```python
# SAI: Fit scaler trên TOÀN BỘ data trước khi split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['close_scaled'] = scaler.fit_transform(df[['close']])
# mean và std được tính từ CẢ train và test
# → Test data đã "leak" vào scaling parameters

# Sau đó split
train = df[:split_idx]
test = df[split_idx:]
# Quá muộn! Test info đã trong scaler

# ĐÚNG: Split TRƯỚC, fit scaler CHỈ trên train
train = df[:split_idx]
test = df[split_idx:]

scaler = StandardScaler()
scaler.fit(train[['close']])  # Chỉ fit trên train!

train['close_scaled'] = scaler.transform(train[['close']])
test['close_scaled'] = scaler.transform(test[['close']])
```

### 2.4. Feature Selection Leakage

```python
# SAI: Feature selection trên TOÀN BỘ data
from sklearn.feature_selection import SelectKBest, f_regression

X = df[feature_cols]
y = df['target']

selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(X, y)  # LEAKAGE!
# Feature selection dùng correlation với y từ CẢ train + test

# Sau đó split
X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
# Test data đã ảnh hưởng đến feature selection!

# ĐÚNG: Feature selection CHỈ trên train
train_X = df[:split_idx][feature_cols]
train_y = df[:split_idx]['target']
test_X = df[split_idx:][feature_cols]
test_y = df[split_idx:]['target']

selector = SelectKBest(f_regression, k=10)
selector.fit(train_X, train_y)  # Chỉ fit trên train!

train_X_selected = selector.transform(train_X)
test_X_selected = selector.transform(test_X)  # Apply cùng selection
```

### 2.5. Cross-sectional Feature Leakage

```python
# SAI: Rank features dùng future data
# Sector mean return được tính bao gồm future

df['sector_mean_return'] = df.groupby('sector')['return'].transform('mean')
# Nếu "return" bao gồm future returns → LEAKAGE

# ĐÚNG: Dùng LAG của cross-sectional features
df['sector_mean_return_lag1'] = df.groupby('sector')['return'].transform(
    lambda x: x.shift(1).expanding().mean()
)
```

---

## 3. TARGET LEAKAGE

### 3.1. Target Definition Leakage

**Vấn đề: Target chứa current/future info**

```python
# SAI: Target là return ngày HÔM NAY
df['target'] = df['close'] / df['close'].shift(1) - 1
# Vấn đề: Features tại t dùng close[t]
# Target cũng dùng close[t]
# → Model có thể "cheat" bằng cách dùng close[t] từ features

# ĐÚNG: Target là return ngày MAI (shift target)
df['target'] = (df['close'].shift(-1) / df['close']) - 1
# Features dùng info đến t
# Target là return từ t đến t+1
```

### 3.2. Multi-day Return Leakage

```python
# SAI: 5-day return bắt đầu từ HÔM NAY
df['return_5d'] = df['close'].shift(-5) / df['close'] - 1
# Vấn đề: return_5d[t] = close[t+5]/close[t] - 1
# Cần close[t] → Nếu features cũng dùng close[t] thì OK
# NHƯNG nếu features dùng close[t+1] (do bug) → LEAKAGE

# ĐÚNG với clear documentation
df['target_return_5d'] = df['close'].shift(-5) / df['close'] - 1
# Prediction point: t
# Features: info available at end of day t
# Target: return from close[t] to close[t+5]
```

### 3.3. Overlapping Returns Leakage

**Vấn đề: Overlapping returns giữa train và test**

```python
# Ví dụ: 5-day returns
# Day 1: return[1:6] = close[6]/close[1] - 1
# Day 2: return[2:7] = close[7]/close[2] - 1
# Day 3: return[3:8] = close[8]/close[3] - 1

# Nếu Day 3 trong train, Day 5 trong test:
# return[3:8] và return[5:10] đều dùng close[6], close[7], close[8]
# → LEAKAGE!

# FIX: Thêm PURGE GAP = horizon
def create_purged_split(df, horizon=5, test_start_idx=None):
    """
    Tạo split với purge gap để tránh overlapping returns
    """
    train = df[:test_start_idx - horizon]  # Bỏ horizon days cuối train
    test = df[test_start_idx:]
    
    return train, test

# Hoặc: Dùng NON-OVERLAPPING returns
df['return_5d_non_overlap'] = df['close'].shift(-5) / df['close'] - 1
df['return_5d_non_overlap'] = df['return_5d_non_overlap'].iloc[::5]  # Chỉ lấy mỗi 5 ngày
```

---

## 4. EVENT/NEWS FUTURE CONTAMINATION

### 4.1. Announcement Time Leakage

**Vấn đề: Event được ghi nhận TRƯỚC khi publicly available**

```
Timeline:
          Market close     Earnings announced    Market open
                │                  │                  │
Day 1:    ─────15:00──────────17:00─────────────────────────
                │                  │                  │
Data record: [Day 1, earnings=True]    ← WRONG!
                                  ↑
                        Info chỉ available sau 17:00
                        Nhưng data ghi nhận cho Day 1
                        → Model dùng earnings=True để predict Day 1 return
                        → LEAKAGE!

ĐÚNG:
Day 1: earnings=False (chưa biết)
Day 2: earnings=True (đã biết sau khi Day 1 close)
```

**Fix:**
```python
def fix_announcement_time_leakage(df, event_col, event_time_col=None):
    """
    Fix leakage từ announcement timing
    
    Args:
        df: DataFrame
        event_col: Column với event flag
        event_time_col: Column với thời gian event (optional)
    """
    # Simple fix: Shift event forward 1 day
    df[f'{event_col}_available'] = df[event_col].shift(1)
    
    # Better fix: Check announcement time
    if event_time_col:
        # Nếu event sau market close (15:00) → shift forward
        df[f'{event_col}_available'] = df.apply(
            lambda row: 0 if pd.isna(row[event_col]) else (
                row[event_col] if row.get(event_time_col, '15:00') < '15:00'
                else 0
            ),
            axis=1
        ).shift(1)
    
    return df
```

### 4.2. News Sentiment Leakage

```python
# SAI: Dùng sentiment của news được publish trong ngày để predict return ngày đó
# News 10:00 → predict return from 9:00 to 15:00 → LEAKAGE!

# ĐÚNG: Chỉ dùng news published TRƯỚC prediction time
def get_point_in_time_sentiment(df, news_df, max_hours_ago=24):
    """
    Get sentiment chỉ từ news đã published
    """
    results = []
    
    for idx, row in df.iterrows():
        prediction_time = row['datetime']
        
        # Chỉ lấy news published trước prediction_time
        valid_news = news_df[
            (news_df['publish_time'] < prediction_time) &
            (news_df['publish_time'] >= prediction_time - pd.Timedelta(hours=max_hours_ago))
        ]
        
        avg_sentiment = valid_news['sentiment'].mean() if len(valid_news) > 0 else 0
        results.append(avg_sentiment)
    
    df['news_sentiment'] = results
    return df
```

### 4.3. Filing/Report Leakage

```python
# Vấn đề: Financial reports có "as of" date và "filing" date khác nhau

# Ví dụ: Q4 2023 report
# - Period end: 2023-12-31 (as of date)
# - Filing date: 2024-02-15 (khi report được nộp/công bố)

# SAI: Dùng report tại as-of date
df['eps_q4'] = df['date'].apply(lambda d: get_eps(d, period='Q4'))
# Nếu d = 2024-01-15 → Lấy Q4 2023 EPS
# Nhưng Q4 2023 EPS chưa được công bố!
# → LEAKAGE!

# ĐÚNG: Dùng report tại filing date
def get_point_in_time_fundamental(df, fundamental_df):
    """
    Get fundamental data với point-in-time accuracy
    
    fundamental_df phải có:
    - period_end: Ngày kết thúc kỳ báo cáo
    - filing_date: Ngày công bố
    - value: Giá trị (EPS, revenue, etc.)
    """
    results = []
    
    for idx, row in df.iterrows():
        as_of_date = row['date']
        
        # Chỉ lấy reports đã được filed TRƯỚC as_of_date
        available_reports = fundamental_df[
            fundamental_df['filing_date'] < as_of_date
        ]
        
        # Lấy report mới nhất đã available
        if len(available_reports) > 0:
            latest = available_reports.sort_values('filing_date').iloc[-1]
            results.append(latest['value'])
        else:
            results.append(np.nan)
    
    df['fundamental_pit'] = results
    return df
```

### 4.4. Revision/Restatement Leakage

```python
# Vấn đề: Data được REVISED sau khi công bố lần đầu

# Ví dụ: GDP
# - Initial release: GDP Q4 = 2.1%
# - First revision (1 tháng sau): GDP Q4 = 2.3%
# - Final revision (3 tháng sau): GDP Q4 = 2.5%

# Database thường chỉ lưu FINAL value
# → Nếu backtest dùng final value tại initial release date → LEAKAGE!

# ĐÚNG: Dùng vintaged data (lưu tất cả revisions)
class VintageDataLoader:
    """
    Load data với vintage tracking
    """
    def __init__(self, vintage_df):
        """
        vintage_df có columns:
        - indicator: Tên chỉ số (GDP, CPI, etc.)
        - reference_period: Kỳ báo cáo (Q4 2023)
        - release_date: Ngày release
        - value: Giá trị tại thời điểm release
        """
        self.vintage_df = vintage_df
    
    def get_value_as_of(self, indicator, reference_period, as_of_date):
        """
        Get value của indicator như nó ĐƯỢC BIẾT tại as_of_date
        """
        available = self.vintage_df[
            (self.vintage_df['indicator'] == indicator) &
            (self.vintage_df['reference_period'] == reference_period) &
            (self.vintage_df['release_date'] <= as_of_date)
        ]
        
        if len(available) == 0:
            return np.nan
        
        # Lấy release gần nhất
        latest = available.sort_values('release_date').iloc[-1]
        return latest['value']
```

---

## 5. PURGED SPLIT VÀ EMBARGO

### 5.1. Purged Split

**Định nghĩa:** Loại bỏ samples trong train set có thể overlap với test set.

```
Standard split (có leakage tiềm ẩn):
Train: [████████████████████]
Test:                        [████████]

Với 5-day returns, observations cuối train overlap với đầu test!

Purged split:
Train: [████████████████]    
Gap:                    [░░░░]  ← PURGE (removed)
Test:                        [████████]
```

**Implementation:**
```python
def purged_train_test_split(X, y, test_size=0.2, purge_gap=5):
    """
    Train-test split với purge gap để tránh overlap
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Tỷ lệ test set
        purge_gap: Số observations bị loại giữa train và test
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Test là phần cuối
    test_start = n_samples - n_test
    
    # Train kết thúc trước test - purge_gap
    train_end = test_start - purge_gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]
    
    print(f"Train: {len(X_train)} samples (idx 0 to {train_end-1})")
    print(f"Purge: {purge_gap} samples (idx {train_end} to {test_start-1})")
    print(f"Test:  {len(X_test)} samples (idx {test_start} to {n_samples-1})")
    
    return X_train, X_test, y_train, y_test

# Usage
X_train, X_test, y_train, y_test = purged_train_test_split(
    X, y, 
    test_size=0.2, 
    purge_gap=5  # = prediction horizon
)
```

### 5.2. Embargo Period

**Định nghĩa:** Khoảng thời gian SAU train mà model không được test.

**Lý do:** Ngay sau train period, market conditions vẫn tương tự → dễ predict → overestimate performance.

```
Với Embargo:
Train: [████████████████]
Purge:                  [░░]
Embargo:                   [▓▓▓▓]  ← Không dùng để test
Test:                          [████████]
```

```python
def purged_embargo_split(X, y, test_size=0.2, purge_gap=5, embargo_pct=0.01):
    """
    Split với cả purge và embargo
    
    Args:
        embargo_pct: % của data dùng làm embargo (sau train, trước test thực sự)
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_embargo = int(n_samples * embargo_pct)
    
    # Test là phần cuối
    test_start = n_samples - n_test
    
    # Embargo là phần ngay sau train
    embargo_start = test_start - n_embargo
    
    # Train kết thúc trước embargo - purge_gap
    train_end = embargo_start - purge_gap
    
    # Splits
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    # Embargo samples (có thể dùng để monitor nhưng không để evaluate)
    X_embargo = X.iloc[embargo_start:test_start]
    y_embargo = y.iloc[embargo_start:test_start]
    
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]
    
    return X_train, X_test, y_train, y_test, X_embargo, y_embargo
```

### 5.3. Full Purged Cross-Validation

```python
class PurgedTimeSeriesCV:
    """
    Time Series CV với Purge + Embargo
    
    Dành cho research-grade evaluation
    """
    
    def __init__(self, 
                 n_splits=5,
                 purge_gap=5,
                 embargo_pct=0.01,
                 test_size=None):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.test_size = test_size
    
    def split(self, X):
        n_samples = len(X)
        n_embargo = int(n_samples * self.embargo_pct)
        
        # Tính test size nếu không specified
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        splits = []
        
        for fold in range(self.n_splits):
            # Test end (từ cuối data đi lên)
            test_end = n_samples - fold * test_size
            test_start = test_end - test_size
            
            if test_start < 0:
                break
            
            # Train end: test_start - embargo - purge
            train_end = test_start - n_embargo - self.purge_gap
            
            if train_end < test_size:  # Minimum train size
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits[::-1]  # Reverse để fold 1 là earliest
    
    def get_n_splits(self, X=None):
        return len(self.split(X)) if X is not None else self.n_splits


# Usage với sklearn-compatible interface
from sklearn.model_selection import cross_val_score

cv = PurgedTimeSeriesCV(n_splits=5, purge_gap=5, embargo_pct=0.01)

# Manual loop (recommended để có full control)
for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Fold {fold}: Score = {score:.4f}")
```

---

## 6. LEAKAGE DETECTION VÀ TESTING

### 6.1. Correlation Test

**Ý tưởng:** Nếu feature có correlation quá cao với target → có thể leakage

```python
def detect_leakage_by_correlation(X, y, threshold=0.9):
    """
    Phát hiện potential leakage qua correlation bất thường
    
    Correlation > threshold → Very suspicious!
    """
    suspicious = []
    
    for col in X.columns:
        corr = np.corrcoef(X[col].values, y.values)[0, 1]
        
        if abs(corr) > threshold:
            suspicious.append({
                'feature': col,
                'correlation': corr,
                'warning': 'VERY HIGH - Likely leakage!'
            })
        elif abs(corr) > 0.7:
            suspicious.append({
                'feature': col,
                'correlation': corr,
                'warning': 'HIGH - Check carefully'
            })
    
    if suspicious:
        print("⚠️ POTENTIAL LEAKAGE DETECTED:")
        for item in suspicious:
            print(f"  {item['feature']}: r = {item['correlation']:.3f} - {item['warning']}")
    else:
        print("✓ No obvious correlation leakage detected")
    
    return suspicious
```

### 6.2. Train-Test Performance Gap Test

**Ý tưởng:** Gap quá nhỏ giữa train và test → có thể leakage

```python
def detect_leakage_by_performance_gap(model, X_train, y_train, X_test, y_test):
    """
    So sánh train vs test performance
    
    Nếu gần như bằng nhau → Suspicious (quá tốt để thật)
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Gap analysis
    mae_gap = (test_mae - train_mae) / train_mae * 100
    r2_gap = train_r2 - test_r2
    
    print("=== LEAKAGE DETECTION: Performance Gap ===")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Gap: {mae_gap:+.1f}%")
    print()
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"Gap: {r2_gap:+.4f}")
    print()
    
    # Warnings
    if mae_gap < 5:
        print("⚠️ WARNING: Very small performance gap!")
        print("   This may indicate leakage or overfitting.")
    elif mae_gap < 20:
        print("✓ Gap seems reasonable")
    else:
        print("❌ Large gap - possible overfitting (but not leakage)")
    
    if train_r2 > 0.95:
        print("⚠️ WARNING: Train R² > 0.95 is very suspicious for financial data!")
    
    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'mae_gap_pct': mae_gap,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'r2_gap': r2_gap
    }
```

### 6.3. Time Shuffle Test

**Ý tưởng:** Shuffle time order → nếu performance không giảm nhiều → có leakage

```python
def time_shuffle_test(model, X, y, n_iter=10):
    """
    Test: Shuffle time order và so sánh performance
    
    Logic:
    - Nếu model dựa vào temporal patterns → shuffle làm giảm performance
    - Nếu model dựa vào leakage → shuffle không ảnh hưởng nhiều
    """
    from sklearn.model_selection import train_test_split
    
    # Original performance (with time order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model.fit(X_train, y_train)
    original_score = model.score(X_test, y_test)
    
    # Shuffled performance
    shuffled_scores = []
    for i in range(n_iter):
        # Shuffle indices
        shuffled_idx = np.random.permutation(len(X))
        X_shuffled = X.iloc[shuffled_idx].reset_index(drop=True)
        y_shuffled = y.iloc[shuffled_idx].reset_index(drop=True)
        
        # Split (now time order is meaningless)
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_shuffled, y_shuffled, test_size=0.2, random_state=i
        )
        
        model.fit(X_train_s, y_train_s)
        score = model.score(X_test_s, y_test_s)
        shuffled_scores.append(score)
    
    mean_shuffled = np.mean(shuffled_scores)
    std_shuffled = np.std(shuffled_scores)
    
    print("=== TIME SHUFFLE TEST ===")
    print(f"Original (time-ordered) R²: {original_score:.4f}")
    print(f"Shuffled mean R²: {mean_shuffled:.4f} ± {std_shuffled:.4f}")
    print()
    
    if mean_shuffled > original_score * 0.95:
        print("⚠️ WARNING: Shuffled performance nearly same as original!")
        print("   This strongly suggests data leakage.")
        print("   Time order should matter for valid features.")
    else:
        degradation = (original_score - mean_shuffled) / original_score * 100
        print(f"✓ Performance dropped {degradation:.1f}% when shuffled")
        print("   This suggests features are time-dependent (good)")
    
    return {
        'original_score': original_score,
        'shuffled_mean': mean_shuffled,
        'shuffled_std': std_shuffled
    }
```

### 6.4. Feature Timestamp Audit

```python
def audit_feature_timestamps(df, feature_cols, target_col, date_col='date'):
    """
    Kiểm tra xem mỗi feature có chứa future information không
    
    Logic: Feature tại t không nên correlate quá cao với target tại t-1
    (vì target[t-1] là thông tin từ quá khứ)
    """
    print("=== FEATURE TIMESTAMP AUDIT ===")
    print()
    
    issues = []
    
    for col in feature_cols:
        # Correlation với target shift forward (future target)
        corr_with_future = df[col].corr(df[target_col].shift(-1))
        
        # Correlation với target shift backward (past target)  
        corr_with_past = df[col].corr(df[target_col].shift(1))
        
        # Correlation với current target
        corr_current = df[col].corr(df[target_col])
        
        # Warning nếu correlation với future > correlation với current
        if abs(corr_with_future) > abs(corr_current) * 1.1:
            issues.append({
                'feature': col,
                'corr_current': corr_current,
                'corr_future': corr_with_future,
                'warning': 'Higher correlation with FUTURE target!'
            })
            print(f"⚠️ {col}:")
            print(f"   Corr with current target: {corr_current:.3f}")
            print(f"   Corr with future target:  {corr_with_future:.3f}")
            print(f"   → POTENTIAL LEAKAGE!")
            print()
    
    if not issues:
        print("✓ No timestamp issues detected")
    
    return issues
```

---

## 7. IMPLEMENTATION CHECKLIST

### 7.1. Feature Engineering Checklist

```
□ Rolling windows: center=False và có shift(1)
□ Expanding statistics: có shift(1)
□ Cross-sectional features: dùng lagged values
□ Technical indicators: không dùng future prices
□ All features: available TRƯỚC prediction time
```

### 7.2. Data Split Checklist

```
□ Split TRƯỚC khi fit scaler/encoder
□ Feature selection CHỈ trên train
□ Purge gap >= prediction horizon
□ Embargo period cho regime similarity
□ KHÔNG shuffle time series data
```

### 7.3. Event/News Checklist

```
□ Event flags: shift forward nếu announced sau market close
□ News sentiment: chỉ dùng news published TRƯỚC prediction
□ Fundamental data: dùng filing date, KHÔNG PHẢI period end date
□ Revised data: dùng vintage (value as-of specific date)
```

### 7.4. Testing Checklist

```
□ Correlation test: không có feature với |r| > 0.9
□ Performance gap: test MAE > train MAE × 1.05 (ít nhất)
□ Time shuffle test: performance giảm đáng kể khi shuffle
□ Feature timestamp audit: không có future correlation bất thường
```

---

## 8. BÀI TẬP THỰC HÀNH

### Bài tập 1: Phát hiện Leakage

**Yêu cầu:**
1. Load data với các features
2. Chạy correlation test
3. Chạy time shuffle test
4. Identify và fix leakage

### Bài tập 2: Implement Purged CV

**Yêu cầu:**
1. Implement PurgedTimeSeriesCV class
2. Test trên FPT data
3. So sánh với standard TimeSeriesSplit

### Bài tập 3: Point-in-Time Features

**Yêu cầu:**
1. Tạo earnings feature với point-in-time accuracy
2. Test for announcement time leakage
3. Compare model với/không có PIT accuracy

---

## Kiểm tra hiểu bài

- [ ] Liệt kê được 5 loại leakage chính
- [ ] Implement được purged split với embargo
- [ ] Phát hiện được leakage qua correlation test
- [ ] Fix được announcement time leakage
- [ ] Implement được point-in-time features

---

## Tài liệu tham khảo

**Papers:**
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "The Dangers of Leakage in ML for Finance" - multiple authors

**Best Practices:**
- Kaggle Leakage Detection
- sklearn TimeSeriesSplit documentation
