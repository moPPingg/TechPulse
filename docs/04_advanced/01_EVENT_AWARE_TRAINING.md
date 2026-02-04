# Event-Aware Training
## Đánh trọng số cao cho shock events - Pain Point #1

---

## Mục lục

1. [Vấn đề với Training thông thường](#1-vấn-đề-với-training-thông-thường)
2. [Event-Aware Training là gì?](#2-event-aware-training-là-gì)
3. [Causality vs Correlation trong Event Modeling](#3-causality-vs-correlation-trong-event-modeling)
4. [Event Leakage và cách tránh](#4-event-leakage-và-cách-tránh)
5. [Phát hiện Event Days](#5-phát-hiện-event-days)
6. [Event Encoding Strategies](#6-event-encoding-strategies)
7. [Time Decay Modeling](#7-time-decay-modeling)
8. [Weighted Loss Functions](#8-weighted-loss-functions)
9. [Đánh giá Event Impact](#9-đánh-giá-event-impact)
10. [Implementation Guide](#10-implementation-guide)
11. [Bài tập thực hành](#11-bài-tập-thực-hành)

---

## 1. VẤN ĐỀ VỚI TRAINING THÔNG THƯỜNG

### Vấn đề

**Training thông thường:**
```
Loss = MSE(y_pred, y_true)
     = (1/n) × Σ(y_pred - y_true)²

→ Mọi ngày đều được đối xử BÌNH ĐẲNG
```

**Hậu quả:**
```
Normal days (95%):  Error = 1%  → Loss = 0.01
Event days (5%):    Error = 10% → Loss = 1.00

Average Loss = 0.95 × 0.01 + 0.05 × 1.00 = 0.0595

→ Model optimize cho normal days
→ Bỏ qua event days (vì chỉ chiếm 5%)
→ Dự đoán KÉM khi có shock!
```

### Ví dụ thực tế: COVID Crash

```
Normal days:
- Model dự đoán: 100 → Actual: 101 (Error = 1%)

Event days (COVID crash):
- Model dự đoán: 100 → Actual: 85 (Error = 15%)

→ Model KHÔNG học được pattern của crash
   vì chỉ có vài ngày crash trong 10 năm data!
```

### Giải pháp

**Event-Aware Training: Đánh trọng số CAO HƠN cho event days**

```
Loss = Σ weight(i) × (y_pred(i) - y_true(i))²

Trong đó:
- weight = 1.0 cho normal days
- weight = 5.0 cho event days

→ Model phải học tốt cả normal và event days!
```

---

## 2. EVENT-AWARE TRAINING LÀ GÌ?

### Định nghĩa

**Event-Aware Training = Training với weighted loss, đánh trọng số cao hơn cho những ngày có sự kiện quan trọng**

### So sánh

**Traditional Training:**
```
Day 1 (normal):  Loss = 0.01, Weight = 1.0 → Weighted Loss = 0.01
Day 2 (normal):  Loss = 0.02, Weight = 1.0 → Weighted Loss = 0.02
Day 3 (event):   Loss = 1.00, Weight = 1.0 → Weighted Loss = 1.00
Day 4 (normal):  Loss = 0.01, Weight = 1.0 → Weighted Loss = 0.01

Average Loss = (0.01 + 0.02 + 1.00 + 0.01) / 4 = 0.26
```

**Event-Aware Training:**
```
Day 1 (normal):  Loss = 0.01, Weight = 1.0 → Weighted Loss = 0.01
Day 2 (normal):  Loss = 0.02, Weight = 1.0 → Weighted Loss = 0.02
Day 3 (event):   Loss = 1.00, Weight = 5.0 → Weighted Loss = 5.00
Day 4 (normal):  Loss = 0.01, Weight = 1.0 → Weighted Loss = 0.01

Average Loss = (0.01 + 0.02 + 5.00 + 0.01) / 4 = 1.26

→ Model BẮT BUỘC phải học tốt event days!
```

### Lợi ích

1. Dự đoán tốt hơn trên event days
2. Phát hiện sớm shocks/anomalies
3. Risk management tốt hơn
4. Đóng góp nghiên cứu mới

---

## 3. CAUSALITY VS CORRELATION TRONG EVENT MODELING

### 3.1. Vấn đề cốt lõi

**Correlation không phải Causation:**
```
Quan sát: Khi có earnings announcement, giá biến động mạnh
Correlation: earnings_day ↔ high_volatility ✓

Nhưng:
- Earnings announcement CAUSES price movement? (Đúng)
- Hay earnings announcement chỉ CORRELATED với price movement? (Khác nhau!)
```

### 3.2. Confounding Variables

**Ví dụ thực tế:**
```
Event: Fed tăng lãi suất
Correlation: Fed rate hike → Stock down

Nhưng confounding factors:
- Fed tăng lãi suất VÌ inflation cao
- Inflation cao → Stock down
- Fed rate hike và Stock down đều là CONSEQUENCES của inflation

Causal graph:
    Inflation
      /    \
     ↓      ↓
Fed rate → Stock
  hike     down

→ Fed rate hike có thể không CAUSE stock down
→ Cả hai đều là effect của inflation
```

### 3.3. Spurious Correlations trong Finance

```python
# Ví dụ spurious correlation
# "Số lượng phim Nicolas Cage correlate với đuối nước ở bể bơi"
# → Không có causal relationship!

# Trong finance:
# "VIX spike correlate với FPT drop"
# → Có causal relationship?
# → Hay cả hai đều do market sentiment?

def check_granger_causality(event_series, price_series, max_lag=5):
    """
    Granger Causality Test:
    - H0: Event does NOT Granger-cause price
    - H1: Event Granger-causes price
    
    Lưu ý: Granger causality ≠ True causality
    Chỉ test: Event có predictive power cho price không?
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    data = pd.DataFrame({
        'price_return': price_series,
        'event': event_series
    }).dropna()
    
    # Test
    results = grangercausalitytests(data[['price_return', 'event']], max_lag, verbose=False)
    
    # Extract p-values
    p_values = [results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
    
    return p_values

# Usage
p_values = check_granger_causality(df['earnings_event'], df['return'])
print(f"Granger causality p-values: {p_values}")
print(f"Significant at 5%: {[p < 0.05 for p in p_values]}")
```

### 3.4. Best Practices cho Causal Event Modeling

**1. Separate Event Sources:**
```python
# Tách biệt nguồn events
company_events = ['earnings', 'dividend', 'merger']  # Company-specific
market_events = ['fed_rate', 'vix_spike', 'oil_shock']  # Market-wide
sector_events = ['sector_rotation', 'regulation']  # Sector-specific

# Model effects riêng biệt
for event_type in company_events:
    model_company_effect(df, event_type)
```

**2. Control for Confounders:**
```python
# Thêm control variables
features_with_controls = [
    'event_flag',        # Event indicator
    'market_return',     # Control for market
    'sector_return',     # Control for sector
    'vix_level',         # Control for volatility regime
    'volume_ratio'       # Control for liquidity
]

# Causal effect ≈ coefficient of event_flag sau khi control
```

**3. Use Event Windows, not Point Estimates:**
```python
def event_window_analysis(df, event_dates, window_before=5, window_after=10):
    """
    Analyze returns trong event window
    
    CAR = Cumulative Abnormal Return
    """
    results = []
    
    for event_date in event_dates:
        # Get window
        start = event_date - pd.Timedelta(days=window_before)
        end = event_date + pd.Timedelta(days=window_after)
        
        window_data = df[start:end]
        
        # Calculate CAR
        car = window_data['abnormal_return'].sum()
        results.append({
            'event_date': event_date,
            'CAR': car,
            'pre_return': window_data[:event_date]['return'].sum(),
            'post_return': window_data[event_date:]['return'].sum()
        })
    
    return pd.DataFrame(results)
```

---

## 4. EVENT LEAKAGE VÀ CÁCH TRÁNH

### 4.1. Event Leakage là gì?

**Định nghĩa:** Model sử dụng thông tin về event TRƯỚC KHI event xảy ra hoặc được biết.

**Các dạng Event Leakage:**

### 4.2. Dạng 1: Temporal Leakage

```python
# SAI: Event flag tại t biết trước event sẽ xảy ra tại t+1
df['event_tomorrow'] = df['is_event'].shift(-1)  # LOOKAHEAD!
df['prediction'] = model.predict(df[['close', 'event_tomorrow']])

# ĐÚNG: Chỉ dùng event đã xảy ra
df['event_yesterday'] = df['is_event'].shift(1)  # Past event
df['prediction'] = model.predict(df[['close', 'event_yesterday']])
```

### 4.3. Dạng 2: Announcement Time Leakage

```python
# VẤN ĐỀ: Earnings được announce sau market close
# Nhưng data có thể ghi nhận vào ngày đó

# SAI:
# Date: 2024-01-15 (earnings announced at 4:30 PM)
# Training: Dùng earnings_flag = 1 để predict return ngày 15/1
# → Leakage! Earnings được biết SAU market close

# ĐÚNG: Shift event forward 1 day
df['earnings_known'] = df['earnings_flag'].shift(1)
# Ngày 15/1: earnings_known = 0 (chưa biết)
# Ngày 16/1: earnings_known = 1 (đã biết)
```

### 4.4. Dạng 3: Pre-announcement Drift Leakage

```python
# VẤN ĐỀ: Insiders biết trước earnings → Price moves trước announcement

# SAI: Feature capture pre-announcement drift
df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
# Momentum 5 ngày có thể capture insider trading trước earnings
# → Model learns: "high momentum before earnings → good earnings"
# → Nhưng đây là LEAKAGE vì dựa vào insider information

# ĐÚNG: Exclude pre-announcement window từ features
def remove_pre_event_features(df, event_col, window=5):
    """Remove features trong window trước event"""
    mask = df[event_col].rolling(window).max().shift(-window).fillna(0) > 0
    # Hoặc: tính features mà không dùng pre-event window
    return df[~mask]
```

### 4.5. Dạng 4: Survival Bias Leakage

```python
# VẤN ĐỀ: Chỉ train trên stocks còn tồn tại
# Stocks bị delisted (do bankruptcy) bị loại khỏi data

# SAI:
df = load_current_stocks()  # Chỉ có stocks còn tồn tại
# → Không có stocks đã bị delisted
# → Model không học được patterns dẫn đến delisting

# ĐÚNG: Include delisted stocks với proper handling
df = load_all_stocks(include_delisted=True)
df['is_delisted'] = (df['status'] == 'delisted')
# Train với all stocks, đánh dấu delisting events
```

### 4.6. Checklist tránh Event Leakage

```
Point-in-Time Checklist:

□ Event flags chỉ = 1 SAU KHI event đã được publicly known
□ Earnings: Available sau announcement time (not end of day)
□ News: Available sau publication time
□ Filings: Available sau filing time (not event date)

Feature Engineering Checklist:

□ Rolling features không bao gồm future data
□ Pre-event windows được exclude hoặc handle đúng
□ Shift tất cả event features ít nhất 1 period

Data Quality Checklist:

□ Include delisted stocks
□ Handle corporate actions (splits, dividends)
□ No look-ahead in data joins
```

### 4.7. Test for Event Leakage

```python
def test_event_leakage(df, event_col, return_col, window=10):
    """
    Test: Model có leverage thông tin từ sau event không?
    
    Nếu có leakage:
    - Pre-event returns có predict power quá cao
    - Model performance tốt bất thường
    """
    results = []
    
    event_dates = df[df[event_col] == 1].index
    
    for event_date in event_dates:
        # Pre-event returns (should NOT predict event type)
        pre_start = event_date - pd.Timedelta(days=window)
        pre_returns = df[pre_start:event_date][return_col].values
        
        # Post-event returns
        post_end = event_date + pd.Timedelta(days=window)
        post_returns = df[event_date:post_end][return_col].values
        
        results.append({
            'event_date': event_date,
            'pre_return': pre_returns.sum() if len(pre_returns) > 0 else 0,
            'post_return': post_returns.sum() if len(post_returns) > 0 else 0
        })
    
    results_df = pd.DataFrame(results)
    
    # Check: Pre-event returns should be ~0 (random)
    pre_mean = results_df['pre_return'].mean()
    pre_std = results_df['pre_return'].std()
    
    print(f"Pre-event mean return: {pre_mean:.4f} (should be ~0)")
    print(f"Pre-event std: {pre_std:.4f}")
    
    if abs(pre_mean) > 2 * pre_std / np.sqrt(len(results_df)):
        print("WARNING: Significant pre-event drift detected!")
        print("This may indicate event leakage or insider trading.")
    
    return results_df
```

---

## 5. PHÁT HIỆN EVENT DAYS

### Method 1: Statistical Detection

```python
def detect_price_events(df, threshold=3):
    """
    Phát hiện event dựa vào price returns
    Event = return vượt quá threshold × std
    """
    returns = df['return_1d']
    mean = returns.mean()
    std = returns.std()
    
    upper_bound = mean + threshold * std
    lower_bound = mean - threshold * std
    
    is_event = (returns > upper_bound) | (returns < lower_bound)
    
    return is_event

def detect_volume_events(df, threshold=2):
    """Phát hiện event dựa vào volume spike"""
    volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
    is_event = volume_ratio > threshold
    return is_event

def detect_volatility_events(df, window=20, threshold=2):
    """Phát hiện event dựa vào volatility spike"""
    returns = df['return_1d']
    rolling_vol = returns.rolling(window=window).std()
    avg_vol = rolling_vol.mean()
    is_event = rolling_vol > threshold * avg_vol
    return is_event
```

### Method 2: Composite Score

```python
def detect_events_composite(df, min_score=2):
    """
    Phát hiện events bằng composite score
    Event = ít nhất min_score signals kích hoạt
    """
    price_event = detect_price_events(df, 3)
    volume_event = detect_volume_events(df, 2)
    vol_event = detect_volatility_events(df, 20, 2)
    
    score = price_event.astype(int) + volume_event.astype(int) + vol_event.astype(int)
    is_event = score >= min_score
    
    return is_event, score
```

### Method 3: Machine Learning Detection

```python
from sklearn.ensemble import IsolationForest

def detect_events_ml(df, contamination=0.05):
    """Phát hiện events bằng Isolation Forest"""
    features = ['return_1d', 'volume_ratio', 'volatility_20', 'rsi_14']
    X = df[features].dropna()
    
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(X)
    
    is_event = predictions == -1
    return pd.Series(is_event, index=X.index)
```

---

## 6. EVENT ENCODING STRATEGIES

### 6.1. Binary Flags (Simplest)

**Ý tưởng:** Event = 1, Non-event = 0

```python
def binary_event_encoding(df, event_types):
    """
    Binary flags cho mỗi loại event
    
    Returns: DataFrame với columns: event_earnings, event_dividend, etc.
    """
    for event_type in event_types:
        df[f'event_{event_type}'] = df[event_type].astype(int)
    
    return df

# Usage
df = binary_event_encoding(df, ['earnings', 'dividend', 'fed_meeting'])

# Features: [close, ma_20, rsi, event_earnings, event_dividend, event_fed]
```

**Ưu điểm:**
- Đơn giản, dễ interpret
- Ít parameters

**Nhược điểm:**
- Không capture event magnitude
- Không phân biệt event tốt vs xấu

### 6.2. Categorical/Multi-class Encoding

**Ý tưởng:** Phân loại events thành categories

```python
def categorical_event_encoding(df):
    """
    Categorical encoding cho event types
    0: No event
    1: Positive earnings surprise
    2: Negative earnings surprise
    3: Dividend announcement
    4: Fed rate hike
    5: Fed rate cut
    ...
    """
    df['event_category'] = 0  # Default: no event
    
    # Positive earnings
    mask_pos_earnings = (df['earnings_event'] == 1) & (df['earnings_surprise'] > 0)
    df.loc[mask_pos_earnings, 'event_category'] = 1
    
    # Negative earnings
    mask_neg_earnings = (df['earnings_event'] == 1) & (df['earnings_surprise'] < 0)
    df.loc[mask_neg_earnings, 'event_category'] = 2
    
    # Fed rate hike
    mask_fed_hike = (df['fed_event'] == 1) & (df['rate_change'] > 0)
    df.loc[mask_fed_hike, 'event_category'] = 4
    
    # One-hot encode
    event_dummies = pd.get_dummies(df['event_category'], prefix='event_cat')
    df = pd.concat([df, event_dummies], axis=1)
    
    return df
```

### 6.3. Sentiment Scores

**Ý tưởng:** Encode event với sentiment strength

```python
def sentiment_event_encoding(df, news_sentiment_col='news_sentiment'):
    """
    Sentiment-based event encoding
    
    Range: [-1, 1]
    -1: Very negative event
     0: Neutral/No event
    +1: Very positive event
    """
    # Option 1: Direct sentiment score
    df['event_sentiment'] = df[news_sentiment_col].fillna(0)
    
    # Option 2: Combine multiple sources
    df['event_sentiment'] = (
        0.4 * df['news_sentiment'].fillna(0) +
        0.3 * df['analyst_sentiment'].fillna(0) +
        0.3 * df['social_sentiment'].fillna(0)
    )
    
    # Option 3: Earnings surprise as sentiment
    df['earnings_sentiment'] = np.where(
        df['earnings_event'] == 1,
        np.tanh(df['earnings_surprise'] / df['earnings_surprise'].std()),  # Normalize
        0
    )
    
    return df
```

### 6.4. Event Embeddings (Deep Learning)

**Ý tưởng:** Learn dense representations cho events

```python
import torch
import torch.nn as nn

class EventEmbedding(nn.Module):
    """
    Learnable embeddings cho event types
    
    Thay vì one-hot [0,0,1,0,0] → Dense [0.2, -0.1, 0.5, 0.3]
    """
    def __init__(self, num_event_types, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_event_types, embedding_dim)
    
    def forward(self, event_ids):
        return self.embedding(event_ids)

# Usage
# Event types: 0=no_event, 1=earnings, 2=dividend, 3=fed, ...
event_embed = EventEmbedding(num_event_types=10, embedding_dim=16)

# Input: batch of event type IDs
event_ids = torch.tensor([0, 1, 3, 0, 2])  # [no_event, earnings, fed, no_event, dividend]
embeddings = event_embed(event_ids)  # Shape: (5, 16)
```

**Combined Model với Event Embeddings:**
```python
class EventAwareModel(nn.Module):
    def __init__(self, num_features, num_event_types, event_embed_dim=16, hidden_dim=64):
        super().__init__()
        
        # Event embedding
        self.event_embed = nn.Embedding(num_event_types, event_embed_dim)
        
        # Main network
        self.fc1 = nn.Linear(num_features + event_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x_features, x_event_type):
        # Get event embedding
        event_emb = self.event_embed(x_event_type)  # (batch, event_embed_dim)
        
        # Concatenate features and event embedding
        x = torch.cat([x_features, event_emb], dim=1)  # (batch, num_features + event_embed_dim)
        
        # Forward
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
```

### 6.5. So sánh Encoding Strategies

| Strategy | Complexity | Interpretability | Expressiveness | Use Case |
|----------|------------|-----------------|----------------|----------|
| **Binary** | Low | High | Low | Simple events |
| **Categorical** | Medium | Medium | Medium | Discrete event types |
| **Sentiment** | Medium | High | Medium | News, earnings |
| **Embeddings** | High | Low | High | Many event types, DL |

---

## 7. TIME DECAY MODELING

### 7.1. Tại sao cần Time Decay?

**Vấn đề:** Event impact giảm dần theo thời gian

```
Day 0 (event day):   Impact = 100%
Day 1:               Impact = 80%
Day 5:               Impact = 30%
Day 10:              Impact = 10%
Day 20:              Impact ≈ 0%
```

**Không có decay:**
```python
# SAI: Binary flag không capture decay
df['event_flag'] = df['is_event']
# Day 0: event_flag = 1
# Day 1: event_flag = 0  ← Event impact đã mất hoàn toàn!
```

### 7.2. Exponential Decay

**Công thức:**
```
impact(t) = exp(-λ × t) × initial_impact

λ = decay rate
t = days since event
```

```python
def exponential_decay_features(df, event_col, decay_rate=0.1, max_window=20):
    """
    Tạo features với exponential decay
    
    Args:
        df: DataFrame
        event_col: Event column name
        decay_rate: Lambda (higher = faster decay)
        max_window: Maximum days to look back
    
    Returns:
        Series with decayed event impact
    """
    impact = np.zeros(len(df))
    
    for i in range(len(df)):
        # Look back up to max_window days
        for lag in range(min(i + 1, max_window + 1)):
            if df[event_col].iloc[i - lag] == 1:
                # Add decayed impact
                impact[i] += np.exp(-decay_rate * lag)
    
    return pd.Series(impact, index=df.index, name=f'{event_col}_decay')

# Usage
df['event_impact'] = exponential_decay_features(df, 'is_event', decay_rate=0.1)

# Visualization
print(f"Day 0: impact = {np.exp(-0.1 * 0):.2f}")  # 1.00
print(f"Day 5: impact = {np.exp(-0.1 * 5):.2f}")  # 0.61
print(f"Day 10: impact = {np.exp(-0.1 * 10):.2f}")  # 0.37
print(f"Day 20: impact = {np.exp(-0.1 * 20):.2f}")  # 0.14
```

### 7.3. Linear Decay

```python
def linear_decay_features(df, event_col, half_life=10):
    """
    Linear decay features
    
    Args:
        half_life: Days until impact reduces by 50%
    """
    impact = np.zeros(len(df))
    
    for i in range(len(df)):
        for lag in range(min(i + 1, 2 * half_life)):
            if df[event_col].iloc[i - lag] == 1:
                # Linear decay
                decay = max(0, 1 - lag / (2 * half_life))
                impact[i] += decay
    
    return pd.Series(impact, index=df.index)
```

### 7.4. Event-Specific Decay Rates

```python
def multi_event_decay(df, event_configs):
    """
    Different decay rates cho different event types
    
    Args:
        event_configs: Dict of {event_col: decay_rate}
            - Earnings: decay_rate = 0.05 (slow decay, long impact)
            - News: decay_rate = 0.3 (fast decay, short impact)
    """
    total_impact = np.zeros(len(df))
    
    for event_col, decay_rate in event_configs.items():
        impact = exponential_decay_features(df, event_col, decay_rate)
        total_impact += impact
    
    return pd.Series(total_impact, index=df.index, name='total_event_impact')

# Usage
event_configs = {
    'earnings_event': 0.05,     # Slow decay (14 days half-life)
    'fed_event': 0.07,          # Medium decay (10 days half-life)
    'news_event': 0.2,          # Fast decay (3.5 days half-life)
    'technical_event': 0.3      # Very fast decay (2.3 days half-life)
}

df['event_impact'] = multi_event_decay(df, event_configs)
```

### 7.5. Learn Decay Rate

```python
from scipy.optimize import minimize

def optimize_decay_rate(df, event_col, target_col):
    """
    Optimize decay rate để maximize predictive power
    """
    def objective(decay_rate):
        # Create decayed feature
        impact = exponential_decay_features(df, event_col, decay_rate[0])
        
        # Correlation với target
        corr = np.corrcoef(impact.values, df[target_col].values)[0, 1]
        
        # Minimize negative correlation (maximize correlation)
        return -abs(corr)
    
    # Optimize
    result = minimize(objective, x0=[0.1], bounds=[(0.01, 1.0)])
    optimal_decay = result.x[0]
    
    print(f"Optimal decay rate: {optimal_decay:.4f}")
    print(f"Half-life: {np.log(2) / optimal_decay:.1f} days")
    
    return optimal_decay

# Usage
optimal_decay = optimize_decay_rate(df, 'earnings_event', 'return_5d')
```

---

## 8. WEIGHTED LOSS FUNCTIONS (D3.1 - Chi tiết)

### 8.1. Công thức Event-Weighted MAE

**Standard MAE:**
```
MAE = (1/n) × Σ|y_true(i) - y_pred(i)|
```

**Event-Weighted MAE:**
```
Weighted_MAE = Σ[w(i) × |y_true(i) - y_pred(i)|] / Σw(i)

Trong đó:
- w(i) = 1.0 nếu i là normal day
- w(i) = w_event nếu i là event day
```

**Implementation:**
```python
import numpy as np
import torch
import torch.nn as nn

def weighted_mae_loss(y_true, y_pred, weights):
    """
    Event-Weighted MAE
    
    Args:
        y_true: Actual values (numpy array)
        y_pred: Predicted values (numpy array)
        weights: Sample weights (numpy array)
    
    Returns:
        Weighted MAE
    """
    abs_errors = np.abs(y_true - y_pred)
    weighted_errors = weights * abs_errors
    return np.sum(weighted_errors) / np.sum(weights)

# PyTorch version
class WeightedMAELoss(nn.Module):
    """Event-Weighted MAE Loss cho PyTorch"""
    
    def forward(self, y_pred, y_true, weights):
        abs_errors = torch.abs(y_pred - y_true)
        weighted_errors = weights * abs_errors
        return torch.sum(weighted_errors) / torch.sum(weights)
```

### 8.2. Công thức Event-Weighted RMSE

**Standard RMSE:**
```
RMSE = √[(1/n) × Σ(y_true(i) - y_pred(i))²]
```

**Event-Weighted RMSE:**
```
Weighted_RMSE = √[Σ[w(i) × (y_true(i) - y_pred(i))²] / Σw(i)]
```

**Implementation:**
```python
def weighted_rmse_loss(y_true, y_pred, weights):
    """
    Event-Weighted RMSE
    """
    squared_errors = (y_true - y_pred) ** 2
    weighted_errors = weights * squared_errors
    return np.sqrt(np.sum(weighted_errors) / np.sum(weights))

# PyTorch version
class WeightedMSELoss(nn.Module):
    """Event-Weighted MSE Loss cho PyTorch"""
    
    def forward(self, y_pred, y_true, weights):
        squared_errors = (y_pred - y_true) ** 2
        weighted_errors = weights * squared_errors
        return torch.sum(weighted_errors) / torch.sum(weights)

class WeightedRMSELoss(nn.Module):
    """Event-Weighted RMSE Loss"""
    
    def forward(self, y_pred, y_true, weights):
        squared_errors = (y_pred - y_true) ** 2
        weighted_errors = weights * squared_errors
        mse = torch.sum(weighted_errors) / torch.sum(weights)
        return torch.sqrt(mse)
```

### 8.3. Event Mask và Event Window

**Event Mask (Binary):**
```python
def create_event_mask(df, event_col='is_event'):
    """
    Tạo binary mask cho event days
    
    Returns:
        Boolean array: True = event day
    """
    return df[event_col].values.astype(bool)
```

**Event Window (Mở rộng xung quanh event):**
```python
def create_event_window_mask(df, event_col='is_event', 
                             window_before=2, window_after=3):
    """
    Tạo event window mask
    
    Không chỉ event day, mà cả các ngày xung quanh
    vì event impact có thể kéo dài
    
    Args:
        window_before: Số ngày TRƯỚC event (anticipation)
        window_after: Số ngày SAU event (reaction)
    
    Returns:
        Boolean array với event windows marked
    
    Ví dụ với window_before=2, window_after=3:
    
    Event tại t=10:
    t:    6  7  8  9  10  11  12  13  14
    mask: 0  0  1  1   1   1   1   1   0
              ↑  ↑   ↑   ↑   ↑   ↑
              before event  after
    """
    mask = np.zeros(len(df), dtype=bool)
    event_indices = np.where(df[event_col].values)[0]
    
    for idx in event_indices:
        start = max(0, idx - window_before)
        end = min(len(df), idx + window_after + 1)
        mask[start:end] = True
    
    return mask

# Usage
event_mask = create_event_window_mask(
    df, 
    event_col='is_event',
    window_before=2,  # 2 ngày trước (market anticipation)
    window_after=5    # 5 ngày sau (market reaction)
)
print(f"Event window days: {event_mask.sum()} / {len(df)} ({event_mask.mean()*100:.1f}%)")
```

**Graduated Event Window (Weight giảm dần):**
```python
def create_graduated_event_weights(df, event_col='is_event',
                                   event_weight=5.0,
                                   window_before=2, 
                                   window_after=5,
                                   decay='linear'):
    """
    Tạo weights giảm dần từ event day
    
    Event day: weight = event_weight
    Days around: weight giảm dần theo khoảng cách
    
    Args:
        decay: 'linear' hoặc 'exponential'
    """
    weights = np.ones(len(df))
    event_indices = np.where(df[event_col].values)[0]
    
    for event_idx in event_indices:
        # Event day chính
        weights[event_idx] = event_weight
        
        # Days before event (anticipation)
        for offset in range(1, window_before + 1):
            idx = event_idx - offset
            if idx >= 0:
                if decay == 'linear':
                    w = 1 + (event_weight - 1) * (1 - offset / (window_before + 1))
                else:  # exponential
                    w = 1 + (event_weight - 1) * np.exp(-offset / window_before)
                weights[idx] = max(weights[idx], w)
        
        # Days after event (reaction)
        for offset in range(1, window_after + 1):
            idx = event_idx + offset
            if idx < len(df):
                if decay == 'linear':
                    w = 1 + (event_weight - 1) * (1 - offset / (window_after + 1))
                else:  # exponential
                    w = 1 + (event_weight - 1) * np.exp(-offset / window_after)
                weights[idx] = max(weights[idx], w)
    
    return weights

# Visualization
weights = create_graduated_event_weights(
    df, event_weight=5.0, window_before=2, window_after=5, decay='exponential'
)
print("Weight profile around event:")
print("  Before: ", weights[event_idx-3:event_idx])
print("  Event:  ", weights[event_idx])
print("  After:  ", weights[event_idx+1:event_idx+6])
```

### 8.4. Weight Scaling Strategies

**Strategy 1: Fixed Weights**
```python
def assign_fixed_weights(df, event_col='is_event', event_weight=5.0):
    """
    Fixed weight cho event days
    
    Simple nhưng cần tune event_weight
    """
    weights = np.ones(len(df))
    weights[df[event_col].values] = event_weight
    return weights
```

**Strategy 2: Inverse Frequency Weights (Class-balanced)**
```python
def assign_inverse_frequency_weights(df, event_col='is_event'):
    """
    Weight tỷ lệ nghịch với frequency
    
    Giống class_weight='balanced' trong sklearn
    
    Công thức:
    w_event = n_total / (2 × n_events)
    w_normal = n_total / (2 × n_normal)
    
    Ví dụ: 1000 samples, 50 events
    w_event = 1000 / (2 × 50) = 10
    w_normal = 1000 / (2 × 950) = 0.526
    """
    n_total = len(df)
    n_events = df[event_col].sum()
    n_normal = n_total - n_events
    
    event_weight = n_total / (2 * n_events) if n_events > 0 else 1.0
    normal_weight = n_total / (2 * n_normal) if n_normal > 0 else 1.0
    
    weights = np.where(df[event_col].values, event_weight, normal_weight)
    return weights
```

**Strategy 3: Event Magnitude Weights**
```python
def assign_magnitude_weights(df, event_col='is_event', 
                            magnitude_col='return_abs',
                            base_weight=1.0,
                            scale_factor=2.0):
    """
    Weight based on event magnitude
    
    Bigger shocks → Bigger weights
    
    Công thức:
    w(i) = base_weight × (1 + scale_factor × magnitude(i))
    
    Trong đó magnitude có thể là:
    - |return| của ngày đó
    - Z-score của return
    - Volatility spike
    """
    weights = np.ones(len(df)) * base_weight
    
    # Chỉ scale weight cho event days
    event_mask = df[event_col].values.astype(bool)
    
    # Normalize magnitude to [0, 1]
    magnitudes = df[magnitude_col].values
    mag_min, mag_max = magnitudes[event_mask].min(), magnitudes[event_mask].max()
    normalized_mag = (magnitudes - mag_min) / (mag_max - mag_min + 1e-8)
    
    weights[event_mask] = base_weight * (1 + scale_factor * normalized_mag[event_mask])
    
    return weights
```

**Strategy 4: Temporal Importance Weights**
```python
def assign_temporal_weights(df, event_col='is_event',
                           recency_decay=0.001):
    """
    Gần đây hơn → Quan trọng hơn
    
    Kết hợp event weight với temporal importance
    
    Công thức:
    w(i) = event_weight(i) × exp(-decay × (T - t(i)))
    
    Trong đó:
    - T = thời điểm cuối cùng
    - t(i) = thời điểm của sample i
    - decay = tốc độ giảm (nhỏ = giảm chậm)
    """
    n = len(df)
    
    # Base weights (event vs non-event)
    event_weight = n / (2 * df[event_col].sum()) if df[event_col].sum() > 0 else 1.0
    weights = np.where(df[event_col].values, event_weight, 1.0)
    
    # Temporal decay
    time_indices = np.arange(n)
    temporal_weights = np.exp(-recency_decay * (n - time_indices))
    
    # Combine
    final_weights = weights * temporal_weights
    
    # Normalize để mean = 1
    final_weights = final_weights / final_weights.mean()
    
    return final_weights
```

### 8.5. Phân tích: Khi nào nên Weight vs không Weight

**Nên dùng Event Weighting khi:**
```
✓ Events hiếm nhưng quan trọng (< 10% samples)
✓ Model mục tiêu là capture extreme movements
✓ Risk management là priority
✓ Có ground truth labels cho events
```

**KHÔNG nên dùng hoặc cần cẩn thận khi:**
```
✗ Events quá nhiều (> 30% samples) → Weight không cần thiết
✗ Event definition không rõ ràng → Noisy weights
✗ Model đã overfit events → Weight làm tệ hơn
✗ Events không có predictive pattern → Bias vào noise
```

**Potential Bias từ Over-weighting:**
```python
def analyze_weight_bias(y_true, y_pred_weighted, y_pred_unweighted, is_event):
    """
    Phân tích bias do weighting
    """
    # Normal days performance
    normal_mask = ~is_event
    mae_normal_w = np.mean(np.abs(y_true[normal_mask] - y_pred_weighted[normal_mask]))
    mae_normal_u = np.mean(np.abs(y_true[normal_mask] - y_pred_unweighted[normal_mask]))
    
    # Event days performance
    event_mask = is_event
    mae_event_w = np.mean(np.abs(y_true[event_mask] - y_pred_weighted[event_mask]))
    mae_event_u = np.mean(np.abs(y_true[event_mask] - y_pred_unweighted[event_mask]))
    
    print("=== WEIGHT BIAS ANALYSIS ===")
    print(f"Normal Days MAE:")
    print(f"  Weighted:   {mae_normal_w:.4f}")
    print(f"  Unweighted: {mae_normal_u:.4f}")
    print(f"  Degradation: {(mae_normal_w - mae_normal_u) / mae_normal_u * 100:+.1f}%")
    print()
    print(f"Event Days MAE:")
    print(f"  Weighted:   {mae_event_w:.4f}")
    print(f"  Unweighted: {mae_event_u:.4f}")
    print(f"  Improvement: {(mae_event_u - mae_event_w) / mae_event_u * 100:+.1f}%")
    
    # Trade-off analysis
    if mae_normal_w > mae_normal_u * 1.1:
        print("\n⚠️ WARNING: Normal day performance degraded significantly!")
        print("   Consider reducing event weight or using graduated weights.")
```

### 8.6. Integrate vào Training Loop

**PyTorch Training Loop:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class EventAwareTrainer:
    """
    Trainer với Event-Weighted Loss
    """
    
    def __init__(self, model, loss_type='mse'):
        self.model = model
        
        if loss_type == 'mse':
            self.criterion = WeightedMSELoss()
        elif loss_type == 'mae':
            self.criterion = WeightedMAELoss()
        else:
            self.criterion = WeightedMSELoss()
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            X, y, weights = batch
            
            # Forward
            optimizer.zero_grad()
            y_pred = self.model(X)
            
            # Weighted loss
            loss = self.criterion(y_pred, y, weights)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def fit(self, X_train, y_train, weights_train, 
            X_val=None, y_val=None, weights_val=None,
            epochs=100, lr=0.001, batch_size=32, early_stopping=10):
        """
        Full training với validation và early stopping
        """
        # Prepare data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1),
            torch.FloatTensor(weights_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_loss)
            
            # Validate
            if X_val is not None:
                val_loss = self.evaluate(X_val, y_val, weights_val)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}", end='')
                if X_val is not None:
                    print(f", Val Loss = {val_loss:.4f}")
                else:
                    print()
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pt'))
        
        return history
    
    def evaluate(self, X, y, weights):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            y_t = torch.FloatTensor(y).unsqueeze(1)
            w_t = torch.FloatTensor(weights).unsqueeze(1)
            
            y_pred = self.model(X_t)
            loss = self.criterion(y_pred, y_t, w_t)
        
        return loss.item()

# Usage
model = YourModel(input_dim=X_train.shape[1])
trainer = EventAwareTrainer(model, loss_type='mse')

weights = assign_inverse_frequency_weights(df_train, event_col='is_event')

history = trainer.fit(
    X_train, y_train, weights,
    X_val, y_val, weights_val,
    epochs=100,
    lr=0.001,
    early_stopping=10
)
```

---

## 9. ĐÁNH GIÁ EVENT IMPACT

### 9.1. Tại sao cần đánh giá riêng?

**Vấn đề:** Thêm events có thể KHÔNG improve performance!

```
Model A (no events):     MAE = 4.5
Model B (with events):   MAE = 4.4

Q: Events có thực sự giúp không?
- Improvement 2.2% có significant không?
- Improvement có đến từ events hay luck?
```

### 9.2. Event-Specific Evaluation

```python
def evaluate_by_event(y_true, y_pred, is_event):
    """Tính metrics riêng cho normal và event days"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    normal_mask = ~is_event
    event_mask = is_event
    
    results = {
        'MAE_overall': mean_absolute_error(y_true, y_pred),
        'MAE_normal': mean_absolute_error(y_true[normal_mask], y_pred[normal_mask]),
        'MAE_event': mean_absolute_error(y_true[event_mask], y_pred[event_mask]),
        'N_normal': normal_mask.sum(),
        'N_event': event_mask.sum()
    }
    
    return results
```

### 9.3. Ablation Study

```python
def event_ablation_study(X_train, y_train, X_test, y_test, is_event_test):
    """
    Ablation study: So sánh model có và không có event features
    """
    from sklearn.linear_model import Ridge
    
    # Feature sets
    base_features = ['close', 'ma_20', 'rsi_14', 'macd']
    event_features = ['event_flag', 'event_impact', 'event_sentiment']
    
    results = []
    
    # Model 1: Base features only
    model_base = Ridge()
    model_base.fit(X_train[base_features], y_train)
    pred_base = model_base.predict(X_test[base_features])
    
    metrics_base = evaluate_by_event(y_test.values, pred_base, is_event_test.values)
    metrics_base['Model'] = 'Base (no events)'
    results.append(metrics_base)
    
    # Model 2: Base + Event features
    all_features = base_features + event_features
    model_full = Ridge()
    model_full.fit(X_train[all_features], y_train)
    pred_full = model_full.predict(X_test[all_features])
    
    metrics_full = evaluate_by_event(y_test.values, pred_full, is_event_test.values)
    metrics_full['Model'] = 'Base + Events'
    results.append(metrics_full)
    
    # Compare
    results_df = pd.DataFrame(results)
    print("\n=== ABLATION STUDY ===")
    print(results_df.to_string(index=False))
    
    # Improvement on event days
    improvement_event = (
        (results[0]['MAE_event'] - results[1]['MAE_event']) / 
        results[0]['MAE_event'] * 100
    )
    print(f"\nImprovement on event days: {improvement_event:.2f}%")
    
    return results_df
```

### 9.4. Statistical Significance of Event Impact

```python
def test_event_feature_significance(X_train, y_train, X_test, y_test, 
                                    event_feature_col, n_bootstrap=100):
    """
    Test: Event features có significant impact không?
    
    Method: Bootstrap comparison
    """
    from sklearn.linear_model import Ridge
    
    base_features = [c for c in X_train.columns if c != event_feature_col]
    
    mae_base_list = []
    mae_full_list = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train.iloc[idx]
        y_boot = y_train.iloc[idx]
        
        # Model without event features
        model_base = Ridge()
        model_base.fit(X_boot[base_features], y_boot)
        pred_base = model_base.predict(X_test[base_features])
        mae_base = mean_absolute_error(y_test, pred_base)
        mae_base_list.append(mae_base)
        
        # Model with event features
        model_full = Ridge()
        model_full.fit(X_boot, y_boot)
        pred_full = model_full.predict(X_test)
        mae_full = mean_absolute_error(y_test, pred_full)
        mae_full_list.append(mae_full)
    
    # Compute improvement distribution
    improvements = [(b - f) / b * 100 for b, f in zip(mae_base_list, mae_full_list)]
    
    mean_improvement = np.mean(improvements)
    ci_lower = np.percentile(improvements, 2.5)
    ci_upper = np.percentile(improvements, 97.5)
    
    print(f"\nMean Improvement: {mean_improvement:.2f}%")
    print(f"95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    
    if ci_lower > 0:
        print("→ Event features SIGNIFICANTLY improve model")
    else:
        print("→ Event features may NOT significantly improve model")
    
    return {
        'mean_improvement': mean_improvement,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': ci_lower > 0
    }
```

### 9.5. Permutation Test

```python
def permutation_test_events(X_train, y_train, X_test, y_test, 
                           event_feature_col, n_permutations=100):
    """
    Permutation test: Shuffle event features và so sánh
    
    H0: Event features không có predictive power
    H1: Event features có predictive power
    """
    from sklearn.linear_model import Ridge
    
    # Original model
    model = Ridge()
    model.fit(X_train, y_train)
    mae_original = mean_absolute_error(y_test, model.predict(X_test))
    
    # Permuted models
    mae_permuted = []
    
    for _ in range(n_permutations):
        X_train_perm = X_train.copy()
        X_train_perm[event_feature_col] = np.random.permutation(X_train_perm[event_feature_col])
        
        model_perm = Ridge()
        model_perm.fit(X_train_perm, y_train)
        
        X_test_perm = X_test.copy()
        X_test_perm[event_feature_col] = np.random.permutation(X_test_perm[event_feature_col])
        
        mae_perm = mean_absolute_error(y_test, model_perm.predict(X_test_perm))
        mae_permuted.append(mae_perm)
    
    # P-value: Proportion of permuted MAE <= original MAE
    p_value = np.mean([m <= mae_original for m in mae_permuted])
    
    print(f"\nOriginal MAE: {mae_original:.4f}")
    print(f"Mean Permuted MAE: {np.mean(mae_permuted):.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("→ Event features have SIGNIFICANT predictive power")
    else:
        print("→ Event features may NOT have significant predictive power")
    
    return p_value
```

---

## 10. IMPLEMENTATION GUIDE

### Full Pipeline

```python
def event_aware_pipeline(df, feature_cols, target_col):
    """
    Complete event-aware training pipeline
    """
    # Step 1: Detect events
    df['is_event'], df['event_score'] = detect_events_composite(df, min_score=2)
    print(f"Detected {df['is_event'].sum()} events ({df['is_event'].mean()*100:.2f}%)")
    
    # Step 2: Create event features
    df['event_flag'] = df['is_event'].astype(int)
    df['event_impact'] = exponential_decay_features(df, 'is_event', decay_rate=0.1)
    
    # Step 3: Assign weights
    weights = assign_proportional_weights(df, event_col='is_event')
    
    # Step 4: Prepare data
    X = df[feature_cols + ['event_flag', 'event_impact']]
    y = df[target_col]
    
    # Step 5: Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    weights_train = weights[:split_idx]
    is_event_test = df['is_event'][split_idx:]
    
    # Step 6: Train with weighted loss
    from sklearn.linear_model import Ridge
    model = Ridge()
    model.fit(X_train, y_train, sample_weight=weights_train)
    
    # Step 7: Evaluate
    y_pred = model.predict(X_test)
    results = evaluate_by_event(y_test.values, y_pred, is_event_test.values)
    
    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    # Step 8: Ablation study
    ablation_results = event_ablation_study(
        X_train, y_train, X_test, y_test, is_event_test
    )
    
    return model, results
```

---

## 11. BÀI TẬP THỰC HÀNH

### Bài tập 1: Event Detection và Encoding

**Yêu cầu:**
1. Implement 3 event encoding strategies (binary, sentiment, embeddings)
2. Compare predictive power của mỗi encoding
3. Test for event leakage

### Bài tập 2: Time Decay Optimization

**Yêu cầu:**
1. Implement exponential và linear decay
2. Optimize decay rate cho earnings events
3. Compare với no-decay baseline

### Bài tập 3: Causality Analysis

**Yêu cầu:**
1. Run Granger causality test cho events
2. Identify confounding variables
3. Control for confounders và re-evaluate

### Bài tập 4: Full Evaluation

**Yêu cầu:**
1. Run ablation study
2. Bootstrap confidence intervals
3. Permutation test for significance

---

## Kiểm tra hiểu bài

- [ ] Phân biệt được causality vs correlation trong events
- [ ] Identify và tránh được event leakage
- [ ] Implement được multiple encoding strategies
- [ ] Implement được time decay modeling
- [ ] Đánh giá được event impact với statistical tests

---

## Tài liệu tham khảo

**Papers:**
- "Learning from Imbalanced Data" - He & Garcia (2009)
- "Event Studies in Economics and Finance" - MacKinlay (1997)
- "The Cross-Section of Expected Stock Returns" - Fama & French (1992)

**Related Work:**
- Event Study Methodology
- Causal Inference in Finance
- Time-Weighted Features

---

## Bước tiếp theo

Sau khi hoàn thành:
- `02_REGIME_DETECTION.md` - Phát hiện regime change
- `03_TAIL_RISK_METRICS.md` - Metrics cho tail events
