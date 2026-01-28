# 🎯 EVENT-AWARE TRAINING
## Đánh trọng số cao cho shock events - Pain Point #1

---

## 📚 MỤC LỤC

1. [Vấn đề với Training thông thường](#1-vấn-đề-với-training-thông-thường)
2. [Event-Aware Training là gì?](#2-event-aware-training-là-gì)
3. [Phát hiện Event Days](#3-phát-hiện-event-days)
4. [Weighted Loss Functions](#4-weighted-loss-functions)
5. [Event-Aware Metrics](#5-event-aware-metrics)
6. [Implementation Guide](#6-implementation-guide)
7. [Bài tập thực hành](#7-bài-tập-thực-hành)

---

## 1. VẤN ĐỀ VỚI TRAINING THÔNG THƯỜNG

### 🤔 Vấn đề

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

### 📊 Ví dụ thực tế

**COVID Crash (Feb-Mar 2020):**
```
Normal days:
- Model dự đoán: 100 → Actual: 101 (Error = 1%)

Event days (COVID crash):
- Model dự đoán: 100 → Actual: 85 (Error = 15%)

→ Model KHÔNG học được pattern của crash
   vì chỉ có vài ngày crash trong 10 năm data!
```

### 💡 Giải pháp

> **Event-Aware Training: Đánh trọng số CAO HƠN cho event days**

```
Loss = Σ weight(i) × (y_pred(i) - y_true(i))²

Trong đó:
- weight = 1.0 cho normal days
- weight = 5.0 cho event days

→ Model phải học tốt cả normal và event days!
```

---

## 2. EVENT-AWARE TRAINING LÀ GÌ?

### 🎯 Định nghĩa

> **Event-Aware Training = Training với weighted loss, đánh trọng số cao hơn cho những ngày có sự kiện quan trọng**

### 📊 So sánh

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
Day 3 (event):   Loss = 1.00, Weight = 5.0 → Weighted Loss = 5.00 ⚠️
Day 4 (normal):  Loss = 0.01, Weight = 1.0 → Weighted Loss = 0.01

Average Loss = (0.01 + 0.02 + 5.00 + 0.01) / 4 = 1.26

→ Model BẮT BUỘC phải học tốt event days!
```

### 💡 Lợi ích

1. **Dự đoán tốt hơn trên event days**
2. **Phát hiện sớm shocks/anomalies**
3. **Risk management tốt hơn**
4. **Đóng góp nghiên cứu mới** (ít paper làm điều này!)

---

## 3. PHÁT HIỆN EVENT DAYS

### 🎯 Định nghĩa Event Day

**Event Day = Ngày có biến động BẤT THƯỜNG**

**Tiêu chí:**
1. **Price shock:** Return > 3σ (3 standard deviations)
2. **Volume spike:** Volume > 2× average
3. **Volatility spike:** Volatility > 2× average
4. **News event:** Có tin tức quan trọng
5. **Filing event:** Có báo cáo tài chính

### 📊 Method 1: Statistical Detection

**Dựa vào Price:**
```python
def detect_price_events(df, threshold=3):
    """
    Phát hiện event dựa vào price returns
    
    Args:
        df: DataFrame với cột 'return_1d'
        threshold: Số standard deviations (default: 3)
    
    Returns:
        Boolean series: True = event day
    """
    returns = df['return_1d']
    mean = returns.mean()
    std = returns.std()
    
    # Event = return vượt quá threshold × std
    upper_bound = mean + threshold * std
    lower_bound = mean - threshold * std
    
    is_event = (returns > upper_bound) | (returns < lower_bound)
    
    return is_event

# Sử dụng
df['is_price_event'] = detect_price_events(df, threshold=3)
print(f"Detected {df['is_price_event'].sum()} price events")
```

**Dựa vào Volume:**
```python
def detect_volume_events(df, threshold=2):
    """
    Phát hiện event dựa vào volume spike
    
    Args:
        df: DataFrame với cột 'volume' và 'volume_ma_20'
        threshold: Multiplier (default: 2)
    
    Returns:
        Boolean series: True = event day
    """
    # Volume ratio = volume / moving average
    volume_ratio = df['volume'] / df['volume_ma_20']
    
    # Event = volume > threshold × average
    is_event = volume_ratio > threshold
    
    return is_event

# Sử dụng
df['is_volume_event'] = detect_volume_events(df, threshold=2)
print(f"Detected {df['is_volume_event'].sum()} volume events")
```

**Dựa vào Volatility:**
```python
def detect_volatility_events(df, window=20, threshold=2):
    """
    Phát hiện event dựa vào volatility spike
    
    Args:
        df: DataFrame với cột 'return_1d'
        window: Window cho rolling volatility
        threshold: Multiplier (default: 2)
    
    Returns:
        Boolean series: True = event day
    """
    # Tính rolling volatility
    returns = df['return_1d']
    rolling_vol = returns.rolling(window=window).std()
    
    # Tính average volatility
    avg_vol = rolling_vol.mean()
    
    # Event = volatility > threshold × average
    is_event = rolling_vol > threshold * avg_vol
    
    return is_event

# Sử dụng
df['is_vol_event'] = detect_volatility_events(df, window=20, threshold=2)
print(f"Detected {df['is_vol_event'].sum()} volatility events")
```

### 📊 Method 2: Composite Score

**Kết hợp nhiều signals:**
```python
def detect_events_composite(df, 
                           price_threshold=3,
                           volume_threshold=2,
                           vol_threshold=2,
                           min_score=2):
    """
    Phát hiện events bằng composite score
    
    Event = ít nhất min_score signals kích hoạt
    
    Args:
        df: DataFrame
        price_threshold: Threshold cho price
        volume_threshold: Threshold cho volume
        vol_threshold: Threshold cho volatility
        min_score: Số signals tối thiểu (default: 2)
    
    Returns:
        Boolean series: True = event day
    """
    # Detect từng loại
    price_event = detect_price_events(df, price_threshold)
    volume_event = detect_volume_events(df, volume_threshold)
    vol_event = detect_volatility_events(df, 20, vol_threshold)
    
    # Tính score (số signals kích hoạt)
    score = price_event.astype(int) + volume_event.astype(int) + vol_event.astype(int)
    
    # Event = score >= min_score
    is_event = score >= min_score
    
    return is_event, score

# Sử dụng
df['is_event'], df['event_score'] = detect_events_composite(df, min_score=2)

print(f"\n=== EVENT DETECTION SUMMARY ===")
print(f"Total days: {len(df)}")
print(f"Event days: {df['is_event'].sum()} ({df['is_event'].mean()*100:.2f}%)")
print(f"\nEvent score distribution:")
print(df['event_score'].value_counts().sort_index())
```

### 📊 Method 3: Machine Learning Detection

**Train model phát hiện anomalies:**
```python
from sklearn.ensemble import IsolationForest

def detect_events_ml(df, contamination=0.05):
    """
    Phát hiện events bằng Isolation Forest
    
    Args:
        df: DataFrame
        contamination: Tỷ lệ anomalies dự kiến (default: 5%)
    
    Returns:
        Boolean series: True = event day
    """
    # Features cho anomaly detection
    features = ['return_1d', 'volume_ratio', 'volatility_20', 
                'rsi_14', 'daily_range_pct']
    X = df[features].dropna()
    
    # Train Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(X)
    
    # -1 = anomaly, 1 = normal
    is_event = predictions == -1
    
    return pd.Series(is_event, index=X.index)

# Sử dụng
df['is_ml_event'] = detect_events_ml(df, contamination=0.05)
print(f"Detected {df['is_ml_event'].sum()} ML events")
```

### 💡 Visualize Events

```python
import matplotlib.pyplot as plt

def visualize_events(df, event_col='is_event'):
    """
    Visualize events trên price chart
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Price chart với event markers
    axes[0].plot(df.index, df['close'], label='Close Price', alpha=0.7)
    event_days = df[df[event_col]]
    axes[0].scatter(event_days.index, event_days['close'], 
                   color='red', s=50, label='Event Days', zorder=5)
    axes[0].set_title('Price with Event Days')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Returns
    axes[1].plot(df.index, df['return_1d'], label='Returns', alpha=0.7)
    axes[1].scatter(event_days.index, event_days['return_1d'], 
                   color='red', s=50, label='Event Days', zorder=5)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_title('Returns with Event Days')
    axes[1].set_ylabel('Returns (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Volume
    axes[2].bar(df.index, df['volume'], label='Volume', alpha=0.7)
    axes[2].scatter(event_days.index, event_days['volume'], 
                   color='red', s=50, label='Event Days', zorder=5)
    axes[2].set_title('Volume with Event Days')
    axes[2].set_ylabel('Volume')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('event_detection.png', dpi=300)
    plt.show()

# Visualize
visualize_events(df, event_col='is_event')
```

---

## 4. WEIGHTED LOSS FUNCTIONS

### 🎯 Weighted MSE

**Công thức:**
```
Weighted MSE = (1/n) × Σ w(i) × (y_pred(i) - y_true(i))²

Trong đó:
- w(i) = weight cho sample i
- w(i) = 1.0 cho normal days
- w(i) = k > 1.0 cho event days (k = 3, 5, 10, ...)
```

**Implementation:**
```python
def weighted_mse_loss(y_true, y_pred, weights):
    """
    Weighted MSE loss
    
    Args:
        y_true: True values
        y_pred: Predictions
        weights: Sample weights
    
    Returns:
        Weighted MSE
    """
    squared_errors = (y_true - y_pred) ** 2
    weighted_errors = weights * squared_errors
    return np.mean(weighted_errors)

# Ví dụ
y_true = np.array([100, 102, 85, 103])  # Day 3 là event (85)
y_pred = np.array([101, 103, 95, 104])
weights = np.array([1.0, 1.0, 5.0, 1.0])  # Day 3 có weight = 5.0

loss = weighted_mse_loss(y_true, y_pred, weights)
print(f"Weighted MSE: {loss:.2f}")

# So sánh với MSE thông thường
normal_mse = np.mean((y_true - y_pred) ** 2)
print(f"Normal MSE: {normal_mse:.2f}")
```

### 🔧 Weighted Loss cho PyTorch

```python
import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss cho PyTorch
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, y_pred, y_true, weights):
        """
        Args:
            y_pred: Predictions (batch_size, 1)
            y_true: True values (batch_size, 1)
            weights: Sample weights (batch_size, 1)
        
        Returns:
            Weighted MSE loss
        """
        squared_errors = (y_pred - y_true) ** 2
        weighted_errors = weights * squared_errors
        return torch.mean(weighted_errors)

# Sử dụng
criterion = WeightedMSELoss()

# Trong training loop
for batch in dataloader:
    X, y, weights = batch
    
    # Forward
    y_pred = model(X)
    
    # Loss với weights
    loss = criterion(y_pred, y, weights)
    
    # Backward
    loss.backward()
    optimizer.step()
```

### 🔧 Weighted Loss cho TensorFlow/Keras

```python
import tensorflow as tf

def weighted_mse_loss_tf(y_true, y_pred, weights):
    """
    Weighted MSE Loss cho TensorFlow
    """
    squared_errors = tf.square(y_true - y_pred)
    weighted_errors = weights * squared_errors
    return tf.reduce_mean(weighted_errors)

# Hoặc dùng sample_weight trong fit()
model.fit(
    X_train, y_train,
    sample_weight=train_weights,  # ← Truyền weights vào đây
    epochs=100,
    batch_size=32
)
```

### 💡 Chọn Weight như thế nào?

**Strategy 1: Fixed Weights**
```python
def assign_fixed_weights(df, event_col='is_event', event_weight=5.0):
    """
    Fixed weight cho event days
    """
    weights = np.ones(len(df))
    weights[df[event_col]] = event_weight
    return weights

weights = assign_fixed_weights(df, event_weight=5.0)
```

**Strategy 2: Proportional Weights**
```python
def assign_proportional_weights(df, event_col='is_event'):
    """
    Weight tỷ lệ nghịch với số lượng
    
    Ví dụ:
    - Normal days: 95% → weight = 1.0
    - Event days: 5% → weight = 95/5 = 19.0
    """
    n_total = len(df)
    n_events = df[event_col].sum()
    n_normal = n_total - n_events
    
    event_weight = n_normal / n_events if n_events > 0 else 1.0
    
    weights = np.ones(len(df))
    weights[df[event_col]] = event_weight
    
    return weights

weights = assign_proportional_weights(df)
```

**Strategy 3: Score-Based Weights**
```python
def assign_score_based_weights(df, score_col='event_score', base_weight=1.0):
    """
    Weight dựa vào event score
    
    Score 0: weight = 1.0
    Score 1: weight = 2.0
    Score 2: weight = 4.0
    Score 3: weight = 8.0
    """
    weights = base_weight * (2 ** df[score_col])
    return weights

weights = assign_score_based_weights(df)
```

---

## 5. EVENT-AWARE METRICS

### 🎯 Tại sao cần Event-Aware Metrics?

**Vấn đề với metrics thông thường:**
```
Overall MSE = 3.0 (trông tốt!)

Nhưng:
- MSE trên normal days = 1.0 (tốt)
- MSE trên event days = 15.0 (tệ!)

→ Model tốt trên normal, KÉM trên events
   nhưng overall MSE không phản ánh điều này!
```

### 📊 Event-Specific Metrics

**1. Separate Metrics cho Normal vs Event Days:**
```python
def evaluate_by_event(y_true, y_pred, is_event):
    """
    Tính metrics riêng cho normal và event days
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Normal days
    normal_mask = ~is_event
    mse_normal = mean_squared_error(y_true[normal_mask], y_pred[normal_mask])
    mae_normal = mean_absolute_error(y_true[normal_mask], y_pred[normal_mask])
    
    # Event days
    event_mask = is_event
    mse_event = mean_squared_error(y_true[event_mask], y_pred[event_mask])
    mae_event = mean_absolute_error(y_true[event_mask], y_pred[event_mask])
    
    # Overall
    mse_overall = mean_squared_error(y_true, y_pred)
    mae_overall = mean_absolute_error(y_true, y_pred)
    
    results = {
        'MSE_overall': mse_overall,
        'MSE_normal': mse_normal,
        'MSE_event': mse_event,
        'MAE_overall': mae_overall,
        'MAE_normal': mae_normal,
        'MAE_event': mae_event,
        'Event_ratio': is_event.mean()
    }
    
    return results

# Sử dụng
results = evaluate_by_event(y_test, y_pred, test_is_event)

print("\n=== EVENT-AWARE EVALUATION ===")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

**2. Tail Loss (Focus on Extreme Errors):**
```python
def tail_loss(y_true, y_pred, quantile=0.95):
    """
    Tail Loss: MSE chỉ tính trên errors lớn nhất
    
    Args:
        y_true: True values
        y_pred: Predictions
        quantile: Quantile threshold (default: 0.95 = top 5% errors)
    
    Returns:
        Tail MSE
    """
    errors = np.abs(y_true - y_pred)
    threshold = np.quantile(errors, quantile)
    
    # Chỉ tính MSE trên errors > threshold
    tail_mask = errors > threshold
    tail_mse = np.mean((y_true[tail_mask] - y_pred[tail_mask]) ** 2)
    
    return tail_mse

# Sử dụng
tail_mse = tail_loss(y_test, y_pred, quantile=0.95)
print(f"Tail MSE (top 5% errors): {tail_mse:.2f}")
```

**3. Direction Accuracy trên Event Days:**
```python
def direction_accuracy_event(y_true, y_pred, is_event):
    """
    Direction accuracy: Dự đoán đúng hướng tăng/giảm
    Tính riêng cho event days
    """
    # Tính direction (1 = tăng, 0 = giảm)
    true_direction = (y_true > 0).astype(int)
    pred_direction = (y_pred > 0).astype(int)
    
    # Accuracy trên event days
    event_mask = is_event
    correct = (true_direction[event_mask] == pred_direction[event_mask])
    accuracy = correct.mean()
    
    return accuracy

# Sử dụng
dir_acc = direction_accuracy_event(y_test_returns, y_pred_returns, test_is_event)
print(f"Direction Accuracy (event days): {dir_acc*100:.2f}%")
```

---

## 6. IMPLEMENTATION GUIDE

### 🔧 Full Pipeline

**Step 1: Detect Events**
```python
# Detect events
df['is_event'], df['event_score'] = detect_events_composite(
    df, 
    price_threshold=3,
    volume_threshold=2,
    vol_threshold=2,
    min_score=2
)

print(f"Detected {df['is_event'].sum()} events ({df['is_event'].mean()*100:.2f}%)")
```

**Step 2: Assign Weights**
```python
# Assign weights
weights = assign_proportional_weights(df, event_col='is_event')

print(f"Normal weight: {weights[~df['is_event']].mean():.2f}")
print(f"Event weight: {weights[df['is_event']].mean():.2f}")
```

**Step 3: Prepare Data**
```python
# Features và target
feature_cols = ['close', 'ma_20', 'rsi_14', 'macd', 'volatility_20']
X = df[feature_cols]
y = df['close'].shift(-1)  # Target: giá ngày mai

# Combine
data = pd.concat([X, y.rename('target'), 
                  df['is_event'], 
                  pd.Series(weights, index=df.index, name='weight')], 
                 axis=1).dropna()

# Split
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

X_train = train_data[feature_cols]
y_train = train_data['target']
weights_train = train_data['weight']
is_event_train = train_data['is_event']

X_test = test_data[feature_cols]
y_test = test_data['target']
weights_test = test_data['weight']
is_event_test = test_data['is_event']
```

**Step 4: Train với Weighted Loss**
```python
# Option 1: sklearn với sample_weight
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train, sample_weight=weights_train)

# Option 2: Custom training loop
# (Xem phần PyTorch/TensorFlow ở trên)
```

**Step 5: Evaluate**
```python
# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Event-aware evaluation
train_results = evaluate_by_event(y_train, y_pred_train, is_event_train)
test_results = evaluate_by_event(y_test, y_pred_test, is_event_test)

print("\n=== TRAINING RESULTS ===")
for metric, value in train_results.items():
    print(f"{metric}: {value:.4f}")

print("\n=== TEST RESULTS ===")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# Tail loss
tail_mse_train = tail_loss(y_train, y_pred_train, quantile=0.95)
tail_mse_test = tail_loss(y_test, y_pred_test, quantile=0.95)

print(f"\nTail MSE (train): {tail_mse_train:.2f}")
print(f"Tail MSE (test): {tail_mse_test:.2f}")
```

**Step 6: Compare với Baseline**
```python
# Train baseline (no weights)
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)  # Không có sample_weight

# Predictions
baseline_pred_test = baseline_model.predict(X_test)

# Compare
baseline_results = evaluate_by_event(y_test, baseline_pred_test, is_event_test)

print("\n=== COMPARISON ===")
print(f"{'Metric':<20} {'Baseline':<12} {'Event-Aware':<12} {'Improvement':<12}")
print("-" * 56)
for metric in ['MSE_overall', 'MSE_normal', 'MSE_event']:
    baseline_val = baseline_results[metric]
    event_aware_val = test_results[metric]
    improvement = (baseline_val - event_aware_val) / baseline_val * 100
    print(f"{metric:<20} {baseline_val:<12.4f} {event_aware_val:<12.4f} {improvement:>10.2f}%")
```

---

## 7. BÀI TẬP THỰC HÀNH

### 🎯 Bài tập 1: Event Detection

**Đề bài:**
Implement 3 methods phát hiện events cho FPT:
1. Statistical (price + volume + volatility)
2. Composite score
3. Machine Learning (Isolation Forest)

**Yêu cầu:**
- Detect events trên toàn bộ data
- So sánh 3 methods
- Visualize events
- Phân tích: Events có overlap không? Method nào tốt nhất?

**Kiểm tra:**
- [ ] Implement được 3 methods
- [ ] Detect được events
- [ ] Visualize đẹp
- [ ] Phân tích và so sánh

---

### 🎯 Bài tập 2: Event-Aware Training

**Đề bài:**
Train Linear Regression với event-aware loss

**Yêu cầu:**
- Detect events
- Assign weights (thử 3 strategies)
- Train với weighted loss
- So sánh với baseline (no weights)
- Evaluate với event-aware metrics

**Kiểm tra:**
- [ ] Train được với weighted loss
- [ ] So sánh được với baseline
- [ ] Chứng minh được improvement trên event days
- [ ] Viết báo cáo phân tích

---

### 🎯 Bài tập 3: Case Study - COVID Crash

**Đề bài:**
Phân tích performance của model trên COVID crash (Feb-Mar 2020)

**Yêu cầu:**
- Identify COVID crash period
- Train 2 models: Baseline vs Event-Aware
- Evaluate trên crash period
- Visualize predictions vs actual
- Phân tích: Model nào dự đoán tốt hơn? Tại sao?

**Kiểm tra:**
- [ ] Identify được crash period
- [ ] Train được 2 models
- [ ] So sánh performance
- [ ] Visualize và giải thích

---

## ✅ KIỂM TRA HIỂU BÀI

Trước khi sang bài tiếp theo, hãy đảm bảo bạn:

- [ ] Hiểu vấn đề với training thông thường
- [ ] Hiểu event-aware training là gì
- [ ] Implement được 3 methods phát hiện events
- [ ] Implement được weighted loss
- [ ] Hiểu cách chọn weights
- [ ] Implement được event-aware metrics
- [ ] Train được model với event-aware loss
- [ ] Chứng minh được improvement
- [ ] Làm được 3 bài tập thực hành

**Nếu chưa pass hết checklist, đọc lại phần tương ứng!**

---

## 📚 TÀI LIỆU THAM KHẢO

**Papers:**
- "Learning from Imbalanced Data" - He & Garcia (2009)
- "Cost-Sensitive Learning" - Elkan (2001)
- "Focal Loss for Dense Object Detection" - Lin et al. (2017)

**Related Work:**
- Hard Example Mining
- Curriculum Learning
- Importance Sampling

---

## 🚀 BƯỚC TIẾP THEO

Sau khi hoàn thành bài này, sang:
- `02_REGIME_DETECTION.md` - Phát hiện regime change
- `03_TAIL_RISK_METRICS.md` - Metrics cho tail events

**Chúc bạn học tốt! 🎓**
