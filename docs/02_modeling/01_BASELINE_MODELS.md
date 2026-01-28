# 📊 BASELINE MODELS CHO TIME SERIES
## ARIMA, GARCH và Linear Models - Nền tảng để so sánh

---

## 📚 MỤC LỤC

1. [Tại sao cần Baseline?](#1-tại-sao-cần-baseline)
2. [Linear Regression](#2-linear-regression)
3. [ARIMA Models](#3-arima-models)
4. [GARCH Models](#4-garch-models)
5. [Naive Forecasting](#5-naive-forecasting)
6. [So sánh các Baselines](#6-so-sánh-các-baselines)
7. [Bài tập thực hành](#7-bài-tập-thực-hành)

---

## 1. TẠI SAO CẦN BASELINE?

### 🎯 Baseline là gì?

> **Baseline = Model đơn giản nhất để so sánh**

**Mục đích:**
- Đo lường xem model phức tạp có thực sự tốt hơn không
- Tránh "overkill" (dùng model phức tạp cho bài toán đơn giản)
- Hiểu được data trước khi dùng deep learning

### 📊 Ví dụ thực tế

**Tình huống:**
```
Bạn: "Tôi dùng LSTM dự đoán FPT, MSE = 5.0"
Reviewer: "So với baseline?"
Bạn: "Ừm... chưa có baseline..."
Reviewer: "Nếu chỉ dự đoán = giá hôm qua, MSE = 3.0"
→ LSTM của bạn còn tệ hơn baseline! ❌
```

**Đúng cách:**
```
Bạn: "Baseline (Naive) MSE = 5.0"
Bạn: "Linear Regression MSE = 4.2"
Bạn: "ARIMA MSE = 3.8"
Bạn: "LSTM MSE = 2.5"
→ LSTM tốt hơn baseline 50%! ✅
```

### 💡 Quy tắc vàng

> **LUÔN LUÔN implement baseline trước khi làm model phức tạp!**

---

## 2. LINEAR REGRESSION

### 🎯 Linear Regression cho Time Series

**Ý tưởng:**
```
price_tomorrow = w1×close_today + w2×ma20 + w3×rsi + w4×macd + b
```

**Ưu điểm:**
- Đơn giản, nhanh
- Dễ interpret (xem weights)
- Baseline tốt

**Nhược điểm:**
- Giả định linear relationship
- Không capture được non-linear patterns

### 🔧 Implementation

**Bước 1: Chuẩn bị data**
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load features
df = pd.read_csv('data/features/vn30/FPT.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Chọn features
feature_cols = ['close', 'ma_20', 'rsi_14', 'macd', 'volatility_20']
X = df[feature_cols]

# Target: giá ngày mai
y = df['close'].shift(-1)

# Drop NaN
data = pd.concat([X, y.rename('target')], axis=1).dropna()
X = data[feature_cols]
y = data['target']

# Train/test split (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

**Bước 2: Train model**
```python
# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\nModel trained!")
print(f"Intercept: {model.intercept_:.2f}")
print("\nCoefficients:")
for feat, coef in zip(feature_cols, model.coef_):
    print(f"  {feat}: {coef:.4f}")
```

**Bước 3: Evaluate**
```python
# Training metrics
train_mse = mean_squared_error(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)

# Test metrics
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)

print("\n=== EVALUATION ===")
print(f"Training MSE:  {train_mse:.2f}")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Training MAE:  {train_mae:.2f}")
print()
print(f"Test MSE:  {test_mse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test MAE:  {test_mae:.2f}")
```

**Bước 4: Visualize**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# Training predictions
plt.subplot(1, 2, 1)
plt.plot(y_train.values, label='Actual', alpha=0.7)
plt.plot(y_pred_train, label='Predicted', alpha=0.7)
plt.title(f'Training Set (MSE={train_mse:.2f})')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()

# Test predictions
plt.subplot(1, 2, 2)
plt.plot(y_test.values, label='Actual', alpha=0.7)
plt.plot(y_pred_test, label='Predicted', alpha=0.7)
plt.title(f'Test Set (MSE={test_mse:.2f})')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=300)
plt.show()
```

### 💡 Interpret Results

**Feature Importance (từ coefficients):**
```python
# Sắp xếp features theo importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['coefficient'])
plt.xlabel('Coefficient')
plt.title('Feature Importance (Linear Regression)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()
```

---

## 3. ARIMA MODELS

### 🤔 ARIMA là gì?

**ARIMA = AutoRegressive Integrated Moving Average**

**Phân tích từng phần:**
- **AR (AutoRegressive):** Dùng giá trị quá khứ để dự đoán
- **I (Integrated):** Differencing để làm stationary
- **MA (Moving Average):** Dùng errors quá khứ để dự đoán

### 📐 ARIMA(p, d, q)

**p (AR order):**
- Số lags của giá trị quá khứ
- Ví dụ: p=2 → dùng t-1 và t-2

**d (Differencing order):**
- Số lần differencing
- d=0: Không differencing
- d=1: First difference (price[t] - price[t-1])
- d=2: Second difference

**q (MA order):**
- Số lags của errors quá khứ
- Ví dụ: q=1 → dùng error tại t-1

### 🎯 Ví dụ: ARIMA(1,1,1)

**Công thức:**
```
Δy(t) = c + φ₁×Δy(t-1) + θ₁×ε(t-1) + ε(t)
 ↑       ↑    ↑           ↑
 Diff  Const  AR(1)       MA(1)

Trong đó:
- Δy(t) = y(t) - y(t-1) (first difference)
- φ₁: AR coefficient
- θ₁: MA coefficient
- ε(t): Error tại thời điểm t
```

### 🔧 Implementation

**Bước 1: Kiểm tra Stationarity**
```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, name='Series'):
    """
    Kiểm tra stationarity bằng ADF test
    """
    result = adfuller(series.dropna())
    
    print(f"\n=== ADF Test for {name} ===")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print("→ STATIONARY (p-value < 0.05)")
    else:
        print("→ NON-STATIONARY (p-value >= 0.05)")
    
    return result[1] < 0.05

# Kiểm tra price
is_stationary_price = check_stationarity(df['close'], 'Price')

# Kiểm tra returns
df['returns'] = df['close'].pct_change()
is_stationary_returns = check_stationarity(df['returns'], 'Returns')
```

**Bước 2: Chọn p, d, q**

**Cách 1: Auto ARIMA**
```python
from pmdarima import auto_arima

# Auto ARIMA sẽ tự động tìm p, d, q tốt nhất
model = auto_arima(
    df['close'],
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,  # Tự động tìm d
    seasonal=False,
    trace=True,  # In ra quá trình tìm kiếm
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(f"\nBest model: ARIMA{model.order}")
print(model.summary())
```

**Cách 2: Manual (dùng ACF/PACF)**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ACF plot (để chọn q)
plot_acf(df['returns'].dropna(), lags=20, ax=axes[0])
axes[0].set_title('ACF Plot')

# PACF plot (để chọn p)
plot_pacf(df['returns'].dropna(), lags=20, ax=axes[1])
axes[1].set_title('PACF Plot')

plt.tight_layout()
plt.show()

# Quy tắc:
# - Nếu ACF cuts off sau lag q → MA(q)
# - Nếu PACF cuts off sau lag p → AR(p)
# - Nếu cả 2 đều decay dần → ARMA(p,q)
```

**Bước 3: Train ARIMA**
```python
from statsmodels.tsa.arima.model import ARIMA

# Split data
train_size = int(len(df) * 0.8)
train = df['close'][:train_size]
test = df['close'][train_size:]

# Train ARIMA(1,1,1)
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

print(model_fit.summary())
```

**Bước 4: Forecast**
```python
# Forecast test period
forecast = model_fit.forecast(steps=len(test))

# Metrics
test_mse = mean_squared_error(test, forecast)
test_mae = mean_absolute_error(test, forecast)
test_rmse = np.sqrt(test_mse)

print(f"\n=== ARIMA EVALUATION ===")
print(f"Test MSE:  {test_mse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test MAE:  {test_mae:.2f}")

# Visualize
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='Training', alpha=0.7)
plt.plot(test.index, test, label='Actual Test', alpha=0.7)
plt.plot(test.index, forecast, label='ARIMA Forecast', alpha=0.7)
plt.title(f'ARIMA(1,1,1) Forecast (MSE={test_mse:.2f})')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=300)
plt.show()
```

### 💡 Khi nào dùng ARIMA?

**Dùng khi:**
- Data có autocorrelation mạnh
- Muốn model đơn giản, interpret được
- Dự đoán ngắn hạn (1-7 ngày)

**KHÔNG dùng khi:**
- Data có nhiều external factors (news, events)
- Cần dự đoán dài hạn (>1 tháng)
- Data có non-linear patterns phức tạp

---

## 4. GARCH MODELS

### 🤔 GARCH là gì?

**GARCH = Generalized AutoRegressive Conditional Heteroskedasticity**

**Mục đích:**
- Dự đoán **VOLATILITY** (độ biến động)
- KHÔNG dự đoán giá trực tiếp

**Tại sao quan trọng?**
- Volatility cao = Rủi ro cao
- Volatility clustering: Biến động lớn thường theo sau biến động lớn
- Quan trọng cho risk management

### 📐 GARCH(1,1)

**Công thức:**
```
σ²(t) = ω + α×ε²(t-1) + β×σ²(t-1)
 ↑       ↑    ↑           ↑
Vol(t) Const Error(t-1)  Vol(t-1)

Trong đó:
- σ²(t): Variance (volatility²) tại thời điểm t
- ε²(t-1): Squared error tại t-1
- α: ARCH coefficient
- β: GARCH coefficient
```

**Ý nghĩa:**
- α cao: Shocks ảnh hưởng mạnh đến volatility
- β cao: Volatility persistence (biến động kéo dài)
- α + β ≈ 1: Volatility rất persistent

### 🔧 Implementation

**Bước 1: Chuẩn bị returns**
```python
from arch import arch_model

# Tính returns (%)
df['returns'] = df['close'].pct_change() * 100
returns = df['returns'].dropna()

# Split
train_size = int(len(returns) * 0.8)
train_returns = returns[:train_size]
test_returns = returns[train_size:]

print(f"Training samples: {len(train_returns)}")
print(f"Test samples: {len(test_returns)}")
```

**Bước 2: Train GARCH**
```python
# Define GARCH(1,1) model
model = arch_model(
    train_returns,
    vol='Garch',  # GARCH model
    p=1,          # GARCH order
    q=1           # ARCH order
)

# Fit model
model_fit = model.fit(disp='off')
print(model_fit.summary())
```

**Bước 3: Forecast Volatility**
```python
# Forecast
forecast = model_fit.forecast(horizon=len(test_returns))

# Extract forecasted variance
forecast_variance = forecast.variance.values[-1, :]
forecast_volatility = np.sqrt(forecast_variance)

# Actual volatility (rolling std)
actual_volatility = test_returns.rolling(window=20).std()

# Metrics
vol_mse = mean_squared_error(
    actual_volatility.dropna(),
    forecast_volatility[:len(actual_volatility.dropna())]
)

print(f"\n=== GARCH EVALUATION ===")
print(f"Volatility MSE: {vol_mse:.4f}")
```

**Bước 4: Visualize**
```python
plt.figure(figsize=(14, 8))

# Returns
plt.subplot(2, 1, 1)
plt.plot(test_returns.index, test_returns, label='Returns', alpha=0.5)
plt.title('Test Returns')
plt.ylabel('Returns (%)')
plt.legend()

# Volatility
plt.subplot(2, 1, 2)
plt.plot(actual_volatility.index, actual_volatility, 
         label='Actual Volatility (Rolling Std)', alpha=0.7)
plt.plot(test_returns.index[:len(forecast_volatility)], forecast_volatility, 
         label='GARCH Forecast', alpha=0.7)
plt.title('Volatility Forecast')
plt.ylabel('Volatility (%)')
plt.xlabel('Date')
plt.legend()

plt.tight_layout()
plt.savefig('garch_forecast.png', dpi=300)
plt.show()
```

### 💡 Khi nào dùng GARCH?

**Dùng khi:**
- Cần dự đoán volatility/risk
- Data có volatility clustering
- Risk management, option pricing

**KHÔNG dùng khi:**
- Cần dự đoán giá trực tiếp (dùng ARIMA hoặc ML)

---

## 5. NAIVE FORECASTING

### 🎯 Naive Methods

**Đơn giản nhưng hiệu quả!**

#### **1. Naive Forecast (Last Value)**

**Công thức:**
```
ŷ(t+1) = y(t)

Ví dụ:
Giá hôm nay: 100
Dự đoán ngày mai: 100
```

**Code:**
```python
def naive_forecast(train, test):
    """
    Naive forecast: Dự đoán = giá trị cuối cùng của training
    """
    forecast = np.full(len(test), train.iloc[-1])
    return forecast

# Evaluate
forecast = naive_forecast(train, test)
mse = mean_squared_error(test, forecast)
print(f"Naive MSE: {mse:.2f}")
```

#### **2. Seasonal Naive**

**Công thức:**
```
ŷ(t+1) = y(t-m)

Trong đó m = seasonal period

Ví dụ (weekly seasonality, m=5):
Dự đoán Thứ 2 tuần này = Thứ 2 tuần trước
```

**Code:**
```python
def seasonal_naive_forecast(train, test, period=5):
    """
    Seasonal naive: Dự đoán = giá trị cùng kỳ trước
    """
    forecast = []
    for i in range(len(test)):
        if i < period:
            # Dùng giá trị từ training
            forecast.append(train.iloc[-(period-i)])
        else:
            # Dùng giá trị từ test
            forecast.append(forecast[i-period])
    return np.array(forecast)
```

#### **3. Moving Average**

**Công thức:**
```
ŷ(t+1) = (y(t) + y(t-1) + ... + y(t-k+1)) / k

Ví dụ (k=5):
Dự đoán ngày mai = Trung bình 5 ngày gần nhất
```

**Code:**
```python
def moving_average_forecast(train, test, window=5):
    """
    Moving average forecast
    """
    forecast = []
    history = list(train[-window:])
    
    for i in range(len(test)):
        # Dự đoán = trung bình window gần nhất
        pred = np.mean(history)
        forecast.append(pred)
        
        # Update history với actual value
        history.append(test.iloc[i])
        history.pop(0)
    
    return np.array(forecast)
```

---

## 6. SO SÁNH CÁC BASELINES

### 📊 Benchmark Template

```python
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(name, y_true, y_pred):
    """
    Evaluate and return metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# Collect results
results = []

# 1. Naive
forecast_naive = naive_forecast(train, test)
results.append(evaluate_model('Naive', test, forecast_naive))

# 2. Moving Average
forecast_ma = moving_average_forecast(train, test, window=5)
results.append(evaluate_model('Moving Average (5)', test, forecast_ma))

# 3. Linear Regression
# (đã train ở trên)
results.append(evaluate_model('Linear Regression', y_test, y_pred_test))

# 4. ARIMA
# (đã train ở trên)
results.append(evaluate_model('ARIMA(1,1,1)', test, forecast))

# Create comparison table
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('MSE')

print("\n=== BASELINE COMPARISON ===")
print(comparison_df.to_string(index=False))

# Visualize
comparison_df.plot(x='Model', y=['MSE', 'MAE'], kind='bar', figsize=(10, 6))
plt.title('Baseline Models Comparison')
plt.ylabel('Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('baseline_comparison.png', dpi=300)
plt.show()
```

### 💡 Interpret Results

**Ví dụ kết quả:**
```
Model                 MSE    RMSE    MAE    MAPE
Naive                 5.2    2.28    1.8    1.85%
Moving Average (5)    4.8    2.19    1.7    1.75%
ARIMA(1,1,1)         3.9    1.97    1.5    1.52%
Linear Regression     3.2    1.79    1.3    1.35%
```

**Kết luận:**
- Linear Regression tốt nhất (MSE thấp nhất)
- ARIMA tốt hơn Naive 25%
- Moving Average tốt hơn Naive 8%
- Baseline đã set được "bar" cho deep learning models

---

## 7. BÀI TẬP THỰC HÀNH

### 🎯 Bài tập 1: Implement Full Baseline Pipeline

**Đề bài:**
Implement và so sánh 4 baselines cho FPT:
1. Naive
2. Moving Average (window=5, 10, 20)
3. Linear Regression
4. ARIMA(p,d,q) - tự chọn p,d,q

**Yêu cầu:**
- Train trên 80% data
- Test trên 20% data
- Tính MSE, MAE, RMSE, MAPE
- Vẽ biểu đồ so sánh
- Viết báo cáo ngắn (200-300 từ)

**Kiểm tra:**
- [ ] Implement được 4 baselines
- [ ] Tính được metrics đầy đủ
- [ ] Vẽ được biểu đồ đẹp
- [ ] Viết được báo cáo phân tích

---

### 🎯 Bài tập 2: GARCH cho Volatility Forecasting

**Đề bài:**
Dùng GARCH(1,1) dự đoán volatility của FPT

**Yêu cầu:**
- Tính returns
- Train GARCH(1,1)
- Forecast volatility cho test period
- So sánh với actual volatility (rolling std)
- Phân tích α và β coefficients

**Kiểm tra:**
- [ ] Train được GARCH
- [ ] Forecast được volatility
- [ ] So sánh với actual
- [ ] Giải thích được α, β

---

### 🎯 Bài tập 3: Feature Engineering cho Linear Regression

**Đề bài:**
Cải thiện Linear Regression bằng feature engineering

**Gợi ý features:**
- Lagged features (close_lag1, close_lag5, ...)
- Rolling statistics (rolling_mean_5, rolling_std_10, ...)
- Interaction features (close × ma20, rsi × volume_ratio, ...)
- Polynomial features (close², close³, ...)

**Yêu cầu:**
- Thêm ít nhất 10 features mới
- Train Linear Regression với features mới
- So sánh với baseline Linear Regression
- Phân tích feature importance

**Kiểm tra:**
- [ ] Tạo được features mới
- [ ] Train được model
- [ ] Cải thiện được MSE
- [ ] Phân tích được features quan trọng

---

## ✅ KIỂM TRA HIỂU BÀI

Trước khi sang bài tiếp theo, hãy đảm bảo bạn:

- [ ] Hiểu tại sao cần baseline models
- [ ] Implement được Linear Regression cho time series
- [ ] Hiểu được ARIMA(p,d,q) và cách chọn p,d,q
- [ ] Implement được ARIMA
- [ ] Hiểu được GARCH và khi nào dùng
- [ ] Implement được các naive methods
- [ ] So sánh được các baselines
- [ ] Làm được 3 bài tập thực hành

**Nếu chưa pass hết checklist, đọc lại phần tương ứng!**

---

## 📚 TÀI LIỆU THAM KHẢO

**Books:**
- "Forecasting: Principles and Practice" - Rob Hyndman
- "Time Series Analysis and Its Applications" - Shumway & Stoffer

**Papers:**
- "Forecasting with Exponential Smoothing" - Hyndman et al.
- "ARIMA Models and the Box-Jenkins Methodology" - Box & Jenkins

**Libraries:**
- `statsmodels`: ARIMA, SARIMAX
- `pmdarima`: Auto ARIMA
- `arch`: GARCH models

---

## 🚀 BƯỚC TIẾP THEO

Sau khi hoàn thành bài này, sang:
- `02_ML_MODELS.md` - XGBoost, LightGBM, Random Forest

**Chúc bạn học tốt! 🎓**
