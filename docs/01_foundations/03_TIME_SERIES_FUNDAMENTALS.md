# ⏰ TIME SERIES FUNDAMENTALS
## Hiểu đặc thù của dữ liệu chuỗi thời gian

---

## 📚 MỤC LỤC

1. [Time Series là gì?](#1-time-series-là-gì)
2. [Đặc điểm quan trọng](#2-đặc-điểm-quan-trọng)
3. [Components của Time Series](#3-components-của-time-series)
4. [Stationarity](#4-stationarity)
5. [Autocorrelation](#5-autocorrelation)
6. [Seasonality](#6-seasonality)
7. [Forecasting Horizons](#7-forecasting-horizons)
8. [Bài tập thực hành](#8-bài-tập-thực-hành)

---

## 1. TIME SERIES LÀ GÌ?

### 🤔 Định nghĩa đơn giản

> **Time Series = Dữ liệu theo thời gian**

Dữ liệu được thu thập theo thứ tự thời gian, mỗi điểm dữ liệu gắn với một thời điểm cụ thể.

### 📊 Ví dụ đời thường

**Time Series:**
- Giá cổ phiếu hàng ngày
- Nhiệt độ hàng giờ
- Doanh số bán hàng hàng tháng
- Nhịp tim mỗi giây

**KHÔNG phải Time Series:**
- Chiều cao của học sinh trong lớp
- Giá nhà ở các quận khác nhau
- Điểm thi của sinh viên

### 🎯 Ví dụ với FPT

```
Ngày        Giá đóng cửa
2024-01-01  100,000
2024-01-02  102,000  ← Phụ thuộc vào 01/01
2024-01-03  105,000  ← Phụ thuộc vào 01/02
2024-01-04  103,000  ← Phụ thuộc vào 01/03
...

→ Đây là TIME SERIES vì:
  - Có thứ tự thời gian
  - Giá trị hôm nay phụ thuộc vào hôm qua
```

---

## 2. ĐẶC ĐIỂM QUAN TRỌNG

### ⏰ 1. Temporal Ordering (Thứ tự thời gian)

**Đặc điểm:**
- Dữ liệu có thứ tự, KHÔNG THỂ đảo ngẫu nhiên
- Thứ tự quan trọng!

**Ví dụ:**
```
❌ SAI: Shuffle data
[2024-01-05, 2024-01-01, 2024-01-03, 2024-01-02]
→ Mất thứ tự thời gian!

✅ ĐÚNG: Giữ nguyên thứ tự
[2024-01-01, 2024-01-02, 2024-01-03, 2024-01-04, 2024-01-05]
→ Thứ tự đúng!
```

**Hệ quả:**
- KHÔNG thể dùng random train/test split
- PHẢI dùng sequential split (chia theo thời gian)

### 🔗 2. Temporal Dependence (Phụ thuộc thời gian)

**Đặc điểm:**
- Giá trị hôm nay phụ thuộc vào hôm qua
- Giá trị tương lai phụ thuộc vào quá khứ

**Ví dụ:**
```
Nếu FPT tăng 5 ngày liên tiếp:
→ Ngày thứ 6 có xu hướng tăng tiếp (momentum)
hoặc điều chỉnh giảm (overbought)

Nếu nhiệt độ hôm nay 30°C:
→ Ngày mai khó có thể 10°C (phụ thuộc vào hôm nay)
```

**Hệ quả:**
- Cần dùng models có "memory" (LSTM, GRU)
- Cần features từ quá khứ (lagged features)

### 📈 3. Trend (Xu hướng)

**Đặc điểm:**
- Có xu hướng tăng/giảm dài hạn

**Ví dụ:**
```
Giá FPT 2015-2024:
2015: 30,000
2020: 70,000  ← Xu hướng tăng
2024: 100,000

→ Trend: Tăng dần theo thời gian
```

### 🔄 4. Seasonality (Tính mùa vụ)

**Đặc điểm:**
- Lặp lại theo chu kỳ cố định

**Ví dụ:**
```
Doanh số bán lẻ:
- Tháng 12: Cao (Noel, Tết)
- Tháng 1-2: Thấp
→ Seasonality: Chu kỳ 12 tháng

Giá cổ phiếu:
- Thứ 2: Thường biến động mạnh (Monday effect)
- Thứ 6: Thường giảm (Friday effect)
→ Seasonality: Chu kỳ 5 ngày (tuần)
```

---

## 3. COMPONENTS CỦA TIME SERIES

### 📊 Phân tích thành phần

**Time Series = Trend + Seasonality + Cycle + Noise**

```
Y(t) = T(t) + S(t) + C(t) + ε(t)
       ↑      ↑      ↑      ↑
     Trend  Season Cycle  Noise
```

### 🎯 Ví dụ cụ thể với FPT

**1. Trend (T):**
```
Xu hướng dài hạn:
2015: 30K → 2024: 100K
→ Trend tăng ~10K/năm
```

**2. Seasonality (S):**
```
Chu kỳ lặp lại:
- Tháng 1-3: Thường tăng (báo cáo tài chính tốt)
- Tháng 7-9: Thường giảm (off-season)
→ Seasonality: Chu kỳ 12 tháng
```

**3. Cycle (C):**
```
Chu kỳ không đều:
- 2015-2018: Bull market (tăng)
- 2018-2020: Bear market (giảm)
- 2020-2021: Bull market (tăng)
→ Cycle: Không cố định, phụ thuộc kinh tế
```

**4. Noise (ε):**
```
Biến động ngẫu nhiên:
- Tin tức bất ngờ
- Thao túng giá
- Lỗi dữ liệu
→ Noise: Không dự đoán được
```

### 📈 Visualize Components

```
Price (Y)
  ↑
  │     ╱╲    ╱╲    ╱╲    ╱╲      ← Actual (Y)
  │    ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
  │   ╱    ╲╱    ╲╱    ╲╱    ╲
  │  ╱                          ╲  ← Trend (T)
  │ ╱____________________________╲
  └────────────────────────────────→ Time

Seasonality (S):  ╱╲╱╲╱╲╱╲  (lặp lại đều)
Cycle (C):        ╱‾‾‾╲___╱‾‾‾╲  (không đều)
Noise (ε):        ⋮⋮⋮⋮⋮⋮⋮⋮  (ngẫu nhiên)
```

---

## 4. STATIONARITY

### 🤔 Stationarity là gì?

> **Stationary = Tính chất thống kê không đổi theo thời gian**

**Đơn giản:**
- Mean (trung bình) không đổi
- Variance (phương sai) không đổi
- Covariance (hiệp phương sai) chỉ phụ thuộc vào khoảng cách, không phụ thuộc vào thời điểm

### 📊 Ví dụ

**STATIONARY (Tốt cho modeling):**
```
Returns (% thay đổi hàng ngày):
Mean ≈ 0%, Variance ≈ 2%

Day 1: +1.5%
Day 2: -0.8%
Day 3: +2.1%
...
Day 1000: +1.2%

→ Mean và Variance ổn định theo thời gian
```

**NON-STATIONARY (Khó modeling):**
```
Price (giá tuyệt đối):
2015: Mean = 30K, Variance = 5K
2020: Mean = 70K, Variance = 15K
2024: Mean = 100K, Variance = 25K

→ Mean và Variance tăng theo thời gian
```

### 🔧 Cách kiểm tra Stationarity

**Visual Test (Nhìn biểu đồ):**
```python
import matplotlib.pyplot as plt

# Plot price
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df['close'])
plt.title('Price (Non-Stationary)')

# Plot returns
plt.subplot(2, 1, 2)
plt.plot(df['return_1d'])
plt.title('Returns (Stationary)')
plt.show()
```

**Statistical Test (ADF Test):**
```python
from statsmodels.tsa.stattools import adfuller

# Test price
result = adfuller(df['close'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Nếu p-value < 0.05 → Stationary
# Nếu p-value > 0.05 → Non-Stationary
```

### 💡 Cách chuyển Non-Stationary → Stationary

**1. Differencing (Lấy sai phân):**
```python
# First difference
df['price_diff'] = df['close'].diff()

# Hoặc dùng returns
df['returns'] = df['close'].pct_change()
```

**2. Log Transform:**
```python
import numpy as np
df['log_price'] = np.log(df['close'])
df['log_returns'] = df['log_price'].diff()
```

**3. Detrending (Loại bỏ trend):**
```python
from scipy import signal
df['detrended'] = signal.detrend(df['close'])
```

### 🎯 Tại sao Stationarity quan trọng?

**Lý do:**
1. Nhiều models giả định data là stationary (ARIMA, GARCH)
2. Stationary data dễ dự đoán hơn
3. Statistical tests hoạt động tốt hơn trên stationary data

**Trong TechPulse:**
- Price: Non-stationary → Khó dự đoán trực tiếp
- Returns: Stationary → Dễ dự đoán hơn
- Features (MA, RSI): Gần stationary → Tốt cho ML

---

## 5. AUTOCORRELATION

### 🤔 Autocorrelation là gì?

> **Autocorrelation = Tương quan của chuỗi với chính nó ở các thời điểm khác nhau**

**Đơn giản:**
- Đo xem giá trị hôm nay có liên quan đến giá trị hôm qua không
- Đo xem giá trị hôm nay có liên quan đến giá trị 5 ngày trước không

### 📊 Ví dụ

**Positive Autocorrelation:**
```
Nếu hôm nay tăng → Ngày mai có xu hướng tăng
Nếu hôm nay giảm → Ngày mai có xu hướng giảm

Day 1: +2%
Day 2: +1.5%  ← Cùng dấu với Day 1
Day 3: +1.8%  ← Cùng dấu với Day 2
→ Positive autocorrelation (momentum)
```

**Negative Autocorrelation:**
```
Nếu hôm nay tăng → Ngày mai có xu hướng giảm
Nếu hôm nay giảm → Ngày mai có xu hướng tăng

Day 1: +2%
Day 2: -1.5%  ← Ngược dấu với Day 1
Day 3: +1.8%  ← Ngược dấu với Day 2
→ Negative autocorrelation (mean reversion)
```

**No Autocorrelation:**
```
Hôm nay tăng/giảm không ảnh hưởng đến ngày mai

Day 1: +2%
Day 2: -0.5%  ← Ngẫu nhiên
Day 3: +1.2%  ← Ngẫu nhiên
→ No autocorrelation (random walk)
```

### 🔧 Cách tính Autocorrelation

**ACF (Autocorrelation Function):**
```python
from statsmodels.graphics.tsaplots import plot_acf

# Plot ACF
plot_acf(df['return_1d'].dropna(), lags=20)
plt.title('Autocorrelation Function')
plt.show()
```

**Interpretation:**
```
Lag 1:  Correlation với 1 ngày trước
Lag 5:  Correlation với 5 ngày trước
Lag 20: Correlation với 20 ngày trước

Nếu bar vượt ra ngoài vùng xanh:
→ Có autocorrelation có ý nghĩa thống kê
```

### 💡 Ý nghĩa trong Forecasting

**High Autocorrelation:**
- Quá khứ ảnh hưởng mạnh đến tương lai
- Dễ dự đoán hơn
- Nên dùng models có "memory" (LSTM, ARIMA)

**Low Autocorrelation:**
- Quá khứ ảnh hưởng yếu đến tương lai
- Khó dự đoán (gần random walk)
- Có thể dùng simple models

---

## 6. SEASONALITY

### 🤔 Seasonality là gì?

> **Seasonality = Pattern lặp lại theo chu kỳ cố định**

### 📊 Các loại Seasonality

**1. Daily Seasonality:**
```
Trong ngày:
- 9:00-10:00: Volume cao (mở cửa)
- 11:00-13:00: Volume thấp (nghỉ trưa)
- 14:00-15:00: Volume cao (đóng cửa)
→ Chu kỳ: 1 ngày
```

**2. Weekly Seasonality:**
```
Trong tuần:
- Thứ 2: Biến động mạnh (Monday effect)
- Thứ 3-5: Ổn định
- Thứ 6: Giảm (Friday effect)
→ Chu kỳ: 5 ngày (tuần giao dịch)
```

**3. Monthly Seasonality:**
```
Trong tháng:
- Đầu tháng: Tăng (lương về, tiền đầu tư)
- Giữa tháng: Ổn định
- Cuối tháng: Giảm (cần tiền chi tiêu)
→ Chu kỳ: 1 tháng
```

**4. Yearly Seasonality:**
```
Trong năm:
- Q1: Tăng (báo cáo tài chính tốt)
- Q2: Ổn định
- Q3: Giảm (off-season)
- Q4: Tăng (kỳ vọng năm mới)
→ Chu kỳ: 12 tháng
```

### 🔧 Cách phát hiện Seasonality

**1. Visual Inspection:**
```python
# Plot by month
df['month'] = df['date'].dt.month
df.groupby('month')['return_1d'].mean().plot(kind='bar')
plt.title('Average Returns by Month')
plt.show()
```

**2. Seasonal Decomposition:**
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose
result = seasonal_decompose(df['close'], model='multiplicative', period=252)  # 252 = trading days/year

# Plot
result.plot()
plt.show()
```

**3. Fourier Transform:**
```python
from scipy.fft import fft, fftfreq

# FFT
fft_values = fft(df['close'].values)
frequencies = fftfreq(len(df), d=1)  # d=1 day

# Plot
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_values)[:len(frequencies)//2])
plt.title('Frequency Spectrum')
plt.show()
```

### 💡 Cách xử lý Seasonality

**1. Seasonal Differencing:**
```python
# Remove yearly seasonality
df['close_deseason'] = df['close'] - df['close'].shift(252)
```

**2. Seasonal Features:**
```python
# Add seasonal features
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
```

**3. Seasonal Models:**
```python
# SARIMA (Seasonal ARIMA)
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df['close'], order=(1,1,1), seasonal_order=(1,1,1,12))
```

---

## 7. FORECASTING HORIZONS

### 🎯 Các loại Forecasting

**1. One-Step-Ahead (1 bước):**
```
Dùng: [t-10, t-9, ..., t-1, t]
Dự đoán: t+1

Ví dụ:
Dùng 10 ngày gần nhất → Dự đoán ngày mai
```

**2. Multi-Step-Ahead (nhiều bước):**
```
Dùng: [t-10, t-9, ..., t-1, t]
Dự đoán: [t+1, t+2, t+3, t+4, t+5]

Ví dụ:
Dùng 10 ngày gần nhất → Dự đoán 5 ngày tới
```

**3. Direct Multi-Step:**
```
Train 5 models riêng:
- Model 1: Dự đoán t+1
- Model 2: Dự đoán t+2
- Model 3: Dự đoán t+3
- Model 4: Dự đoán t+4
- Model 5: Dự đoán t+5
```

**4. Recursive Multi-Step:**
```
Train 1 model:
- Dự đoán t+1
- Dùng t+1 (predicted) để dự đoán t+2
- Dùng t+2 (predicted) để dự đoán t+3
- ...
```

### 📊 So sánh

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| **One-Step** | Chính xác nhất | Chỉ dự đoán 1 bước |
| **Direct Multi-Step** | Mỗi horizon có model riêng | Cần train nhiều models |
| **Recursive Multi-Step** | Chỉ cần 1 model | Lỗi tích lũy theo thời gian |

### 💡 Trong TechPulse

**Khuyến nghị:**
1. **Short-term (1-5 ngày):** One-step hoặc Direct multi-step
2. **Medium-term (1-4 tuần):** Direct multi-step
3. **Long-term (1-3 tháng):** Khó, cần thêm external data (news, macro)

---

## 8. BÀI TẬP THỰC HÀNH

### 🎯 Bài tập 1: Phân tích Components

**Đề bài:**
Phân tích giá FPT thành Trend + Seasonality + Residual

**Code:**
```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/features/vn30/FPT.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Decompose
result = seasonal_decompose(df['close'], model='multiplicative', period=252)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
result.observed.plot(ax=axes[0], title='Original')
result.trend.plot(ax=axes[1], title='Trend')
result.seasonal.plot(ax=axes[2], title='Seasonality')
result.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()
```

**Kiểm tra:**
- [ ] Vẽ được 4 biểu đồ
- [ ] Giải thích được trend của FPT
- [ ] Nhận diện được seasonality (nếu có)
- [ ] Hiểu được residual là gì

---

### 🎯 Bài tập 2: Kiểm tra Stationarity

**Đề bài:**
Kiểm tra xem Price và Returns có stationary không

**Code:**
```python
from statsmodels.tsa.stattools import adfuller

# Test price
result_price = adfuller(df['close'].dropna())
print("Price:")
print(f"  ADF Statistic: {result_price[0]:.4f}")
print(f"  p-value: {result_price[1]:.4f}")
print(f"  Stationary: {'Yes' if result_price[1] < 0.05 else 'No'}")

# Test returns
result_returns = adfuller(df['return_1d'].dropna())
print("\nReturns:")
print(f"  ADF Statistic: {result_returns[0]:.4f}")
print(f"  p-value: {result_returns[1]:.4f}")
print(f"  Stationary: {'Yes' if result_returns[1] < 0.05 else 'No'}")
```

**Kiểm tra:**
- [ ] Chạy được ADF test
- [ ] Giải thích được p-value
- [ ] Kết luận đúng về stationarity
- [ ] Hiểu tại sao returns thường stationary hơn price

---

### 🎯 Bài tập 3: Phân tích Autocorrelation

**Đề bài:**
Vẽ ACF plot cho Returns và giải thích

**Code:**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ACF
plot_acf(df['return_1d'].dropna(), lags=20, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')

# PACF
plot_pacf(df['return_1d'].dropna(), lags=20, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()
```

**Kiểm tra:**
- [ ] Vẽ được ACF và PACF
- [ ] Giải thích được ý nghĩa của bars
- [ ] Nhận diện được significant lags
- [ ] Kết luận về autocorrelation của FPT returns

---

## ✅ KIỂM TRA HIỂU BÀI

Trước khi sang bài tiếp theo, hãy đảm bảo bạn:

- [ ] Giải thích được time series khác gì với dữ liệu thông thường
- [ ] Liệt kê được 4 đặc điểm quan trọng của time series
- [ ] Phân tích được components: Trend, Seasonality, Cycle, Noise
- [ ] Hiểu được stationarity và tại sao nó quan trọng
- [ ] Tính được autocorrelation và giải thích ý nghĩa
- [ ] Nhận diện được seasonality trong data
- [ ] Phân biệt được các loại forecasting horizons
- [ ] Làm được 3 bài tập thực hành

**Nếu chưa pass hết checklist, đọc lại phần tương ứng!**

---

## 📚 TÀI LIỆU THAM KHẢO

**Books:**
- "Forecasting: Principles and Practice" - Rob Hyndman
- "Time Series Analysis and Its Applications" - Shumway & Stoffer

**Online Courses:**
- Coursera: Practical Time Series Analysis
- DataCamp: Time Series with Python

**Papers:**
- "Time Series Analysis: Forecasting and Control" - Box & Jenkins

---

## 🚀 BƯỚC TIẾP THEO

Sau khi hoàn thành bài này, sang:
- `02_modeling/01_BASELINE_MODELS.md` - Implement baseline models

**Chúc bạn học tốt! 🎓**
