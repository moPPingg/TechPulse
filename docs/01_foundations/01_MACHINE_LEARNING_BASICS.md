# 🎓 MACHINE LEARNING CƠ BẢN CHO TIME SERIES
## Học để hiểu - Không phải để nhớ

---

## 📚 MỤC LỤC

1. [Machine Learning là gì?](#1-machine-learning-là-gì)
2. [Supervised Learning](#2-supervised-learning)
3. [Regression vs Classification](#3-regression-vs-classification)
4. [Train/Test Split](#4-traintest-split)
5. [Overfitting vs Underfitting](#5-overfitting-vs-underfitting)
6. [Metrics đánh giá](#6-metrics-đánh-giá)
7. [Bài tập thực hành](#7-bài-tập-thực-hành)

---

## 1. MACHINE LEARNING LÀ GÌ?

### 🤔 Tình huống đời thường

**Bạn muốn dự đoán giá cổ phiếu ngày mai:**

**Cách truyền thống (lập trình thông thường):**
```
Nếu giá hôm nay > giá hôm qua:
    → Ngày mai tăng
Nếu không:
    → Ngày mai giảm
```
❌ **Vấn đề:** Quá đơn giản, không chính xác

**Cách Machine Learning:**
```
1. Cho máy xem 10 năm dữ liệu lịch sử
2. Máy tự học pattern (quy luật)
3. Máy dự đoán dựa trên pattern đã học
```
✅ **Ưu điểm:** Máy tự học, không cần viết rules phức tạp

### 📖 Định nghĩa đơn giản

> **Machine Learning = Dạy máy học từ dữ liệu**

Thay vì bạn nói cho máy "làm thế này, làm thế kia", bạn cho máy xem nhiều ví dụ, máy tự học cách làm.

### 🎯 Ví dụ cụ thể

**Bài toán:** Dự đoán giá FPT ngày mai

**Input (X):**
- Giá hôm nay: 100,000
- Giá hôm qua: 98,000
- Volume hôm nay: 1,500,000
- RSI: 65
- MACD: 0.5

**Output (y):**
- Giá ngày mai: 102,000 (dự đoán)

**ML làm gì?**
```
ML model học từ 10 năm dữ liệu:
"Khi RSI > 60 và MACD > 0 và giá tăng 2 ngày liên tiếp
 → Ngày mai thường tăng thêm 1-2%"
```

---

## 2. SUPERVISED LEARNING

### 🎓 Học có giám sát

**Giống như học ở trường:**
- Thầy cho bài tập (X) và đáp án (y)
- Học sinh làm bài, so sánh với đáp án
- Sai → sửa, đúng → nhớ
- Lặp lại cho đến khi học sinh làm đúng

**Trong ML:**
- Bạn cho máy dữ liệu (X) và kết quả đúng (y)
- Máy dự đoán, so sánh với kết quả đúng
- Sai → điều chỉnh model
- Đúng → nhớ pattern
- Lặp lại cho đến khi máy dự đoán tốt

### 📊 Ví dụ với dữ liệu FPT

**Dữ liệu training (máy học từ đây):**

| Ngày | Close (X) | MA20 (X) | RSI (X) | Close ngày mai (y) |
|------|-----------|----------|---------|-------------------|
| 1/1  | 100       | 95       | 50      | 102 ✅ (đáp án)   |
| 1/2  | 102       | 96       | 55      | 105 ✅            |
| 1/3  | 105       | 97       | 60      | 103 ✅            |
| ...  | ...       | ...      | ...     | ...               |

**Máy học:**
```
Lần 1: Dự đoán 1/1 → 98 (sai, đáp án là 102)
       → Điều chỉnh model

Lần 2: Dự đoán 1/1 → 101 (gần hơn!)
       → Điều chỉnh tiếp

Lần 3: Dự đoán 1/1 → 102 (đúng!)
       → Nhớ pattern này
```

### 🔑 Công thức tổng quát

```
Supervised Learning:
- Input: X (features)
- Output: y (target/label)
- Goal: Học hàm f sao cho f(X) ≈ y

f(X) = y
↑      ↑
Model  Kết quả thực tế
```

---

## 3. REGRESSION VS CLASSIFICATION

### 🎯 Phân biệt 2 loại bài toán

**REGRESSION (Hồi quy):**
- Dự đoán **SỐ LIÊN TỤC**
- Ví dụ: Dự đoán giá cổ phiếu (100.5, 102.3, 98.7, ...)

**CLASSIFICATION (Phân loại):**
- Dự đoán **NHÃN RỜI RẠC**
- Ví dụ: Dự đoán tăng/giảm (Tăng, Giảm)

### 📊 Ví dụ cụ thể

**Bài toán 1: Dự đoán giá FPT ngày mai**
```
Input:  close=100, ma20=95, rsi=60
Output: 102.5 (số liên tục)
→ REGRESSION
```

**Bài toán 2: Dự đoán FPT tăng hay giảm**
```
Input:  close=100, ma20=95, rsi=60
Output: "Tăng" (nhãn rời rạc)
→ CLASSIFICATION
```

### 🔍 Dự án TechPulse dùng gì?

**Chủ yếu: REGRESSION**
- Dự đoán giá cụ thể: 102,000 VNĐ
- Dự đoán return: +2.5%

**Có thể dùng: CLASSIFICATION**
- Dự đoán tăng/giảm (binary)
- Dự đoán mức độ: Tăng mạnh/Tăng nhẹ/Giảm nhẹ/Giảm mạnh

---

## 4. TRAIN/TEST SPLIT

### 🎓 Tại sao cần chia dữ liệu?

**Ví dụ học sinh:**
- Học từ sách giáo khoa (training data)
- Thi bài mới chưa từng thấy (test data)
- Nếu chỉ học vẹt sách → thi bài mới sẽ kém

**Trong ML:**
- Train trên dữ liệu cũ (2015-2023)
- Test trên dữ liệu mới (2024)
- Nếu model chỉ nhớ training data → test kém (overfitting)

### 📊 Cách chia dữ liệu

**Quy tắc chung:**
```
Training set: 70-80%  → Máy học từ đây
Test set:     20-30%  → Đánh giá model
```

**Với Time Series (QUAN TRỌNG!):**
```
❌ SAI: Chia ngẫu nhiên
   [2015][2020][2018][2023] → Training
   [2019][2021][2017][2024] → Test
   (Lý do: Không thể dùng tương lai dự đoán quá khứ!)

✅ ĐÚNG: Chia theo thời gian
   [2015][2016][2017][2018][2019][2020][2021][2022] → Training
   [2023][2024] → Test
   (Lý do: Giống thực tế - dùng quá khứ dự đoán tương lai)
```

### 🔧 Cách implement

**Bước 1: Sắp xếp theo thời gian**
```python
# Giả sử df có cột 'date'
df = df.sort_values('date')
```

**Bước 2: Chia 80/20**
```python
# Tính điểm chia
split_idx = int(len(df) * 0.8)

# Chia data
train_df = df[:split_idx]   # 80% đầu
test_df = df[split_idx:]     # 20% cuối
```

**Bước 3: Tách X và y**
```python
# Features (X)
X_train = train_df[['close', 'ma20', 'rsi', 'macd']]
X_test = test_df[['close', 'ma20', 'rsi', 'macd']]

# Target (y) - giá ngày mai
y_train = train_df['close'].shift(-1)  # Shift để lấy giá ngày mai
y_test = test_df['close'].shift(-1)
```

---

## 5. OVERFITTING VS UNDERFITTING

### 🎯 Hiểu qua ví dụ học sinh

**UNDERFITTING (Học kém):**
```
Học sinh chỉ học: "Nếu giá tăng → ngày mai tăng"
→ Quá đơn giản, không nắm bắt được pattern phức tạp
→ Điểm thấp cả training lẫn test
```

**OVERFITTING (Học vẹt):**
```
Học sinh nhớ từng câu trong sách:
"Ngày 1/1/2020 giá 100 → ngày 2/1 giá 102"
"Ngày 2/1/2020 giá 102 → ngày 3/1 giá 105"
→ Nhớ chi tiết quá, không tổng quát
→ Điểm cao training, điểm thấp test
```

**GOOD FIT (Học tốt):**
```
Học sinh hiểu pattern:
"Khi RSI > 70 và volume tăng đột biến → thường giảm"
→ Tổng quát hóa tốt
→ Điểm cao cả training lẫn test
```

### 📊 Biểu đồ minh họa

```
Error
  ↑
  │     Underfitting        Good Fit      Overfitting
  │         ╱╲                 ╱╲             ╱╲
  │        ╱  ╲               ╱  ╲           ╱  ╲
  │       ╱    ╲             ╱    ╲         ╱    ╲
  │      ╱      ╲           ╱      ╲       ╱      ╲
  │     ╱        ╲         ╱        ╲     ╱        ╲___
  │    ╱          ╲       ╱          ╲   ╱            Test Error
  │___╱____________╲_____╱____________╲_╱_____________
  │                                    ╲
  │                                     ╲___Training Error
  └────────────────────────────────────────────────→
                Model Complexity
```

### 🔍 Cách phát hiện

**Dấu hiệu Underfitting:**
- Training error cao
- Test error cao
- Model quá đơn giản

**Dấu hiệu Overfitting:**
- Training error rất thấp (~0)
- Test error cao
- Chênh lệch lớn giữa train và test

**Dấu hiệu Good Fit:**
- Training error thấp
- Test error thấp
- Chênh lệch nhỏ giữa train và test

### 💡 Cách khắc phục

**Underfitting → Tăng độ phức tạp:**
- Thêm features
- Dùng model phức tạp hơn (LSTM thay vì Linear)
- Tăng số epochs training

**Overfitting → Giảm độ phức tạp:**
- Regularization (L1, L2)
- Dropout (với Neural Networks)
- Early stopping
- Thêm dữ liệu training
- Giảm số features

---

## 6. METRICS ĐÁNH GIÁ

### 📊 Tại sao cần metrics?

**Không có metrics:**
```
Bạn: "Model của tôi tốt!"
Reviewer: "Tốt như thế nào? Bằng chứng?"
Bạn: "Ừm... nhìn có vẻ tốt..."
→ Không thuyết phục!
```

**Có metrics:**
```
Bạn: "Model của tôi có MSE = 0.5, MAE = 0.3"
Reviewer: "So với baseline?"
Bạn: "Baseline MSE = 1.2, tôi giảm được 58%"
→ Thuyết phục!
```

### 🎯 Các metrics quan trọng

#### **1. MSE (Mean Squared Error)**

**Công thức:**
```
MSE = (1/n) × Σ(y_true - y_pred)²

Ví dụ:
y_true = [100, 102, 105]
y_pred = [98,  103, 104]
error  = [2,   -1,  1]
squared= [4,   1,   1]
MSE    = (4 + 1 + 1) / 3 = 2.0
```

**Ý nghĩa:**
- Đo "sai số bình phương trung bình"
- Phạt nặng các lỗi lớn (vì bình phương)
- Đơn vị: (đơn vị của y)²

**Khi nào dùng:**
- Khi muốn phạt nặng outliers
- Khi sai số lớn quan trọng hơn sai số nhỏ

#### **2. MAE (Mean Absolute Error)**

**Công thức:**
```
MAE = (1/n) × Σ|y_true - y_pred|

Ví dụ:
y_true = [100, 102, 105]
y_pred = [98,  103, 104]
error  = [2,   -1,  1]
abs    = [2,   1,   1]
MAE    = (2 + 1 + 1) / 3 = 1.33
```

**Ý nghĩa:**
- Đo "sai số tuyệt đối trung bình"
- Phạt đều các lỗi (không bình phương)
- Đơn vị: đơn vị của y

**Khi nào dùng:**
- Khi muốn đối xử công bằng với mọi lỗi
- Dễ interpret hơn MSE

#### **3. RMSE (Root Mean Squared Error)**

**Công thức:**
```
RMSE = √MSE

Ví dụ:
MSE = 2.0
RMSE = √2.0 = 1.41
```

**Ý nghĩa:**
- Giống MSE nhưng đơn vị giống y
- Dễ interpret hơn MSE

#### **4. MAPE (Mean Absolute Percentage Error)**

**Công thức:**
```
MAPE = (1/n) × Σ|((y_true - y_pred) / y_true)| × 100%

Ví dụ:
y_true = [100, 102, 105]
y_pred = [98,  103, 104]
error% = [2%,  -0.98%, 0.95%]
abs%   = [2%,  0.98%,  0.95%]
MAPE   = (2 + 0.98 + 0.95) / 3 = 1.31%
```

**Ý nghĩa:**
- Đo "sai số phần trăm trung bình"
- Không phụ thuộc vào scale của y
- Đơn vị: %

**Khi nào dùng:**
- Khi muốn so sánh models trên datasets khác nhau
- Khi muốn metric dễ hiểu (%)

### 📊 So sánh các metrics

| Metric | Ưu điểm | Nhược điểm | Khi nào dùng |
|--------|---------|------------|--------------|
| **MSE** | Phạt nặng outliers | Khó interpret, đơn vị lạ | Khi outliers quan trọng |
| **MAE** | Dễ hiểu, đơn vị rõ | Không phạt nặng outliers | Khi muốn metric đơn giản |
| **RMSE** | Dễ hiểu hơn MSE | Vẫn phạt nặng outliers | Khi muốn MSE nhưng dễ đọc |
| **MAPE** | Scale-free, dễ so sánh | Lỗi khi y_true = 0 | Khi so sánh nhiều datasets |

### 💡 Metrics nào cho TechPulse?

**Khuyến nghị:**
1. **MAE** - Metric chính (dễ hiểu, ổn định)
2. **RMSE** - Metric phụ (phạt outliers)
3. **MAPE** - So sánh giữa các mã cổ phiếu

---

## 7. BÀI TẬP THỰC HÀNH

### 🎯 Bài tập 1: Hiểu Train/Test Split

**Đề bài:**
Bạn có dữ liệu FPT từ 2020-2024 (1,250 dòng). Hãy:
1. Chia 80/20 train/test
2. Tính số dòng mỗi set
3. Xác định khoảng thời gian mỗi set

**Gợi ý:**
```python
# Bước 1: Load data
df = pd.read_csv('data/features/vn30/FPT.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Bước 2: Chia
split_idx = int(len(df) * 0.8)
train_df = df[:split_idx]
test_df = df[split_idx:]

# Bước 3: In thông tin
print(f"Training: {len(train_df)} dòng, từ {train_df['date'].min()} đến {train_df['date'].max()}")
print(f"Test: {len(test_df)} dòng, từ {test_df['date'].min()} đến {test_df['date'].max()}")
```

**Kiểm tra:**
- [ ] Training set có ~1,000 dòng (80%)
- [ ] Test set có ~250 dòng (20%)
- [ ] Training set đến trước test set theo thời gian

---

### 🎯 Bài tập 2: Tính Metrics

**Đề bài:**
Cho predictions và actual values, tính MSE, MAE, RMSE, MAPE

```python
y_true = [100, 105, 102, 108, 110]
y_pred = [98,  107, 101, 110, 108]
```

**Gợi ý:**
```python
import numpy as np

# MSE
mse = np.mean((y_true - y_pred) ** 2)

# MAE
mae = np.mean(np.abs(y_true - y_pred))

# RMSE
rmse = np.sqrt(mse)

# MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
```

**Đáp án:**
- MSE: 4.4
- MAE: 2.0
- RMSE: 2.1
- MAPE: 1.94%

**Kiểm tra:**
- [ ] Tính được MSE đúng
- [ ] Tính được MAE đúng
- [ ] Hiểu tại sao MSE > MAE
- [ ] Giải thích được MAPE = 1.94% nghĩa là gì

---

### 🎯 Bài tập 3: Implement Linear Regression

**Đề bài:**
Dùng Linear Regression dự đoán giá FPT ngày mai

**Bước 1: Chuẩn bị data**
```python
# Load features data
df = pd.read_csv('data/features/vn30/FPT.csv')

# Chọn features
features = ['close', 'ma_20', 'rsi_14', 'macd']
X = df[features]

# Target: giá ngày mai
y = df['close'].shift(-1)

# Drop NaN
df_clean = pd.concat([X, y], axis=1).dropna()
X = df_clean[features]
y = df_clean['close']

# Train/test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

**Bước 2: Train model**
```python
from sklearn.linear_model import LinearRegression

# Tạo model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

**Bước 3: Evaluate**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
```

**Bước 4: Visualize**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.legend()
plt.title('FPT Price Prediction - Linear Regression')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()
```

**Kiểm tra:**
- [ ] Model train thành công
- [ ] Tính được metrics
- [ ] Vẽ được biểu đồ
- [ ] Giải thích được kết quả

---

## ✅ KIỂM TRA HIỂU BÀI

Trước khi sang bài tiếp theo, hãy đảm bảo bạn:

- [ ] Giải thích được Machine Learning là gì bằng lời của mình
- [ ] Phân biệt được Supervised vs Unsupervised Learning
- [ ] Phân biệt được Regression vs Classification
- [ ] Hiểu tại sao phải chia train/test với time series
- [ ] Phân biệt được Overfitting vs Underfitting
- [ ] Tính được MSE, MAE, RMSE, MAPE bằng tay
- [ ] Implement được Linear Regression cho FPT
- [ ] Giải thích được kết quả dự báo

**Nếu chưa pass hết checklist, đọc lại phần tương ứng!**

---

## 📚 TÀI LIỆU THAM KHẢO

**Videos (YouTube):**
- StatQuest: Machine Learning Fundamentals
- 3Blue1Brown: Neural Networks series
- Krish Naik: Machine Learning Playlist

**Courses:**
- Andrew Ng - Machine Learning (Coursera)
- Fast.ai - Practical Deep Learning

**Books:**
- "Hands-On Machine Learning" - Aurélien Géron
- "Introduction to Statistical Learning" - James et al.

---

## 🚀 BƯỚC TIẾP THEO

Sau khi hoàn thành bài này, sang:
- `03_TIME_SERIES_FUNDAMENTALS.md` - Hiểu đặc thù của time series

**Chúc bạn học tốt! 🎓**
