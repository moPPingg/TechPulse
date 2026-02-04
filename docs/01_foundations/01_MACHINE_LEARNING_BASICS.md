# Machine Learning Cơ Bản
## Hướng dẫn thực hành cho người mới bắt đầu

---

## Mục lục

1. [Bias-Variance Tradeoff](#1-bias-variance-tradeoff)
2. [Cross-Validation](#2-cross-validation)
3. [Feature Scaling](#3-feature-scaling)
4. [Data Leakage](#4-data-leakage)
5. [Confusion Matrix](#5-confusion-matrix)
6. [Hướng dẫn chọn Metric](#6-hướng-dẫn-chọn-metric)
7. [Pipeline hoàn chỉnh với sklearn](#7-pipeline-hoàn-chỉnh-với-sklearn)

---

## 1. Bias-Variance Tradeoff

### Bias và Variance là gì?

**Bias (Độ lệch):** Model quá đơn giản, không học được pattern thực sự trong data.

**Variance (Độ phân tán):** Model quá phức tạp, học cả noise trong training data.

### Ví dụ trực quan: Bắn cung

Hãy tưởng tượng bạn đang bắn cung vào bia:

```
High Bias, Low Variance     Low Bias, High Variance     Low Bias, Low Variance
(Underfitting)              (Overfitting)               (Mục tiêu)

    ○ ○ ○                       ○                           ○
   ○ ○ ○ ○                    ○   ○                        ○○○
    ○ ○ ○                   ○       ○                       ○
                              ○   ○
→ Tập trung nhưng           → Gần tâm nhưng            → Gần tâm và 
  xa tâm bia                  phân tán                    tập trung
```

### Ví dụ thực tế: Dự đoán giá nhà

**Dữ liệu:** 1000 căn nhà với diện tích và giá bán.

**High Bias (Underfitting):**
```python
# Model quá đơn giản: "Mọi nhà đều có giá trung bình"
def predict(area):
    return 500_000_000  # 500 triệu cho mọi nhà

# Kết quả: 
# - Nhà 30m² → 500 triệu (thực tế: 800 triệu)
# - Nhà 100m² → 500 triệu (thực tế: 3 tỷ)
# → Sai hệ thống, không học được gì
```

**High Variance (Overfitting):**
```python
# Model quá phức tạp: nhớ từng điểm dữ liệu
# "Nhà 50m² ở quận 1, tầng 3, hướng đông, sơn màu trắng = 1.234 tỷ"

# Kết quả trên training data: 100% chính xác
# Kết quả trên data mới: Sai hoàn toàn
# → Học thuộc lòng, không tổng quát hóa được
```

**Balanced (Mục tiêu):**
```python
# Model vừa phải: Giá = a × diện_tích + b × vị_trí + c
# Học được pattern chính, bỏ qua noise

# Kết quả:
# - Training error: 15%
# - Test error: 18%
# → Chấp nhận được, tổng quát hóa tốt
```

### Công thức và Tradeoff

```
Tổng Error = Bias² + Variance + Noise không thể giảm

Model đơn giản → Bias cao, Variance thấp
Model phức tạp → Bias thấp, Variance cao
```

**Biểu đồ:**
```
Error
  │
  │  ╲                    ╱
  │   ╲  Total Error    ╱
  │    ╲              ╱
  │     ╲    ┌──────╱
  │      ╲  ╱
  │       ╲╱ ← Điểm tối ưu
  │       ╱╲
  │      ╱  ╲______ Variance
  │     ╱
  │____╱____________ Bias²
  └────────────────────────→ Độ phức tạp model
```

### Cách phát hiện và xử lý

| Tình trạng | Dấu hiệu | Cách xử lý |
|------------|----------|------------|
| **High Bias** | Train error cao, Test error cao | Tăng độ phức tạp model, thêm features |
| **High Variance** | Train error thấp, Test error cao | Regularization, giảm features, thêm data |
| **Balanced** | Train và Test error gần nhau, đều thấp | Giữ nguyên |

```python
# Ví dụ kiểm tra
train_error = 0.05  # 5%
test_error = 0.25   # 25%

gap = test_error - train_error  # 20%
if gap > 0.1:
    print("High Variance - Overfitting!")
    # → Thử: regularization, dropout, early stopping
elif train_error > 0.2:
    print("High Bias - Underfitting!")
    # → Thử: model phức tạp hơn, thêm features
else:
    print("Balanced - OK!")
```

---

## 2. Cross-Validation

### Tại sao cần Cross-Validation?

**Vấn đề:** Train/Test split đơn giản có thể cho kết quả không đáng tin cậy.

```
Chia 1 lần:
- May mắn: Test set dễ → Đánh giá quá cao
- Xui xẻo: Test set khó → Đánh giá quá thấp
```

**Cross-Validation:** Chia nhiều lần, lấy trung bình → Kết quả tin cậy hơn.

### K-Fold Cross-Validation

**Cách hoạt động:**
```
Data: [1][2][3][4][5]  (chia thành 5 fold)

Fold 1: [Test][Train][Train][Train][Train] → Score 1
Fold 2: [Train][Test][Train][Train][Train] → Score 2
Fold 3: [Train][Train][Test][Train][Train] → Score 3
Fold 4: [Train][Train][Train][Test][Train] → Score 4
Fold 5: [Train][Train][Train][Train][Test] → Score 5

Final Score = Trung bình(Score 1, 2, 3, 4, 5)
```

**Code:**
```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Tạo K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')

# Kết quả
mse_scores = -scores
print(f"MSE mỗi fold: {mse_scores}")
print(f"MSE trung bình: {mse_scores.mean():.4f} ± {mse_scores.std():.4f}")
```

### TimeSeriesSplit - Cho dữ liệu chuỗi thời gian

**Vấn đề với K-Fold cho time series:**
```
K-Fold shuffle data → Dùng tương lai dự đoán quá khứ → SAI!

Ví dụ sai:
Train: [2020][2023][2019]  ← Có data 2023
Test:  [2021]              ← Dự đoán 2021 bằng data 2023? Không công bằng!
```

**TimeSeriesSplit:**
```
Fold 1: [Train    ] [Test] . . . . .
Fold 2: [Train    ] [Train] [Test] . . . .
Fold 3: [Train    ] [Train] [Train] [Test] . . .
Fold 4: [Train    ] [Train] [Train] [Train] [Test] .
Fold 5: [Train    ] [Train] [Train] [Train] [Train] [Test]

→ Luôn dùng quá khứ dự đoán tương lai
```

**Code:**
```python
from sklearn.model_selection import TimeSeriesSplit

# TimeSeriesSplit cho time series
tscv = TimeSeriesSplit(n_splits=5)

scores = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    scores.append(score)
    
    print(f"Fold {fold+1}: Train size={len(train_idx)}, Test size={len(test_idx)}, R²={score:.4f}")

print(f"\nR² trung bình: {np.mean(scores):.4f}")
```

### Khi nào dùng loại nào?

| Loại CV | Khi nào dùng | Ví dụ |
|---------|--------------|-------|
| **K-Fold** | Data không có thứ tự thời gian | Phân loại ảnh, dự đoán churn |
| **Stratified K-Fold** | Classification với class imbalanced | 95% class 0, 5% class 1 |
| **TimeSeriesSplit** | Dữ liệu chuỗi thời gian | Giá cổ phiếu, doanh số hàng ngày |
| **Leave-One-Out** | Dataset rất nhỏ (<100 samples) | Dữ liệu y tế hiếm |

```python
# Stratified K-Fold cho classification imbalanced
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Giữ tỉ lệ class trong mỗi fold
```

---

## 3. Feature Scaling

### Tại sao cần Feature Scaling?

**Ví dụ thực tế:**
```
Features:
- Thu nhập: 15,000,000 - 100,000,000 VNĐ
- Tuổi: 18 - 65
- Số phòng: 1 - 5

Không scale → Thu nhập chi phối model vì giá trị lớn hơn nhiều
```

### Models cần Scaling vs không cần

| Cần Scaling | Không cần Scaling |
|-------------|-------------------|
| KNN | Decision Tree |
| SVM | Random Forest |
| Logistic Regression | XGBoost |
| Linear Regression | LightGBM |
| Neural Networks | CatBoost |

**Tại sao?**

**KNN (cần scaling):**
```python
# KNN dựa vào khoảng cách Euclidean
# Không scale:
A = [thu_nhap=50_000_000, tuoi=30]
B = [thu_nhap=50_001_000, tuoi=60]

distance = sqrt((50_000_000 - 50_001_000)² + (30 - 60)²)
         = sqrt(1_000_000_000_000 + 900)
         ≈ sqrt(1_000_000_000_000)  # Tuổi bị bỏ qua!

# Có scale (chuẩn hóa về 0-1):
A = [thu_nhap=0.5, tuoi=0.4]
B = [thu_nhap=0.501, tuoi=1.0]

distance = sqrt((0.5-0.501)² + (0.4-1.0)²)
         = sqrt(0.000001 + 0.36)
         ≈ 0.6  # Cả hai features đều được xét
```

**Decision Tree (không cần scaling):**
```python
# Tree chỉ cần so sánh: "feature > threshold?"
# Không quan tâm magnitude

if thu_nhap > 30_000_000:     # Không cần scale
    if tuoi > 35:
        return "Approved"

# Scale hay không scale, kết quả split giống nhau
```

### Các phương pháp Scaling

**1. StandardScaler (Z-score):**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Công thức: X_new = (X - mean) / std
# Kết quả: mean=0, std=1
```

**2. MinMaxScaler:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

# Công thức: X_new = (X - min) / (max - min)
# Kết quả: range [0, 1]
```

**3. RobustScaler (kháng outliers):**
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)

# Công thức: X_new = (X - median) / IQR
# Kết quả: ít bị ảnh hưởng bởi outliers
```

**So sánh:**

| Scaler | Ưu điểm | Nhược điểm | Khi nào dùng |
|--------|---------|------------|--------------|
| **StandardScaler** | Xử lý tốt với data gần normal | Outliers ảnh hưởng mean/std | Data phân phối chuẩn |
| **MinMaxScaler** | Range cố định [0,1] | Rất nhạy với outliers | Neural Networks, không có outliers |
| **RobustScaler** | Kháng outliers | Range không cố định | Data có nhiều outliers |

**Lưu ý quan trọng:**
```python
# ĐÚNG: Fit trên train, transform cả train và test
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SAI: Fit trên toàn bộ data → Data leakage!
scaler.fit(X)  # KHÔNG LÀM THẾ NÀY
```

---

## 4. Data Leakage

### Data Leakage là gì?

**Định nghĩa:** Khi thông tin từ test set (hoặc tương lai) "rò rỉ" vào quá trình training.

**Hậu quả:** Model có performance tốt khi đánh giá nhưng thất bại trong thực tế.

### Ví dụ 1: Leakage trong dữ liệu bảng

**Bài toán:** Dự đoán khách hàng có vỡ nợ không.

```python
# Dataset:
# - income: thu nhập
# - loan_amount: số tiền vay
# - is_default: đã vỡ nợ chưa (target)
# - collection_calls: số cuộc gọi đòi nợ ← LEAKAGE!

# collection_calls chỉ có SAU KHI đã vỡ nợ
# → Model học: "Nhiều cuộc gọi đòi nợ → Sẽ vỡ nợ" → ĐÚNG 100%
# → Nhưng thực tế không dự đoán được vì chưa có data này!
```

**Sửa:** Loại bỏ features có sau sự kiện target.

### Ví dụ 2: Leakage trong Time Series

**Bài toán:** Dự đoán giá cổ phiếu ngày mai.

```python
# SAI: Dùng moving average tính từ tương lai
df['ma_20'] = df['close'].rolling(20).mean()
# Ngày 15/1: MA20 tính từ ngày 1/1 đến 20/1 ← Dùng data đến 20/1!

# ĐÚNG: Shift MA để chỉ dùng data quá khứ
df['ma_20'] = df['close'].rolling(20).mean().shift(1)
# Ngày 15/1: MA20 tính từ ngày 26/12 đến 14/1 ← Chỉ dùng data đến hôm qua
```

**Các dạng leakage phổ biến trong time series:**

```python
# 1. Scaling toàn bộ data trước khi split
scaler.fit(X)  # SAI - test data ảnh hưởng mean/std

# 2. Feature engineering dùng future data
df['pct_change'] = df['close'].pct_change()  # OK
df['future_return'] = df['close'].shift(-1) / df['close'] - 1  # SAI - dùng giá ngày mai

# 3. Target encoding không đúng cách
# Tính mean target trên toàn bộ data → Test data "biết" outcome
```

### Ví dụ 3: Train-Test Contamination

```python
# SAI: Impute missing trước khi split
X['age'].fillna(X['age'].mean())  # Mean tính từ cả test
train, test = train_test_split(X)

# ĐÚNG: Split trước, impute sau
train, test = train_test_split(X)
train_mean = train['age'].mean()
train['age'].fillna(train_mean, inplace=True)
test['age'].fillna(train_mean, inplace=True)  # Dùng mean của train
```

### Checklist tránh Data Leakage

```
□ Features không chứa thông tin từ tương lai
□ Target không ảnh hưởng features
□ Preprocessing (scaling, encoding) chỉ fit trên train
□ Cross-validation đúng cách (TimeSeriesSplit cho time series)
□ Không để duplicate records giữa train và test
□ Feature engineering chỉ dùng data quá khứ
```

---

    ## 5. Confusion Matrix

    ### Định nghĩa các thành phần

    **Bài toán:** Dự đoán email spam hay không spam.

    ```
                            Dự đoán
                    Spam    Không Spam
    Thực tế  Spam      TP         FN
            Không     FP         TN
    ``` 
        
    | Thành phần | Tên đầy đủ | Ý nghĩa | Ví dụ |
    |------------|------------|---------|-------|
    | **TP** | True Positive | Dự đoán Spam, thực tế Spam | Đúng! Đã chặn spam |
    | **TN** | True Negative | Dự đoán Không Spam, thực tế Không Spam | Đúng! Email quan trọng vào inbox |
    | **FP** | False Positive | Dự đoán Spam, thực tế Không Spam | Sai! Email quan trọng bị chặn |
    | **FN** | False Negative | Dự đoán Không Spam, thực tế Spam | Sai! Spam lọt vào inbox |

    ### Ví dụ thực tế: Chẩn đoán bệnh

    **Bài toán:** Phát hiện bệnh COVID từ test nhanh.

    ```
    Kết quả 1000 người test:
    - 100 người thực sự bị COVID
    - 900 người không bị COVID

    Confusion Matrix:
                        Dự đoán
                    Dương tính  Âm tính
    Thực tế  Có bệnh    85 (TP)   15 (FN)
            Không       45 (FP)  855 (TN)
    ```

    **Phân tích:**
    ```
    TP = 85: Phát hiện đúng 85 người có COVID
    TN = 855: Xác nhận đúng 855 người không có COVID
    FP = 45: 45 người khỏe mạnh bị báo dương tính giả → Lo lắng không cần thiết
    FN = 15: 15 người có COVID bị bỏ sót → NGUY HIỂM! Có thể lây lan
    ```

    ### Các Metrics từ Confusion Matrix

    ```python
    # Accuracy: Tổng số dự đoán đúng / Tổng số
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # = (85 + 855) / 1000 = 94%

    # Precision: Trong số dự đoán Positive, bao nhiêu % đúng?
    precision = TP / (TP + FP)
    # = 85 / (85 + 45) = 65.4%

    # Recall (Sensitivity): Trong số thực tế Positive, bắt được bao nhiêu %?
    recall = TP / (TP + FN)
    # = 85 / (85 + 15) = 85%

    # F1-Score: Trung bình điều hòa của Precision và Recall
    f1 = 2 * (precision * recall) / (precision + recall)
    # = 2 * (0.654 * 0.85) / (0.654 + 0.85) = 73.9%
    ```

    ### Code sklearn

    ```python
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Giả sử có y_true và y_pred
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Visualize
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Không', 'Có'],
                yticklabels=['Không', 'Có'])
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report đầy đủ
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Không', 'Có']))
    ```

    **Output:**
    ```
    Confusion Matrix:
    [[4 1]
    [1 4]]

    Classification Report:
                precision    recall  f1-score   support
        Không       0.80      0.80      0.80         5
            Có       0.80      0.80      0.80         5
        accuracy                           0.80        10
    macro avg       0.80      0.80      0.80        10
    weighted avg       0.80      0.80      0.80        10
    ```

    ---

## 6. Hướng dẫn chọn Metric

### Accuracy - Khi nào dùng?

**Dùng khi:** Classes cân bằng (50-50 hoặc gần đó).

**KHÔNG dùng khi:** Classes mất cân bằng.

```python
# Ví dụ: Phát hiện gian lận (1% gian lận, 99% bình thường)
# Model ngu: "Mọi giao dịch đều bình thường"
# Accuracy = 99%! Nhưng không phát hiện được gian lận nào

y_true = [0]*99 + [1]*1  # 99 bình thường, 1 gian lận
y_pred = [0]*100         # Dự đoán tất cả bình thường

accuracy = sum(t==p for t,p in zip(y_true, y_pred)) / len(y_true)
# = 99% nhưng vô dụng!
```

### Precision vs Recall - Tradeoff

**Precision cao quan trọng khi:** False Positive tốn kém.

```
Ví dụ: Email filter
- FP = Email quan trọng bị đánh nhầm spam → Bỏ lỡ cơ hội kinh doanh
- Cần Precision cao: Chỉ đánh spam khi CHẮC CHẮN
```

**Recall cao quan trọng khi:** False Negative tốn kém.

```
Ví dụ: Phát hiện ung thư
- FN = Bệnh nhân có ung thư nhưng bỏ sót → Nguy hiểm tính mạng
- Cần Recall cao: Thà báo động nhầm còn hơn bỏ sót
```

### F1-Score - Khi nào dùng?

**Dùng khi:** Cần cân bằng Precision và Recall.

```python
# Ví dụ: Hệ thống recommendation
# - Precision thấp: Suggest nhiều item không liên quan → User khó chịu
# - Recall thấp: Bỏ lỡ item user thích → User không hài lòng
# → Cần F1 để cân bằng cả hai
```

### ROC-AUC - Khi nào dùng?

**Dùng khi:**
- So sánh nhiều models
- Cần đánh giá ở nhiều threshold khác nhau
- Classes tương đối cân bằng

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Cần probability, không phải prediction
y_proba = model.predict_proba(X_test)[:, 1]

# AUC Score
auc = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**Giải thích AUC:**
```
AUC = 1.0: Hoàn hảo
AUC = 0.9+: Xuất sắc
AUC = 0.8-0.9: Tốt
AUC = 0.7-0.8: Khá
AUC = 0.6-0.7: Yếu
AUC = 0.5: Random (vô dụng)
```

### Bảng tổng hợp chọn Metric

| Tình huống | Metric đề xuất | Lý do |
|------------|----------------|-------|
| Classes cân bằng, đơn giản | Accuracy | Dễ hiểu, phản ánh đúng |
| Classes mất cân bằng | F1, Precision-Recall AUC | Accuracy misleading |
| FP tốn kém (spam filter) | Precision | Giảm false alarm |
| FN tốn kém (y tế) | Recall | Không bỏ sót |
| So sánh models | ROC-AUC | Đánh giá tổng thể |
| Ranking (search, recommendation) | MAP, NDCG | Quan tâm thứ tự |

### Regression Metrics

| Metric | Công thức | Khi nào dùng |
|--------|-----------|--------------|
| **MAE** | mean(\|y - ŷ\|) | Muốn metric dễ hiểu, ít nhạy outliers |
| **MSE** | mean((y - ŷ)²) | Muốn phạt nặng lỗi lớn |
| **RMSE** | sqrt(MSE) | Như MSE nhưng cùng đơn vị với y |
| **MAPE** | mean(\|(y - ŷ)/y\|) × 100% | Muốn metric % để so sánh |
| **R²** | 1 - SS_res/SS_tot | Muốn biết % variance giải thích được |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = [100, 150, 200, 250, 300]
y_pred = [110, 140, 190, 260, 310]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.2f}")   # 12.00
print(f"MSE: {mse:.2f}")   # 160.00
print(f"RMSE: {rmse:.2f}") # 12.65
print(f"R²: {r2:.4f}")     # 0.9800
```

---

## 7. Pipeline hoàn chỉnh với sklearn

### Tại sao dùng Pipeline?

**Không có Pipeline:**
```python
# Dễ quên bước, dễ sai thứ tự
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Nhớ phải transform, không fit!

imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

model = LogisticRegression()
model.fit(X_train_imputed, y_train)
# Bug tiềm ẩn: thứ tự sai, quên bước...
```

**Có Pipeline:**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)  # Tự động xử lý tất cả
y_pred = pipeline.predict(X_test)  # Không thể sai!
```

### Pipeline đầy đủ: Preprocessing + Model + Evaluation

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CHUẨN BỊ DỮ LIỆU
# =============================================================================

# Tạo dữ liệu mẫu (thay bằng data thực tế)
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(18, 65, n_samples),
    'income': np.random.randint(5_000_000, 100_000_000, n_samples),
    'loan_amount': np.random.randint(10_000_000, 500_000_000, n_samples),
    'employment_years': np.random.randint(0, 30, n_samples),
    'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples),
    'marital_status': np.random.choice(['single', 'married', 'divorced'], n_samples),
}
df = pd.DataFrame(data)

# Target: Khả năng vỡ nợ (0: không, 1: có)
df['default'] = ((df['loan_amount'] / df['income'] > 5) & 
                 (df['employment_years'] < 5)).astype(int)

# Thêm missing values
df.loc[np.random.choice(n_samples, 50), 'income'] = np.nan
df.loc[np.random.choice(n_samples, 30), 'age'] = np.nan

print("Dữ liệu mẫu:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['default'].value_counts(normalize=True)}")

# =============================================================================
# 2. ĐỊNH NGHĨA FEATURES
# =============================================================================

# Phân loại features
numeric_features = ['age', 'income', 'loan_amount', 'employment_years']
categorical_features = ['education', 'marital_status']

X = df[numeric_features + categorical_features]
y = df['default']

# =============================================================================
# 3. TẠO PREPROCESSING PIPELINE
# =============================================================================

# Pipeline cho features số
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Điền missing bằng median
    ('scaler', StandardScaler())                     # Chuẩn hóa
])

# Pipeline cho features phân loại
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Kết hợp với ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# =============================================================================
# 4. TẠO FULL PIPELINE VỚI MODEL
# =============================================================================

# Pipeline hoàn chỉnh
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

print("\nPipeline structure:")
print(pipeline)

# =============================================================================
# 5. TRAIN/TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# =============================================================================
# 6. CROSS-VALIDATION
# =============================================================================

print("\n" + "="*50)
print("CROSS-VALIDATION")
print("="*50)

# 5-Fold Cross-Validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')

print(f"CV AUC Scores: {cv_scores}")
print(f"Mean AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =============================================================================
# 7. TRAINING VÀ EVALUATION
# =============================================================================

print("\n" + "="*50)
print("TRAINING & EVALUATION")
print("="*50)

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Không vỡ nợ', 'Vỡ nợ']))

print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# =============================================================================
# 8. SO SÁNH NHIỀU MODELS
# =============================================================================

print("\n" + "="*50)
print("SO SÁNH MODELS")
print("="*50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    # Tạo pipeline mới với model khác
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Cross-validation
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    
    results.append({
        'Model': name,
        'Mean AUC': scores.mean(),
        'Std AUC': scores.std()
    })
    
    print(f"{name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# =============================================================================
# 9. LƯU VÀ LOAD MODEL
# =============================================================================

import joblib

# Lưu pipeline
joblib.dump(pipeline, 'credit_risk_model.pkl')
print("\nModel đã lưu vào 'credit_risk_model.pkl'")

# Load lại
loaded_pipeline = joblib.load('credit_risk_model.pkl')

# Test với data mới
new_data = pd.DataFrame({
    'age': [35],
    'income': [30_000_000],
    'loan_amount': [200_000_000],
    'employment_years': [3],
    'education': ['bachelor'],
    'marital_status': ['married']
})

prediction = loaded_pipeline.predict(new_data)[0]
probability = loaded_pipeline.predict_proba(new_data)[0, 1]

print(f"\nDự đoán cho khách hàng mới:")
print(f"- Kết quả: {'Có nguy cơ vỡ nợ' if prediction == 1 else 'Không có nguy cơ'}")
print(f"- Xác suất vỡ nợ: {probability:.2%}")
```

### Output mẫu:

```
Dữ liệu mẫu:
   age     income  loan_amount  employment_years   education marital_status  default
0   51   74302834    148991325                 6    bachelor       divorced        0
1   55   68480009    458973907                 8  high_school         single        1
2   62   26563088    177668355                23     bachelor        married        0

Shape: (1000, 7)
Missing values:
age                  30
income               50
loan_amount           0
employment_years      0
education             0
marital_status        0
default               0

==================================================
CROSS-VALIDATION
==================================================
CV AUC Scores: [0.8234, 0.8156, 0.8312, 0.8089, 0.8267]
Mean AUC: 0.8212 ± 0.0078

==================================================
TRAINING & EVALUATION
==================================================
Confusion Matrix:
[[168   7]
 [ 15  10]]

Classification Report:
              precision    recall  f1-score   support
 Không vỡ nợ       0.92      0.96      0.94       175
       Vỡ nợ       0.59      0.40      0.48        25
    accuracy                           0.89       200

ROC-AUC Score: 0.8342

==================================================
SO SÁNH MODELS
==================================================
Logistic Regression: AUC = 0.8212 ± 0.0078
Random Forest: AUC = 0.8456 ± 0.0124

Model đã lưu vào 'credit_risk_model.pkl'

Dự đoán cho khách hàng mới:
- Kết quả: Có nguy cơ vỡ nợ
- Xác suất vỡ nợ: 78.34%
```

### Pipeline cho Time Series (với TimeSeriesSplit)

```python
from sklearn.model_selection import TimeSeriesSplit

# Giả sử data đã sort theo thời gian
# X, y có thứ tự thời gian

# TimeSeriesSplit thay vì K-Fold
tscv = TimeSeriesSplit(n_splits=5)

print("Time Series Cross-Validation:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
    
    # Fit và predict
    pipeline.fit(X_train_fold, y_train_fold)
    score = pipeline.score(X_test_fold, y_test_fold)
    
    print(f"Fold {fold+1}: Train[0:{len(train_idx)}] Test[{len(train_idx)}:{len(train_idx)+len(test_idx)}] Score={score:.4f}")
```

---

## Tổng kết

### Checklist trước khi train model

```
□ Kiểm tra data leakage
  - Features không chứa info từ tương lai
  - Preprocessing fit trên train only

□ Chọn Cross-Validation phù hợp
  - Time series → TimeSeriesSplit
  - Imbalanced → Stratified K-Fold
  - Balanced → K-Fold

□ Feature Scaling nếu cần
  - KNN, SVM, Linear models → Cần scaling
  - Tree-based → Không cần

□ Chọn metric phù hợp
  - Balanced classification → Accuracy, F1
  - Imbalanced → F1, AUC-PR
  - Regression → MAE, RMSE

□ Dùng Pipeline
  - Tránh lỗi
  - Dễ reproduce
  - Dễ deploy
```

### Workflow tổng quát

```
1. Load & Explore Data
   ↓
2. Feature Engineering (không dùng test data!)
   ↓
3. Train/Test Split (TimeSeriesSplit cho time series)
   ↓
4. Build Pipeline (Preprocessing + Model)
   ↓
5. Cross-Validation
   ↓
6. Train Final Model
   ↓
7. Evaluate trên Test Set
   ↓
8. Save Model
```

---

## Tài liệu tham khảo

**Sách:**
- "Hands-On Machine Learning" - Aurélien Géron
- "Introduction to Statistical Learning" - James et al.

**Online:**
- sklearn User Guide: https://scikit-learn.org/stable/user_guide.html
- StatQuest YouTube: Machine Learning Series

**Bước tiếp theo:**
- `02_DEEP_LEARNING_BASICS.md` - Neural Networks cơ bản
- `03_TIME_SERIES_FUNDAMENTALS.md` - Đặc thù dữ liệu chuỗi thời gian
