# 01 – Chuyện gì xảy ra trong từng file nguồn?

Tài liệu này **giải thích nội dung** các file trong `src/` mà không sửa bất kỳ file nào. Mục đích: hiểu từng file đang làm gì, để sau đó gắn với mục tiêu forecasting + anomaly + 5 models.

---

## 1. `src/models/ml.py` – File này làm gì?

**Vai trò:** Script minh họa **ML cơ bản** (cross-validation, scaling, đánh giá classification). **Chưa phải** pipeline forecasting VN30 hay so sánh 5 models; đây là nền để sau này gắn vào.

### 1.1. Phần đầu: import và dữ liệu mẫu

```text
Import: pandas, sklearn (KFold, cross_val_score, TimeSeriesSplit, LinearRegression,
        StandardScaler, MinMaxScaler, RobustScaler, confusion_matrix, classification_report),
        matplotlib, seaborn, numpy.
```

- Tạo **dữ liệu giả**: `X` (3 cột f1, f2, f3), `y` = tổ hợp tuyến tính của X + nhiễu. Dùng để chạy ví dụ, **không** đọc data VN30.

### 1.2. KFold cross-validation

- `KFold(n_splits=5, shuffle=True, random_state=42)`: chia data thành 5 phần, shuffle rồi lần lượt dùng 4 phần train, 1 phần test.
- `cross_val_score(..., scoring='neg_mean_squared_error')`: với mỗi fold, fit `LinearRegression` và tính MSE. Scikit-learn trả về “neg MSE” (càng lớn càng tốt), nên code lấy `-scores` để có MSE thực (càng nhỏ càng tốt).
- **Ý nghĩa:** Minh họa cách đánh giá model ổn định qua nhiều fold. Với time series thật sẽ cần **TimeSeriesSplit** (đã có ở dưới) thay vì KFold shuffle.

### 1.3. TimeSeriesSplit

- `TimeSeriesSplit(n_splits=5)`: chia theo thời gian (train luôn là quá khứ, test là tương lai), không shuffle.
- Vòng lặp: với mỗi fold, lấy `train_idx`, `test_idx` → `X_train, X_test`, `Y_train, Y_test` → fit `LinearRegression`, tính `model.score(X_test, Y_test)` (R²).
- **Ý nghĩa:** Đúng cách đánh giá cho dữ liệu chuỗi thời gian; đây là pattern sẽ dùng cho forecasting (1, 5, 20 ngày) sau này.

### 1.4. Scaling (StandardScaler, MinMaxScaler, RobustScaler)

- **StandardScaler:** `(X - mean) / std` → mean 0, std 1.
- **MinMaxScaler:** `(X - min) / (max - min)` → giá trị trong [0, 1].
- **RobustScaler:** `(X - median) / IQR` → ít bị ảnh hưởng bởi outlier.
- Code chỉ minh họa cách dùng trên `X_train`; trong pipeline thật cần fit trên train rồi transform cả train/test để tránh leakage.

### 1.5. Confusion Matrix & Classification Report

- Dùng **dữ liệu mẫu** `y_true`, `y_pred` (0/1) để tính `confusion_matrix` và `classification_report`.
- Vẽ heatmap confusion matrix (matplotlib + seaborn).
- **Ý nghĩa:** Nếu sau này có bài toán classification (ví dụ: tăng/giảm, hoặc anomaly vs bình thường), đây là cách đánh giá sẽ dùng.

### 1.6. Kết luận cho `ml.py`

- File **không** đọc data VN30, **không** có multi-step forecast, **không** có LSTM/Transformer/anomaly.
- File **có**: (1) TimeSeriesSplit đúng cho time series, (2) scaling, (3) metric regression (MSE, R²), (4) metric classification (confusion matrix, report). Đây là “building blocks” để sau này gắn vào pipeline 5 models + multi-step + anomaly.

---

## 2. `src/features/build_features.py` – Chuyện gì xảy ra trong file này?

**Vai trò:** Từ dữ liệu giá đã clean (OHLCV), tính **hơn 45 technical features** (returns, MA, EMA, RSI, MACD, Bollinger, volatility, volume, momentum, price range…) và lưu ra CSV. Đây là **đầu vào** cho mọi model (classical ML, LSTM, Transformer, patch, anomaly).

### 2.1. Các hàm tính feature (từng nhóm)

- **`calculate_returns(df, periods=[1,5,10,20])`**  
  Tính % thay đổi giá sau 1, 5, 10, 20 kỳ. Dùng cho target (return 1d, 5d, 20d) và làm feature.

- **`calculate_moving_averages`**  
  SMA với windows [5,10,20,50]. Cột dạng `ma_5`, `ma_10`, …

- **`calculate_ema`**  
  EMA spans [12, 26]. Dùng trong MACD và làm feature.

- **`calculate_volatility`**  
  Độ lệch chuẩn của returns theo cửa sổ (vd 5, 10, 20). Quan trọng cho risk và anomaly (biến động bất thường).

- **`calculate_rsi`**  
  RSI period 14. Overbought/oversold → có thể liên quan anomaly (cực đoan).

- **`calculate_macd`**  
  MACD, signal, histogram (fast=12, slow=26, signal=9).

- **`calculate_bollinger_bands`**  
  Middle/upper/lower band, width. Giá chạm band có thể dùng cho anomaly/regime.

- **`calculate_volume_features`**  
  Volume MA, volume ratio, volume change. Cần cột `volume`.

- **`calculate_price_momentum`**  
  Hiệu giá hiện tại và giá N kỳ trước (5, 10, 20).

- **`calculate_price_range`**  
  Các đặc trưng theo khoảng giá (high-low, …) trong cửa sổ.

Tất cả đều **chỉ thêm cột** vào DataFrame, không xóa dữ liệu gốc (trừ khi gọi `dropna()` ở bước sau).

### 2.2. `calculate_all_features(df, feature_sets=...)`

- **Vào:** DataFrame có cột `close` (và `date`, `volume` nếu cần).
- **feature_sets:** Danh sách tên nhóm: `'returns'`, `'ma'`, `'ema'`, `'volatility'`, `'rsi'`, `'macd'`, `'bollinger'`, `'volume'`, `'momentum'`, `'price_range'`. Mặc định dùng tất cả.
- **Luồng:** Sắp xếp theo `date` (nếu có) → gọi lần lượt từng hàm tính feature tương ứng → log số cột thêm.
- **Ra:** Cùng DataFrame với rất nhiều cột mới (45+). Một số hàng đầu sẽ có NaN do rolling/EMA, thường được xử lý ở bước sau (`drop_na`).

### 2.3. `build_features_single(filename, clean_dir, features_dir, ...)`

- **Vào:** Tên file CSV trong `clean_dir` (vd `FPT.csv`).
- **Luồng:**  
  1. Đọc CSV từ `clean_dir/filename`.  
  2. Gọi `calculate_all_features(df, feature_sets)`.  
  3. Nếu `drop_na=True` thì `df.dropna()`.  
  4. Nếu `save_file=True` thì ghi CSV ra `features_dir/filename`.  
- **Ra:** DataFrame đã có đủ features hoặc `None` nếu lỗi.

### 2.4. `build_features(clean_dir, features_dir, pattern='*.csv', ...)`

- **Vào:** Thư mục clean, thư mục features, pattern (mặc định mọi CSV).
- **Luồng:**  
  1. Tạo `features_dir` nếu chưa có.  
  2. Duyệt mọi file khớp `pattern` trong `clean_dir`.  
  3. Với mỗi file gọi `build_features_single(...)`, thu kết quả vào dict `{filename: DataFrame}`.  
  4. Nếu `skip_on_error=True` thì file lỗi bỏ qua, không dừng cả batch.  
- **Ra:** Dict mapping tên file → DataFrame (đã có đủ features). File CSV đã được ghi vào `features_dir`.

### 2.5. Kết luận cho `build_features.py`

- File **không** train model, **không** làm forecasting hay anomaly; nó chỉ **tạo input features** từ giá đã clean.
- Các cột returns (1d, 5d, 10d, 20d) và momentum/volatility chính là thứ bạn sẽ dùng làm **target** (multi-step 1, 5, 20 ngày) và **feature** cho 5 models + anomaly. Pipeline VN30 gọi `build_features` sau bước clean.

---

## 3. Pipeline VN30 – `src/pipeline/vnindex30/fetch_vn30.py`

**Vai trò:** Script chạy **full pipeline** cho VN30: Crawl → Clean → Features. Đây là nơi **gắn kết** crawler, clean và `build_features` lại với nhau.

### 3.1. Đầu file: path và encoding

- Thêm project root vào `sys.path` để `import src.*` chạy được.
- Trên Windows, set stdout/stderr UTF-8 để log tiếng Việt không lỗi.

### 3.2. Import và config

- Import: `crawl_many` (runcrawler), `clean_many` (clean_price), `build_features` (build_features), `load_yaml` (file_utils).
- **VN30_SYMBOLS_FALLBACK:** Danh sách 30 mã VN30 cứng, dùng khi không có `symbols.yaml`.
- **`_get_config_path`:** Trả về đường dẫn file trong `configs/` (vd `config.yaml`, `symbols.yaml`).
- **`load_pipeline_config()`:** Đọc `config.yaml` và `symbols.yaml` → trả về dict gồm: `symbols`, `start_date`, `end_date`, `raw_dir`, `clean_dir`, `features_dir`, `page_size`, `skip_on_error`, `clean_opts`, v.v. Nếu thiếu file thì dùng fallback (vd symbols = VN30_SYMBOLS_FALLBACK).

### 3.3. Luồng chính (khi chạy script)

- Gọi `load_pipeline_config()`.
- **Bước 1 – Crawl:** `crawl_many(symbols, start_date, end_date, raw_dir, ...)` → tải dữ liệu thô từ CafeF vào `raw_dir`.
- **Bước 2 – Clean:** `clean_many(raw_dir, clean_dir, ...)` → đọc CSV thô, chuẩn hóa (ngày, cột, loại bỏ lỗi), ghi vào `clean_dir`.
- **Bước 3 – Features:** `build_features(clean_dir, features_dir, ...)` → với mỗi CSV trong `clean_dir`, gọi logic trong `build_features.py` (calculate_all_features → dropna → save) và ghi ra `features_dir`.

Kết quả: trong `data/raw`, `data/clean`, `data/features` bạn có dữ liệu đủ để **train và so sánh 5 models** (Transformer, Patch, LSTM, classical ML, baseline) và làm **multi-step (1, 5, 20 ngày)** + **anomaly detection**. File này không chạy model; nó chỉ chuẩn bị data.

---

## 4. Tóm tắt: Ai làm gì?

| Thành phần | Làm gì | Không làm gì |
|------------|--------|----------------|
| **`ml.py`** | TimeSeriesSplit, scaling, MSE/R², confusion matrix (minh họa) | Không đọc VN30, không forecast, không so sánh 5 models |
| **`build_features.py`** | Từ OHLCV → 45+ features, returns 1/5/10/20d, lưu CSV | Không train, không forecast, không anomaly |
| **`fetch_vn30.py`** | Crawl → Clean → build_features cho VN30 | Không train model, không forecast, không anomaly |

Bước tiếp theo (so sánh 5 models, multi-step, anomaly, event) sẽ nằm trong **mentor/02_PROJECT_GOAL_AND_NEXT_STEPS.md** và cần code mới (không chỉnh sửa nội dung trong `docs/`).
