# 02 – Mục tiêu dự án và bước làm tiếp

Tài liệu này **ánh xạ** mục tiêu (5 models, multi-step, anomaly, event) với code hiện tại và liệt kê bước làm tiếp. Vẫn **không chỉnh sửa** `docs/` hay bất kỳ file nguồn nào; đây chỉ là hướng dẫn.

---

## 1. Mục tiêu dự án (tóm tắt)

- **Hệ thống:** Forecasting + anomaly detection cho cổ phiếu **VN30**.
- **So sánh 5 loại model:**  
  - Transformer-based  
  - Patch-based (vd TimesNet, PatchTST)  
  - LSTM-based  
  - Classical ML (XGBoost, LightGBM, Random Forest, v.v.)  
  - Baseline (ARIMA / linear / naive)
- **Multi-step forecasting:** horizon 1, 5, 20 ngày.
- **Anomaly detection:** phát hiện ngày/đoạn bất thường (volatility, return cực đoan, v.v.).
- **Event / news (sau):** training nhận biết sự kiện, tail-aware, v.v.

---

## 2. Code hiện tại đã có gì?

- **Data:** Crawl (CafeF) → Clean → **Features** (`build_features.py`) → CSV trong `data/features`. Đủ để train mọi model.
- **Features:** Returns 1/5/10/20d, MA, EMA, RSI, MACD, Bollinger, volatility, volume, momentum, price range. Có thể dùng:
  - **Target:** `return_1d`, `return_5d`, `return_20d` (hoặc tương đương) cho multi-step 1, 5, 20 ngày.
  - **Input:** Các cột còn lại (và có thể thêm lag của target).
- **ML minh họa:** `ml.py` có TimeSeriesSplit, scaling, MSE/R², confusion matrix. Chưa có: LSTM, Transformer, patch, anomaly, event.

**Kết luận:** Data và feature đã sẵn; phần còn thiếu là **pipeline training/evaluation** cho 5 models, **multi-step** và **anomaly**.

---

## 3. Ánh xạ mục tiêu → việc cần làm

### 3.1. Multi-step forecasting (1, 5, 20 ngày)

- **Target:**  
  - 1 ngày: `y = return_1d` (hoặc close t+1).  
  - 5 ngày: `y = return_5d` hoặc close t+5.  
  - 20 ngày: `y = return_20d` hoặc close t+20.
- **Input:** Cùng bộ features từ `build_features.py`; có thể thêm lag của giá/returns. Quan trọng: **không dùng tương lai** (tránh lookahead).
- **Việc cần làm:**  
  - Tạo 3 target (hoặc 3 pipeline) cho horizon 1, 5, 20.  
  - Train/test split theo thời gian (TimeSeriesSplit hoặc single cut).  
  - Metric: MAE, RMSE, MAPE (và có thể directional accuracy). So sánh 5 models trên cùng metric.

### 3.2. So sánh 5 models

| Loại | Ví dụ | Việc cần làm |
|------|--------|----------------|
| **Baseline** | Naive (last value), linear regression, ARIMA | Implement hoặc gọi thư viện (statsmodels ARIMA); dùng cùng feature/target. |
| **Classical ML** | XGBoost, LightGBM, Random Forest | Dùng `data/features`; TimeSeriesSplit; tune (optional). |
| **LSTM-based** | LSTM, GRU | Sequence từ features (vd sliding window); PyTorch/TF; cùng train/val/test. |
| **Transformer-based** | Informer, Autoformer, custom encoder | Sequence input; positional encoding; cùng pipeline. |
| **Patch-based** | PatchTST, TimesNet | Chia chuỗi thành patch; cùng target 1/5/20d. |

Gợi ý: Tạo **một script hoặc module** (vd `src/models/compare_models.py` hoặc từng file theo model) đọc từ `data/features`, xây sequence nếu cần, train từng model, lưu metric (và optional checkpoint). Không cần sửa `docs/`; code mới nằm trong `src/` hoặc `scripts/`.

### 3.3. Anomaly detection

- **Mục tiêu:** Đánh dấu ngày hoặc đoạn “bất thường” (volatility cao, return cực đoan, volume dị thường, v.v.).
- **Cách làm có thể:**  
  - **Unsupervised:** Isolation Forest, One-Class SVM, autoencoder reconstruction error trên features (hoặc trên chuỗi returns).  
  - **Threshold:** Dựa trên distribution của returns/volatility (vd ngoài 2–3 std).  
  - **Label (nếu có):** Có thể dùng cho supervised hoặc evaluation.
- **Input:** Cùng DataFrame từ `build_features.py` (returns, volatility, volume, …). Có thể thêm rolling stats.
- **Việc cần làm:** Một module/script đọc `data/features`, chạy 1–2 phương pháp anomaly, xuất (vd cột `is_anomaly` hoặc score). Sau đó có thể kết hợp với event-aware (coi anomaly như “event”).

### 3.4. Event / news (sau)

- **Mục tiêu:** Training nhận biết sự kiện (event-aware), tail-aware, có thể kết hợp tin.
- **Hiện trạng:** Trong repo đã có tài liệu hướng dẫn (vd event-aware, tail-aware) trong `docs/`; **không chỉnh sửa** `docs/`, chỉ đọc để hiểu.
- **Việc cần làm (sau khi đã có 5 models + multi-step + anomaly):**  
  - Có nhãn hoặc proxy “event” (từ anomaly, hoặc từ tin khi đã crawl).  
  - Weight sample theo event (event-aware training) hoặc focus tail (tail-aware).  
  - Có thể thêm nhánh text (embedding tin) nếu đã có pipeline tin tiếng Việt.

---

## 4. Thứ tự làm gợi ý (không sửa docs)

1. **Pipeline dữ liệu thống nhất**  
   - Từ `data/features/*.csv` → load, chọn symbol (vd 1 mã để thử), tạo train/val/test theo thời gian.  
   - Định nghĩa rõ target: return 1d, 5d, 20d (hoặc tương đương).

2. **Baseline + classical ML**  
   - Linear regression, naive forecast.  
   - XGBoost/LightGBM trên cùng bảng features.  
   - Đánh giá MAE/RMSE/MAPE cho horizon 1, 5, 20.

3. **LSTM**  
   - Xây sequence (sliding window) từ features; predict 1, 5, 20 bước.  
   - So sánh metric với baseline và classical ML.

4. **Transformer / Patch**  
   - Implement hoặc dùng thư viện (vd PatchTST, Informer).  
   - Cùng data và target; so sánh metric.

5. **Anomaly**  
   - Module đọc features → chạy Isolation Forest / reconstruction error / threshold.  
   - Output: nhãn hoặc score anomaly; sau gắn với event-aware nếu cần.

6. **Event-aware / tail-aware (sau)**  
   - Dùng tài liệu trong `docs/` làm hướng dẫn; implement weighting hoặc loss riêng trong code, không sửa file trong `docs/`.

---

## 5. Kết luận

- **Đã có:** Data VN30 (crawl → clean → features), hơn 45 features, returns 1/5/10/20d; minh họa TimeSeriesSplit và metric trong `ml.py`.  
- **Chưa có:** Pipeline training/evaluation cho 5 models, multi-step 1/5/20 ngày, anomaly detection, event-aware.  
- **Cách làm:** Thêm code mới trong `src/` (vd `src/models/`, `src/evaluation/`) và `scripts/`; giữ nguyên toàn bộ `docs/` và chỉ dùng `docs/` để đọc, không sửa.  
- **Giải thích chi tiết “chuyện gì xảy ra trong từng file”** nằm trong **mentor/01_WHAT_HAPPENS_IN_SOURCE_FILES.md**.
