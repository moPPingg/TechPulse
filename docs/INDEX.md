# 📚 TÀI LIỆU HỌC TẬP TECHPULSE - INDEX
## Danh mục đầy đủ các tài liệu học tập

---

## 🎯 CÁCH SỬ DỤNG INDEX NÀY

1. **Đọc ROADMAP trước:** `ROADMAP_FULL_PROJECT.md`
2. **Học theo thứ tự:** Từ Phase 1 → Phase 6
3. **Làm đủ bài tập:** Mỗi file có checklist và bài tập
4. **Không bỏ qua:** Mỗi bài đều quan trọng

---

## 📖 DANH MỤC TÀI LIỆU

### 🗺️ **TỔNG QUAN**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `ROADMAP_FULL_PROJECT.md` | Lộ trình tổng thể 24 tuần | 30 phút đọc | ✅ Hoàn thành |
| `INDEX.md` | File này - Danh mục tài liệu | 10 phút đọc | ✅ Hoàn thành |

---

### 📚 **PHASE 1: FOUNDATIONS (Tuần 1-2)**

**Mục tiêu:** Nắm vững nền tảng ML và Time Series

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `01_foundations/01_MACHINE_LEARNING_BASICS.md` | ML cơ bản, supervised learning, metrics | 2-3 giờ | ✅ Hoàn thành |
| `01_foundations/03_TIME_SERIES_FUNDAMENTALS.md` | Time series, stationarity, autocorrelation | 2-3 giờ | ✅ Hoàn thành |
| `01_foundations/02_DEEP_LEARNING_BASICS.md` | Neural networks, backprop, gradient descent | 3-4 giờ | ✅ Hoàn thành |
| `LEARNING_GUIDE_FULL_SYSTEM.md` | **Hướng dẫn toàn bộ hệ thống** (bao gồm features) | 5-8 giờ | ✅ Hoàn thành |

**Checklist Phase 1:**
- [ ] Hiểu ML basics và supervised learning
- [ ] Phân biệt regression vs classification
- [ ] Hiểu train/test split cho time series
- [ ] Tính được MSE, MAE, RMSE, MAPE
- [ ] Hiểu time series components
- [ ] Kiểm tra được stationarity
- [ ] Phân tích được autocorrelation
- [ ] **Hiểu Technical Indicators (EMA, Momentum, Returns, Drawdown)** → Đọc LEARNING_GUIDE section 5.2
- [ ] Hiểu neural networks cơ bản
- [ ] Implement được perceptron
- [ ] Hiểu backpropagation

---

### 🤖 **PHASE 2: MODELING (Tuần 3-12)**

**Mục tiêu:** Implement và benchmark các models

#### **2.1. Baseline Models (Tuần 3-4)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `02_modeling/01_BASELINE_MODELS.md` | Linear Regression, ARIMA, GARCH, Naive | 4-5 giờ | ✅ Hoàn thành |

**Checklist:**
- [ ] Implement Linear Regression cho time series
- [ ] Hiểu và implement ARIMA(p,d,q)
- [ ] Implement GARCH cho volatility
- [ ] Implement naive forecasting methods
- [ ] So sánh các baselines
- [ ] Tạo được benchmark results

#### **2.2. Machine Learning Models (Tuần 5-6)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `02_modeling/02_ML_MODELS.md` | XGBoost, LightGBM, Random Forest | 4-5 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Implement XGBoost
- [ ] Implement LightGBM
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning
- [ ] So sánh với baselines

#### **2.3. LSTM & GRU (Tuần 7-8)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `02_modeling/03_LSTM_GRU.md` | LSTM, GRU cho time series | 5-6 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Hiểu LSTM cell
- [ ] Implement LSTM từ đầu
- [ ] Implement GRU
- [ ] Sequence-to-sequence prediction
- [ ] So sánh LSTM vs XGBoost

#### **2.4. Transformers (Tuần 9-10)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `02_modeling/04_TRANSFORMERS_LTSF.md` | iTransformer, TimesNet, PatchTST | 6-8 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Hiểu self-attention
- [ ] Implement iTransformer
- [ ] Implement TimesNet (optional)
- [ ] Benchmark vs LSTM
- [ ] Analyze attention weights

#### **2.5. Anomaly Detection (Tuần 11-12)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `02_modeling/05_ANOMALY_DETECTION.md` | Anomaly Transformer, TranAD | 5-6 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Hiểu anomaly detection
- [ ] Implement Anomaly Transformer
- [ ] Implement TranAD
- [ ] Detect anomalies trong VN30
- [ ] Validate với real events

---

### 🔗 **PHASE 3: MULTI-MODAL DATA - VIETNAM FOCUS (Tuần 13-16)**

**Mục tiêu:** Kết hợp price + tin tức Việt Nam

#### **3.1. Vietnamese News Data (Tuần 13-14)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `03_multimodal/01_NEWS_DATA_VIETNAM.md` | Crawl CafeF & VnExpress | 3-4 giờ | ⏳ Chưa tạo |
| `03_multimodal/02_VIETNAMESE_TEXT_PROCESSING.md` | Vietnamese NLP & sentiment | 4-5 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Crawl được tin từ CafeF (chứng khoán)
- [ ] Crawl được tin từ VnExpress (kinh tế)
- [ ] Xử lý tiếng Việt (tokenization, dấu)
- [ ] Vietnamese sentiment analysis (PhoBERT)
- [ ] Link news với price VN30
- [ ] Filter spam/clickbait
- [ ] Analyze correlation

#### **3.2. Event-Aware & Fusion (Tuần 15-16)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `03_multimodal/03_EVENT_DETECTION.md` | Phát hiện events từ tin VN + price | 3-4 giờ | ⏳ Chưa tạo |
| `03_multimodal/04_MULTIMODAL_FUSION.md` | Kết hợp price + Vietnamese text | 4-5 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Detect events từ tin CafeF & VnExpress
- [ ] Classify event types (earnings, M&A, scandal, etc.)
- [ ] Implement cross-modal attention (price + Vietnamese text)
- [ ] Train multimodal model
- [ ] Compare với single-modal (price only)

---

### 🎯 **PHASE 4: ADVANCED TOPICS (Tuần 17-20)**

**Mục tiêu:** Implement các kỹ thuật nâng cao

#### **4.1. Event-Aware Training (Tuần 15-16)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `04_advanced/01_EVENT_AWARE_TRAINING.md` | Weighted loss, event detection | 4-5 giờ | ✅ Hoàn thành |

**Checklist:**
- [ ] Detect event days
- [ ] Implement weighted loss
- [ ] Train với event-aware loss
- [ ] Compare với baseline
- [ ] Chứng minh improvement

#### **4.2. Regime Detection (Tuần 17-18)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `04_advanced/02_REGIME_DETECTION.md` | HMM, change point detection | 5-6 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Implement HMM
- [ ] Detect regime changes
- [ ] Separate models cho regimes
- [ ] Online learning mechanism

#### **4.3. Tail Risk Metrics (Tuần 19-20)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `04_advanced/03_TAIL_RISK_METRICS.md` | CVaR, tail loss, shock metrics | 3-4 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Implement tail loss
- [ ] Calculate CVaR
- [ ] Maximum drawdown
- [ ] Hit rate during shocks

#### **4.4. Efficient XAI (Tuần 19-20)**

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `04_advanced/04_EFFICIENT_XAI.md` | SHAP, TimeSHAP, efficient methods | 5-6 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Implement SHAP
- [ ] Implement TimeSHAP
- [ ] Efficient approximations
- [ ] Benchmark accuracy vs speed
- [ ] Visualize explanations

---

### 📊 **PHASE 5: EVALUATION (Tuần 21-22)**

**Mục tiêu:** Đánh giá toàn diện models

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `05_evaluation/01_METRICS_EVALUATION.md` | Comprehensive metrics | 3-4 giờ | ⏳ Chưa tạo |
| `05_evaluation/02_BACKTESTING.md` | Walk-forward validation, backtesting | 4-5 giờ | ⏳ Chưa tạo |
| `05_evaluation/03_CASE_STUDIES.md` | COVID crash, tech bubble case studies | 5-6 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Implement comprehensive metrics
- [ ] Walk-forward validation
- [ ] Backtest trading strategies
- [ ] Case study: COVID crash
- [ ] Case study: Tech bubble
- [ ] Compare all models

---

### 📝 **PHASE 6: PAPER WRITING (Tuần 23-24)**

**Mục tiêu:** Viết paper nghiên cứu

| File | Nội dung | Thời gian | Trạng thái |
|------|----------|-----------|------------|
| `06_paper_writing/01_RESEARCH_METHODOLOGY.md` | Phương pháp nghiên cứu | 2-3 giờ | ⏳ Chưa tạo |
| `06_paper_writing/02_EXPERIMENT_DESIGN.md` | Thiết kế thí nghiệm | 3-4 giờ | ⏳ Chưa tạo |
| `06_paper_writing/03_PAPER_STRUCTURE.md` | Cấu trúc paper, viết từng section | 5-6 giờ | ⏳ Chưa tạo |

**Checklist:**
- [ ] Viết methodology section
- [ ] Design experiments
- [ ] Create figures and tables
- [ ] Write results section
- [ ] Write discussion
- [ ] Complete paper draft

---

## 📊 TIẾN ĐỘ TỔNG THỂ

### **Thống kê:**
```
Tổng số files: 30
Đã hoàn thành: 6 (20%)
Chưa tạo: 24 (80%)

Phase 1 (Foundations): 3/3 files (100%) ✅
Phase 2 (Modeling): 1/5 files (20%) ⏳
Phase 3 (Multimodal): 0/4 files (0%) ⏳
Phase 4 (Advanced): 1/4 files (25%) ⏳
Phase 5 (Evaluation): 0/3 files (0%) ⏳
Phase 6 (Paper): 0/3 files (0%) ⏳
```

### **Files đã hoàn thành:**
1. ✅ ROADMAP_FULL_PROJECT.md
2. ✅ 01_foundations/01_MACHINE_LEARNING_BASICS.md
3. ✅ 01_foundations/02_DEEP_LEARNING_BASICS.md
4. ✅ 01_foundations/03_TIME_SERIES_FUNDAMENTALS.md
5. ✅ 02_modeling/01_BASELINE_MODELS.md
6. ✅ 04_advanced/01_EVENT_AWARE_TRAINING.md

### **Files ưu tiên tiếp theo:**
1. 🔜 02_modeling/02_ML_MODELS.md (Tuần 5-6)
2. 🔜 02_modeling/03_LSTM_GRU.md (Tuần 7-8)
3. 🔜 02_modeling/04_TRANSFORMERS_LTSF.md (Tuần 9-10)

---

## 🎯 CÁCH HỌC HIỆU QUẢ

### **Quy trình học mỗi file:**

```
1. ĐỌC (30-60 phút)
   - Đọc toàn bộ file
   - Ghi chú phần chưa hiểu
   - Xem references nếu cần

2. HIỂU (1-2 giờ)
   - Vẽ sơ đồ, mindmap
   - Giải thích lại bằng lời mình
   - Hỏi ChatGPT/Claude nếu chưa rõ

3. LÀM (3-5 giờ)
   - Code từng bước nhỏ
   - Test ngay từng function
   - Debug khi có lỗi
   - Làm hết bài tập

4. KIỂM TRA (30 phút)
   - Làm checklist cuối file
   - Nếu chưa pass → quay lại bước 2
   - Nếu pass → sang file tiếp theo
```

### **Lưu ý quan trọng:**

1. **KHÔNG bỏ qua bài tập:**
   - Mỗi file có 2-3 bài tập thực hành
   - Bài tập giúp consolidate kiến thức
   - Làm đủ bài tập mới sang file mới

2. **KHÔNG học vội:**
   - Hiểu sâu > Học nhanh
   - 1 file/tuần là tốc độ hợp lý
   - Nếu chưa hiểu, đọc lại

3. **GHI CHÚ và THẢO LUẬN:**
   - Ghi chú những điểm quan trọng
   - Thảo luận với bạn bè/mentor
   - Hỏi trên forums (Stack Overflow, Reddit)

4. **CODE TỪ ĐẦU:**
   - Không copy-paste code
   - Type từng dòng để hiểu
   - Debug từng lỗi để học

---

## 📈 THEO DÕI TIẾN ĐỘ

### **Checklist tổng thể:**

**Tuần 1-2: Foundations**
- [ ] ML Basics
- [ ] Time Series Fundamentals
- [ ] Deep Learning Basics

**Tuần 3-4: Baseline Models**
- [ ] Linear Regression
- [ ] ARIMA
- [ ] GARCH
- [ ] Naive methods

**Tuần 5-6: ML Models**
- [ ] XGBoost
- [ ] LightGBM
- [ ] Feature engineering

**Tuần 7-8: LSTM/GRU**
- [ ] LSTM
- [ ] GRU
- [ ] Seq2Seq

**Tuần 9-10: Transformers**
- [ ] iTransformer
- [ ] TimesNet
- [ ] Attention analysis

**Tuần 11-12: Anomaly Detection**
- [ ] Anomaly Transformer
- [ ] TranAD
- [ ] Event validation

**Tuần 13-14: News Data**
- [ ] News crawling
- [ ] Text processing
- [ ] Sentiment analysis

**Tuần 15-16: Event-Aware**
- [ ] Event detection
- [ ] Weighted loss
- [ ] Multimodal fusion

**Tuần 17-18: Regime Detection**
- [ ] HMM
- [ ] Change point detection
- [ ] Separate models

**Tuần 19-20: XAI**
- [ ] SHAP
- [ ] TimeSHAP
- [ ] Efficient methods

**Tuần 21-22: Evaluation**
- [ ] Comprehensive metrics
- [ ] Backtesting
- [ ] Case studies

**Tuần 23-24: Paper**
- [ ] Methodology
- [ ] Experiments
- [ ] Writing

---

## 💡 TÀI NGUYÊN BỔ SUNG

### **Khi cần giúp đỡ:**

1. **ChatGPT/Claude:**
   - Giải thích concepts
   - Debug code
   - Review code

2. **Stack Overflow:**
   - Lỗi cụ thể
   - Implementation issues

3. **Papers:**
   - Mỗi file có references
   - Đọc papers để hiểu sâu

4. **GitHub:**
   - Xem code của người khác
   - Học best practices

5. **YouTube:**
   - StatQuest
   - 3Blue1Brown
   - Krish Naik

---

## 🎓 KẾT QUẢ MONG ĐỢI

Sau khi hoàn thành toàn bộ tài liệu, bạn sẽ:

1. **Kiến thức:**
   - Hiểu sâu ML/DL cho time series
   - Master LSTM, Transformers
   - Hiểu event-aware training
   - Hiểu XAI methods

2. **Kỹ năng:**
   - Implement models from scratch
   - Debug complex systems
   - Analyze results
   - Write research papers

3. **Sản phẩm:**
   - Full pipeline (crawl → model → evaluate)
   - 10+ models implemented
   - Benchmark results
   - Paper draft

4. **Tự tin:**
   - Làm được research
   - Publish paper
   - Present results

---

## 🚀 BẮT ĐẦU NGAY

**Bước đầu tiên của bạn:**

1. Đọc `ROADMAP_FULL_PROJECT.md` (30 phút)
2. Đọc `01_foundations/01_MACHINE_LEARNING_BASICS.md` (2-3 giờ)
3. Làm bài tập trong file đó
4. Kiểm tra checklist
5. Sang file tiếp theo

**Chúc bạn học tốt! 🎓**

---

*Cập nhật lần cuối: 2026-01-28*
*Tổng số files: 30 (6 hoàn thành, 24 đang phát triển)*
