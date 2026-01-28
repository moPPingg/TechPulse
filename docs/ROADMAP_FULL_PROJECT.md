# 🗺️ ROADMAP HOÀN THIỆN DỰ ÁN TECHPULSE
## Lộ trình từ Foundation → Research-Ready System

---

## 📍 VỊ TRÍ HIỆN TẠI

```
[████░░░░░░░░░░░░░░░░] 25% HOÀN THÀNH

✅ Phase 1: Data Foundation (25%) - HOÀN THÀNH
   - ✅ Crawl price data từ CafeF (VN30)
   - ✅ Clean & validate data
   - ✅ Build 45+ technical features
   - ✅ Pipeline automation

❌ Phase 2: Modeling & Benchmark (35%) - CHƯA BẮT ĐẦU
   - Baseline models (ARIMA, GARCH)
   - ML models (XGBoost, LightGBM)
   - DL models (LSTM, GRU, Transformer)
   - Anomaly detection

❌ Phase 3: Multi-source Data - Vietnam (20%) - CHƯA BẮT ĐẦU
   - Crawl tin tức CafeF & VnExpress
   - Vietnamese NLP & sentiment
   - Event detection
   - Multimodal fusion

❌ Phase 4: Advanced Methods (15%) - CHƯA BẮT ĐẦU
   - Event-aware training
   - Regime detection
   - Efficient XAI

❌ Phase 5: Evaluation & Paper (5%) - CHƯA BẮT ĐẦU
   - Tail risk metrics
   - Backtesting
   - Case studies & paper writing
```

---

## 🎯 MỤC TIÊU CUỐI CÙNG

Xây dựng hệ thống dự báo giá cổ phiếu **Việt Nam** có khả năng:

1. **Dự báo chính xác** trong điều kiện bình thường
2. **Phát hiện sớm** các cú sốc/biến động lớn (tail events)
3. **Thích ứng nhanh** khi thị trường thay đổi (regime change)
4. **Giải thích được** tại sao dự báo như vậy (XAI)
5. **Kết hợp đa nguồn** dữ liệu Việt Nam:
   - ✅ Price data từ CafeF
   - ✅ Tin tức từ CafeF & VnExpress
   - ✅ Vietnamese sentiment analysis
   - ✅ Technical indicators

---

## 📚 CẤU TRÚC TÀI LIỆU HỌC TẬP

### **Folder `docs/` sẽ có các file:**

```
docs/
├── ROADMAP_FULL_PROJECT.md              # ← File này (tổng quan)
│
├── 01_foundations/                       # GIAI ĐOẠN 1: Nền tảng
│   ├── 01_MACHINE_LEARNING_BASICS.md    # ML cơ bản cho time series
│   ├── 02_DEEP_LEARNING_BASICS.md       # DL cơ bản (Neural Networks)
│   └── 03_TIME_SERIES_FUNDAMENTALS.md   # Time series là gì?
│
├── 02_modeling/                          # GIAI ĐOẠN 2: Xây dựng models
│   ├── 01_BASELINE_MODELS.md            # ARIMA, GARCH, Linear
│   ├── 02_ML_MODELS.md                  # XGBoost, LightGBM, RF
│   ├── 03_LSTM_GRU.md                   # LSTM, GRU cho time series
│   ├── 04_TRANSFORMERS_LTSF.md          # Transformer cho LTSF
│   └── 05_ANOMALY_DETECTION.md          # Phát hiện bất thường
│
├── 03_multimodal/                        # GIAI ĐOẠN 3: Đa nguồn dữ liệu VN
│   ├── 01_NEWS_DATA_VIETNAM.md          # Crawl CafeF & VnExpress
│   ├── 02_VIETNAMESE_TEXT_PROCESSING.md # Vietnamese NLP & sentiment
│   ├── 03_EVENT_DETECTION.md            # Phát hiện sự kiện từ tin VN
│   └── 04_MULTIMODAL_FUSION.md          # Kết hợp price + Vietnamese text
│
├── 04_advanced/                          # GIAI ĐOẠN 4: Nâng cao
│   ├── 01_EVENT_AWARE_TRAINING.md       # Training với event weighting
│   ├── 02_REGIME_DETECTION.md           # Phát hiện regime change
│   ├── 03_TAIL_RISK_METRICS.md          # Metrics cho tail events
│   └── 04_EFFICIENT_XAI.md              # Explainability hiệu quả
│
├── 05_evaluation/                        # GIAI ĐOẠN 5: Đánh giá
│   ├── 01_METRICS_EVALUATION.md         # Metrics đánh giá models
│   ├── 02_BACKTESTING.md                # Backtesting strategies
│   └── 03_CASE_STUDIES.md               # Case studies thực tế
│
└── 06_paper_writing/                     # GIAI ĐOẠN 6: Viết paper
    ├── 01_RESEARCH_METHODOLOGY.md       # Phương pháp nghiên cứu
    ├── 02_EXPERIMENT_DESIGN.md          # Thiết kế thí nghiệm
    └── 03_PAPER_STRUCTURE.md            # Cấu trúc paper
```

---

## 🚀 LỘ TRÌNH HỌC TẬP (24 TUẦN = ~6 THÁNG)

**Tổng quan:**
- **Phase 1 (Tuần 1-2):** Foundations - ML & Time Series basics
- **Phase 2 (Tuần 3-12):** Modeling - Từ baseline đến SOTA
- **Phase 3 (Tuần 13-16):** Multi-source Data - Vietnamese news
- **Phase 4 (Tuần 17-20):** Advanced - Event-aware, Regime, XAI
- **Phase 5 (Tuần 21-24):** Evaluation - Metrics, backtesting, paper

---

### **TUẦN 1-2: Nền tảng ML & Time Series**

**Mục tiêu:** Hiểu cơ bản về ML và time series

**Học:**
- `01_foundations/01_MACHINE_LEARNING_BASICS.md`
- `01_foundations/03_TIME_SERIES_FUNDAMENTALS.md`
- `LEARNING_GUIDE_FULL_SYSTEM.md` (Section 5.2: Features)

**Làm:**
- Implement Linear Regression dự báo giá cổ phiếu
- Tính toán metrics: MSE, MAE, MAPE
- Đọc LEARNING_GUIDE Section 5.2.1-5.2.10: Technical Indicators
- Test các features trong `build_features.py`
- Visualize predictions vs actual

**Kiểm tra hiểu bài:**
- [ ] Giải thích được supervised learning là gì
- [ ] Phân biệt được regression vs classification
- [ ] Hiểu được train/test split
- [ ] Tính được MSE, MAE bằng tay
- [ ] **Hiểu Technical Indicators** (RSI, MACD, Bollinger, EMA, Momentum) → LEARNING_GUIDE section 5.2
- [ ] **Phân biệt Simple vs Log Returns**
- [ ] **Hiểu MA vs EMA, biết khi nào dùng gì**

---

### **TUẦN 3-4: Baseline Models**

**Mục tiêu:** Implement các baseline models

**Học:**
- `02_modeling/01_BASELINE_MODELS.md`

**Làm:**
- Implement ARIMA model
- Implement GARCH model (cho volatility)
- So sánh ARIMA vs Linear Regression
- Tạo file `src/models/baseline/arima.py`

**Kiểm tra hiểu bài:**
- [ ] Giải thích được ARIMA(p,d,q) là gì
- [ ] Biết khi nào dùng ARIMA, khi nào dùng GARCH
- [ ] Chạy được ARIMA trên data FPT
- [ ] So sánh được kết quả với Linear Regression

---

### **TUẦN 5-6: Machine Learning Models**

**Mục tiêu:** Implement ML models mạnh hơn baseline

**Học:**
- `02_modeling/02_ML_MODELS.md`

**Làm:**
- Implement XGBoost model
- Implement LightGBM model
- Feature importance analysis
- Hyperparameter tuning

**Kiểm tra hiểu bài:**
- [ ] Giải thích được decision tree là gì
- [ ] Hiểu được boosting vs bagging
- [ ] Tune được hyperparameters
- [ ] Phân tích được feature importance

---

### **TUẦN 7-8: Deep Learning (LSTM/GRU)**

**Mục tiêu:** Hiểu và implement LSTM cho time series

**Học:**
- `01_foundations/02_DEEP_LEARNING_BASICS.md`
- `02_modeling/03_LSTM_GRU.md`

**Làm:**
- Implement LSTM model từ đầu
- Implement GRU model
- Sequence-to-sequence prediction
- Compare LSTM vs XGBoost

**Kiểm tra hiểu bài:**
- [ ] Giải thích được LSTM cell hoạt động như thế nào
- [ ] Hiểu được vanishing gradient problem
- [ ] Chọn được window size, hidden size phù hợp
- [ ] Train được LSTM không bị overfitting

---

### **TUẦN 9-10: Transformers cho LTSF**

**Mục tiêu:** Implement Transformer models cho long-term forecasting

**Học:**
- `02_modeling/04_TRANSFORMERS_LTSF.md`

**Làm:**
- Implement iTransformer
- Implement TimesNet (nếu có thời gian)
- Benchmark: LSTM vs Transformer
- Analyze attention weights

**Kiểm tra hiểu bài:**
- [ ] Giải thích được self-attention là gì
- [ ] Hiểu được tại sao Transformer tốt cho LTSF
- [ ] Implement được multi-head attention
- [ ] Visualize được attention patterns

---

### **TUẦN 11-12: Anomaly Detection**

**Mục tiêu:** Phát hiện bất thường trong time series

**Học:**
- `02_modeling/05_ANOMALY_DETECTION.md`

**Làm:**
- Implement Anomaly Transformer
- Implement TranAD
- Detect anomalies trong VN30 data
- Visualize anomalies

**Kiểm tra hiểu bài:**
- [ ] Phân biệt được point anomaly vs contextual anomaly
- [ ] Implement được reconstruction-based anomaly detection
- [ ] Tune được threshold cho anomaly detection
- [ ] Validate anomalies với real events

---

### **TUẦN 13-14: Multi-source Data (News)**

**Mục tiêu:** Crawl và xử lý tin tức Việt Nam

**Học:**
- `03_multimodal/01_NEWS_DATA_VIETNAM.md` ← Cập nhật cho VN
- `03_multimodal/02_VIETNAMESE_TEXT_PROCESSING.md` ← Vietnamese NLP

**Làm:**
- Crawl tin tức từ **CafeF** (chứng khoán)
- Crawl tin tức từ **VnExpress** (kinh tế)
- Vietnamese sentiment analysis (PhoBERT, vn-sentiment)
- Link news với price data VN30
- Analyze correlation

**Kiểm tra hiểu bài:**
- [ ] Crawl được tin CafeF & VnExpress
- [ ] Xử lý được tiếng Việt (tokenization, dấu)
- [ ] Tính được sentiment score (Vietnamese)
- [ ] Phân tích được correlation giữa sentiment và price
- [ ] Filter được tin spam/clickbait

---

### **TUẦN 15-16: Event Detection & Multimodal Fusion**

**Mục tiêu:** Kết hợp price + Vietnamese news

**Học:**
- `03_multimodal/03_EVENT_DETECTION.md`
- `03_multimodal/04_MULTIMODAL_FUSION.md`

**Làm:**
- Detect event days từ tin tức VN (earnings, M&A, scandal)
- Classify event types và impact
- Implement cross-modal attention (price + text)
- Train multimodal model
- Compare với single-modal (price only)

**Kiểm tra hiểu bài:**
- [ ] Detect được event days từ CafeF & VnExpress
- [ ] Classify được event types
- [ ] Implement được cross-modal attention
- [ ] Train được multimodal model
- [ ] Chứng minh được multimodal tốt hơn single-modal

---

### **TUẦN 17-18: Event-Aware Training & Regime Detection**

**Mục tiêu:** Training với event weighting & detect regime change

**Học:**
- `04_advanced/01_EVENT_AWARE_TRAINING.md`
- `04_advanced/02_REGIME_DETECTION.md`

**Làm:**
- Implement weighted loss function cho event days
- Train model với event-aware loss
- Implement Hidden Markov Model (HMM) cho regime detection
- Detect regime changes trong VN30
- Compare: normal training vs event-aware vs regime-aware

**Kiểm tra hiểu bài:**
- [ ] Implement được weighted loss
- [ ] Chứng minh được event-aware training tốt hơn
- [ ] Giải thích được regime là gì
- [ ] Detect được regime changes trong historical data
- [ ] Train được separate models cho mỗi regime

---

### **TUẦN 19-20: Explainability (XAI)**

**Mục tiêu:** Giải thích predictions của models

**Học:**
- `04_advanced/04_EFFICIENT_XAI.md`

**Làm:**
- Implement SHAP explainer
- Implement TimeSHAP
- Implement efficient approximations
- Visualize explanations

**Kiểm tra hiểu bài:**
- [ ] Giải thích được SHAP values là gì
- [ ] Tính được SHAP values cho predictions
- [ ] Implement được efficient approximations
- [ ] Visualize được feature importance over time

---

### **TUẦN 21-22: Evaluation & Metrics**

**Mục tiêu:** Đánh giá toàn diện models

**Học:**
- `04_advanced/03_TAIL_RISK_METRICS.md`
- `05_evaluation/01_METRICS_EVALUATION.md`
- `05_evaluation/02_BACKTESTING.md`

**Làm:**
- Implement tail risk metrics (CVaR, Tail Loss)
- Backtesting framework
- Walk-forward validation
- Compare all models

**Kiểm tra hiểu bài:**
- [ ] Tính được CVaR, Maximum Drawdown
- [ ] Implement được walk-forward validation
- [ ] Backtest được trading strategy
- [ ] So sánh được models trên multiple metrics

---

### **TUẦN 23-24: Case Studies & Paper**

**Mục tiêu:** Hoàn thiện case studies và viết paper

**Học:**
- `05_evaluation/03_CASE_STUDIES.md`
- `06_paper_writing/01_RESEARCH_METHODOLOGY.md`
- `06_paper_writing/03_PAPER_STRUCTURE.md`

**Làm:**
- Case study: COVID crash (Feb-Mar 2020)
- Case study: Tech bubble (2021-2022)
- Write paper draft
- Create visualizations

**Kiểm tra hiểu bài:**
- [ ] Analyze được model performance trên specific events
- [ ] Explain được predictions với XAI
- [ ] Viết được methodology section
- [ ] Tạo được professional figures

---

## 🎓 CÁCH SỬ DỤNG TÀI LIỆU

### **Quy trình học mỗi tuần:**

```
1. ĐỌC (30 phút - 1 giờ)
   - Đọc file .md tương ứng
   - Ghi chú những điểm chưa hiểu
   - Xem thêm references nếu cần

2. HIỂU (1-2 giờ)
   - Vẽ sơ đồ, mindmap
   - Giải thích lại bằng lời của mình
   - Hỏi ChatGPT/Claude nếu chưa rõ

3. LÀM (3-5 giờ)
   - Code từng bước nhỏ
   - Test ngay từng function
   - Debug khi có lỗi

4. KIỂM TRA (30 phút)
   - Làm checklist "Kiểm tra hiểu bài"
   - Nếu chưa pass, quay lại bước 2
   - Nếu pass, sang tuần tiếp theo
```

### **Khi gặp khó khăn:**

1. **Không hiểu lý thuyết:**
   - Đọc lại phần "Giải thích đời thường" trong file .md
   - Xem video YouTube về topic đó
   - Hỏi ChatGPT/Claude với prompt cụ thể

2. **Code bị lỗi:**
   - Đọc error message kỹ
   - Print ra từng bước để debug
   - Tìm trên StackOverflow
   - Hỏi ChatGPT/Claude với full error message

3. **Kết quả không tốt:**
   - Kiểm tra lại data (có bị lỗi không?)
   - Kiểm tra lại hyperparameters
   - So sánh với baseline
   - Đọc papers để xem người khác làm như thế nào

---

## 📊 THEO DÕI TIẾN ĐỘ

### **Checklist tổng thể:**

```
PHASE 1: FOUNDATIONS (Tuần 1-2)
[ ] Hiểu ML basics
[ ] Hiểu time series fundamentals
[ ] Implement Linear Regression
[ ] Calculate metrics

PHASE 2: BASELINE MODELS (Tuần 3-4)
[ ] Implement ARIMA
[ ] Implement GARCH
[ ] Compare with Linear Regression

PHASE 3: ML MODELS (Tuần 5-6)
[ ] Implement XGBoost
[ ] Implement LightGBM
[ ] Feature importance analysis
[ ] Hyperparameter tuning

PHASE 4: DEEP LEARNING (Tuần 7-8)
[ ] Implement LSTM
[ ] Implement GRU
[ ] Compare with XGBoost

PHASE 5: TRANSFORMERS (Tuần 9-10)
[ ] Implement iTransformer
[ ] Benchmark vs LSTM
[ ] Analyze attention

PHASE 6: ANOMALY DETECTION (Tuần 11-12)
[ ] Implement Anomaly Transformer
[ ] Detect anomalies
[ ] Validate with events

PHASE 7: VIETNAMESE NEWS DATA (Tuần 13-14)
[ ] Crawl CafeF news
[ ] Crawl VnExpress news
[ ] Vietnamese sentiment analysis
[ ] Link news với price VN30

PHASE 8: MULTIMODAL FUSION (Tuần 15-16)
[ ] Event detection từ tin VN
[ ] Event classification
[ ] Cross-modal attention
[ ] Train multimodal model
[ ] Compare với single-modal

PHASE 9: EVENT-AWARE & REGIME (Tuần 17-18)
[ ] Weighted loss cho events
[ ] Event-aware training
[ ] Implement HMM
[ ] Detect regimes
[ ] Separate models

PHASE 10: XAI (Tuần 19-20)
[ ] Implement SHAP
[ ] Implement TimeSHAP
[ ] Efficient approximations

PHASE 11: EVALUATION (Tuần 21-22)
[ ] Tail risk metrics
[ ] Backtesting
[ ] Compare all models

PHASE 12: PAPER (Tuần 23-24)
[ ] Case studies
[ ] Write paper
[ ] Create figures
```

---

## 🎯 KẾT QUẢ MONG ĐỢI

Sau 24 tuần, bạn sẽ có:

1. **Hệ thống hoàn chỉnh:**
   - ✅ Crawl multi-source data (price VN30 + tin tức CafeF/VnExpress)
   - ✅ 10+ models (từ baseline đến SOTA: ARIMA, XGBoost, LSTM, Transformer)
   - ✅ Vietnamese sentiment analysis
   - ✅ Event-aware training mechanism
   - ✅ Regime detection system
   - ✅ XAI module (SHAP, TimeSHAP)

2. **Kiến thức vững:**
   - ✅ ML/DL fundamentals
   - ✅ Time series forecasting (ARIMA → LSTM → Transformer)
   - ✅ Anomaly detection (Anomaly Transformer, TranAD)
   - ✅ Vietnamese NLP (PhoBERT, sentiment analysis)
   - ✅ Multimodal fusion (cross-modal attention)
   - ✅ XAI methods (SHAP, TimeSHAP)

3. **Kết quả nghiên cứu:**
   - ✅ Benchmark results (10+ models trên VN30)
   - ✅ Case studies (COVID crash, tech bubble VN)
   - ✅ Paper draft về event-aware training cho VN stocks
   - ✅ Code repository (open-source ready)
   - ✅ Đóng góp: Vietnamese sentiment analysis cho finance
   - ✅ Đóng góp: Event-aware training cho emerging markets

4. **Kỹ năng:**
   - Implement models from scratch
   - Debug complex systems
   - Analyze results
   - Write research papers

---

## 💡 LỜI KHUYÊN

### **Đừng:**
- ❌ Học quá nhanh, không hiểu sâu
- ❌ Copy code mà không hiểu
- ❌ Bỏ qua checklist "Kiểm tra hiểu bài"
- ❌ Làm nhiều thứ cùng lúc

### **Nên:**
- ✅ Học từng bước, hiểu thật sâu
- ✅ Code từ đầu, debug từng lỗi
- ✅ Làm đủ checklist trước khi sang bước mới
- ✅ Focus vào 1 topic mỗi tuần

### **Nhớ:**
> "Học để hiểu, không phải để nhớ"
> "Code để làm, không phải để copy"
> "Debug để học, không phải để fix"

---

## 🚀 BẮT ĐẦU NGAY

**Bước tiếp theo của bạn:**

1. Đọc file `01_foundations/01_MACHINE_LEARNING_BASICS.md`
2. Làm bài tập trong đó
3. Kiểm tra hiểu bài
4. Sang `01_foundations/03_TIME_SERIES_FUNDAMENTALS.md`

**Chúc bạn học tốt! 🎓**

---

*Cập nhật lần cuối: 2026-01-28*
