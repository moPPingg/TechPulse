# 📝 CHANGELOG - VIETNAM FOCUS UPDATE
## Cập nhật dự án sang hướng chứng khoán Việt Nam

**Ngày cập nhật:** 28/01/2026

---

## 🎯 QUYẾT ĐỊNH CHIẾN LƯỢC

### **Trước đây (Global approach):**
- ❌ SEC filings (US stocks)
- ❌ GDELT (global news)
- ❌ FRED (US macro data)
- ❌ English NLP

### **Bây giờ (Vietnam focus):**
- ✅ CafeF (price data + tin tức chứng khoán VN)
- ✅ VnExpress (tin tức kinh tế VN)
- ✅ Vietnamese NLP & sentiment analysis
- ✅ VN30 stocks focus

---

## 📊 TẠI SAO THAY ĐỔI?

### **Ưu điểm:**

1. **Dễ tiếp cận:**
   - CafeF & VnExpress dễ crawl
   - Không cần API key tốn phí
   - Không bị rate limit nghiêm ngặt

2. **Phù hợp thị trường:**
   - VN30 tech stocks → Tin Việt quan trọng hơn
   - Nhà đầu tư Việt đọc tin Việt
   - Impact trực tiếp, không bị lag

3. **Đóng góp nghiên cứu:**
   - Ít paper làm về VN stocks
   - Vietnamese NLP cho finance là mới
   - Emerging markets research

4. **Đơn giản hóa:**
   - Bỏ SEC filings (phức tạp, không cần)
   - Bỏ GDELT (global, ít liên quan)
   - Focus vào 2 nguồn chất lượng

### **Thách thức:**

1. **Vietnamese NLP:**
   - Khó hơn English NLP
   - Ít tools/models pre-trained
   - Cần xử lý tiếng Việt có dấu

2. **Data quality:**
   - Tin tức có thể thiên lệch
   - Cần filter spam, clickbait

3. **Macro data:**
   - Khó lấy macro data VN
   - API ít, phải crawl nhiều nguồn

---

## 📁 FILES ĐÃ CẬP NHẬT

### **1. ROADMAP_FULL_PROJECT.md**

**Thay đổi chính:**

```diff
- Multi-source data (price + news + filings + macro)
+ Multi-source data Vietnam (price + CafeF news + VnExpress news)

- GDELT (global news)
+ CafeF & VnExpress (Vietnamese news)

- SEC filings
+ Vietnamese sentiment analysis

- Text processing (English)
+ Vietnamese text processing (PhoBERT)
```

**Cấu trúc mới:**

```
Phase 1 (25%): Data Foundation ✅ HOÀN THÀNH
├─ Price data từ CafeF (VN30)
├─ Clean & validate
├─ 45+ technical features
└─ Pipeline automation

Phase 2 (35%): Modeling & Benchmark ⏳ CHƯA BẮT ĐẦU
├─ Baseline (ARIMA, GARCH)
├─ ML (XGBoost, LightGBM)
├─ DL (LSTM, GRU, Transformer)
└─ Anomaly detection

Phase 3 (20%): Multi-source Data - Vietnam ⏳ CHƯA BẮT ĐẦU
├─ Crawl CafeF & VnExpress news
├─ Vietnamese NLP & sentiment
├─ Event detection
└─ Multimodal fusion

Phase 4 (15%): Advanced Methods ⏳ CHƯA BẮT ĐẦU
├─ Event-aware training
├─ Regime detection
└─ Efficient XAI

Phase 5 (5%): Evaluation & Paper ⏳ CHƯA BẮT ĐẦU
├─ Tail risk metrics
├─ Backtesting
└─ Case studies & paper
```

---

### **2. INDEX.md**

**Thay đổi:**

```diff
Phase 3: Multi-Modal Data
- 01_NEWS_DATA.md → Crawl GDELT, VN news
+ 01_NEWS_DATA_VIETNAM.md → Crawl CafeF & VnExpress

- 02_TEXT_PROCESSING.md → NLP, sentiment analysis
+ 02_VIETNAMESE_TEXT_PROCESSING.md → Vietnamese NLP & sentiment

- 03_EVENT_DETECTION.md → Detect events from news + price
+ 03_EVENT_DETECTION.md → Detect events from Vietnamese news + price

- 04_MULTIMODAL_FUSION.md → Combine price + text
+ 04_MULTIMODAL_FUSION.md → Combine price + Vietnamese text
```

**Checklist mới:**

```
Vietnamese News Data (Tuần 13-14):
- [ ] Crawl tin từ CafeF (chứng khoán)
- [ ] Crawl tin từ VnExpress (kinh tế)
- [ ] Xử lý tiếng Việt (tokenization, dấu)
- [ ] Vietnamese sentiment analysis (PhoBERT)
- [ ] Link news với price VN30
- [ ] Filter spam/clickbait
- [ ] Analyze correlation
```

---

### **3. QUICK_START.md**

**Thay đổi:**

```diff
Mục tiêu cuối cùng:
- Xây dựng hệ thống dự báo giá cổ phiếu
+ Xây dựng hệ thống dự báo giá cổ phiếu **Việt Nam**

Đặc điểm nổi bật:
1. Event-Aware Training
- 2. Regime Detection
+ 2. Vietnamese News Integration (CafeF & VnExpress)
3. Efficient XAI

Đóng góp nghiên cứu:
- Ít paper làm điều này
+ - Ít paper làm về VN stocks
+ - Vietnamese sentiment analysis cho finance
+ - Event-aware training cho emerging markets
```

---

### **4. LEARNING_GUIDE_FULL_SYSTEM.md**

**Thay đổi phần PROPOSAL:**

```diff
Bước tiếp theo:

1. Thêm nguồn dữ liệu:
-   - SEC EDGAR (báo cáo tài chính Mỹ)
-   - FRED (dữ liệu vĩ mô)
-   - GDELT (tin tức)
+   - ✅ CafeF News (tin tức chứng khoán VN)
+   - ✅ VnExpress (tin tức kinh tế VN)
+   - ⏳ Vietnamese sentiment analysis (PhoBERT)
+   - ⏳ Macro data VN (nếu có API)

+ 4. Vietnamese NLP & Multimodal:
+    - Vietnamese text processing
+    - Sentiment analysis (PhoBERT)
+    - Event detection từ tin VN
+    - Multimodal fusion
+    - Cross-modal attention

+ 5. Event-Aware Training (PAIN POINT):
+    - Detect event days
+    - Weighted loss function
+    - Shock-focused metrics
+    - Compare normal vs event-aware

+ 6. Regime Detection:
+    - Hidden Markov Model
+    - Detect regime changes
+    - Separate models

+ 7. Efficient XAI:
+    - SHAP, TimeSHAP
+    - Efficient approximations
```

---

## 📄 FILES MỚI TẠO

### **1. 03_multimodal/01_NEWS_DATA_VIETNAM.md** ✅

**Nội dung:**
- Tại sao chọn CafeF & VnExpress
- Kiến trúc crawler
- Implementation chi tiết:
  - CafeF crawler (class + methods)
  - VnExpress crawler (class + methods)
- Data schema
- Best practices:
  - Ethical crawling
  - Rate limiting
  - Error handling
  - Scheduling
- 3 bài tập thực hành

**Highlights:**
```python
class CafeFNewsCrawler:
    def crawl_news_list(self, category, pages)
    def crawl_article_content(self, url)
    
class VnExpressNewsCrawler:
    def crawl_news_list(self, category, pages)
    def crawl_article_content(self, url)
```

---

## 🎯 ROADMAP CẬP NHẬT

### **Tuần 13-14: Vietnamese News Data**

**Mục tiêu:** Crawl và xử lý tin tức Việt Nam

**Học:**
- `03_multimodal/01_NEWS_DATA_VIETNAM.md` ✅ Đã tạo
- `03_multimodal/02_VIETNAMESE_TEXT_PROCESSING.md` ⏳ Chưa tạo

**Làm:**
- Crawl CafeF (chứng khoán)
- Crawl VnExpress (kinh tế)
- Vietnamese sentiment analysis
- Link news với VN30

---

### **Tuần 15-16: Event Detection & Multimodal Fusion**

**Mục tiêu:** Kết hợp price + Vietnamese news

**Học:**
- `03_multimodal/03_EVENT_DETECTION.md` ⏳ Chưa tạo
- `03_multimodal/04_MULTIMODAL_FUSION.md` ⏳ Chưa tạo

**Làm:**
- Detect events từ tin VN
- Classify event types
- Cross-modal attention
- Train multimodal model

---

### **Tuần 17-18: Event-Aware & Regime**

**Mục tiêu:** Training với event weighting & regime detection

**Học:**
- `04_advanced/01_EVENT_AWARE_TRAINING.md` ✅ Đã tạo
- `04_advanced/02_REGIME_DETECTION.md` ⏳ Chưa tạo

**Làm:**
- Weighted loss cho events
- Event-aware training
- HMM cho regime detection
- Compare methods

---

## 📊 TIẾN ĐỘ TỔNG THỂ

### **Files đã tạo: 9/30**

```
✅ ROADMAP_FULL_PROJECT.md
✅ INDEX.md
✅ QUICK_START.md
✅ 01_foundations/01_MACHINE_LEARNING_BASICS.md
✅ 01_foundations/02_DEEP_LEARNING_BASICS.md
✅ 01_foundations/03_TIME_SERIES_FUNDAMENTALS.md
✅ 02_modeling/01_BASELINE_MODELS.md
✅ 04_advanced/01_EVENT_AWARE_TRAINING.md
✅ 03_multimodal/01_NEWS_DATA_VIETNAM.md ← MỚI
✅ LEARNING_GUIDE_FULL_SYSTEM.md (cập nhật)
✅ CHANGELOG_VIETNAM_FOCUS.md ← FILE NÀY
```

### **Files cần tạo: 21**

**Phase 2 - Modeling (4 files):**
- ⏳ 02_modeling/02_ML_MODELS.md
- ⏳ 02_modeling/03_LSTM_GRU.md
- ⏳ 02_modeling/04_TRANSFORMERS_LTSF.md
- ⏳ 02_modeling/05_ANOMALY_DETECTION.md

**Phase 3 - Multimodal Vietnam (3 files):**
- ⏳ 03_multimodal/02_VIETNAMESE_TEXT_PROCESSING.md
- ⏳ 03_multimodal/03_EVENT_DETECTION.md
- ⏳ 03_multimodal/04_MULTIMODAL_FUSION.md

**Phase 4 - Advanced (3 files):**
- ⏳ 04_advanced/02_REGIME_DETECTION.md
- ⏳ 04_advanced/03_TAIL_RISK_METRICS.md
- ⏳ 04_advanced/04_EFFICIENT_XAI.md

**Phase 5 - Evaluation (3 files):**
- ⏳ 05_evaluation/01_METRICS_EVALUATION.md
- ⏳ 05_evaluation/02_BACKTESTING.md
- ⏳ 05_evaluation/03_CASE_STUDIES.md

**Phase 6 - Paper (3 files):**
- ⏳ 06_paper_writing/01_RESEARCH_METHODOLOGY.md
- ⏳ 06_paper_writing/02_EXPERIMENT_DESIGN.md
- ⏳ 06_paper_writing/03_PAPER_STRUCTURE.md

---

## 🚀 BƯỚC TIẾP THEO

### **Ngay bây giờ:**

1. **Đọc file mới:**
   - `03_multimodal/01_NEWS_DATA_VIETNAM.md`

2. **Implement crawlers:**
   - CafeF news crawler
   - VnExpress news crawler

3. **Làm bài tập:**
   - Bài 1: Crawl 100 tin CafeF
   - Bài 2: Crawl 7 ngày VnExpress
   - Bài 3: Combined crawler + deduplicate

### **Tuần này:**

1. Crawl được 1000+ tin từ CafeF
2. Crawl được 500+ tin từ VnExpress
3. Save vào database (CSV/SQLite)
4. Analyze data quality

### **Tuần tới:**

1. Tạo file `02_VIETNAMESE_TEXT_PROCESSING.md`
2. Implement Vietnamese tokenization
3. Implement sentiment analysis (PhoBERT)
4. Link news với price VN30

---

## 💡 KHUYẾN NGHỊ

### **Focus vào:**

1. ✅ **CafeF + VnExpress** (2 nguồn chính)
2. ✅ **Vietnamese NLP** (PhoBERT, underthesea)
3. ✅ **Event-aware training** (pain point chính)
4. ✅ **VN30 stocks** (30 cổ phiếu lớn nhất)

### **Có thể bỏ qua:**

1. ❌ SEC filings (cho US stocks)
2. ❌ GDELT (global news)
3. ❌ FRED (US macro)
4. ❌ English NLP

### **Có thể thêm sau (optional):**

1. ⏳ Cafebiz (tin doanh nghiệp)
2. ⏳ Đầu tư (phân tích chuyên sâu)
3. ⏳ Vneconomy (kinh tế vĩ mô)

---

## 📚 TÀI LIỆU THAM KHẢO

### **Vietnamese NLP:**
- PhoBERT: https://github.com/VinAIResearch/PhoBERT
- underthesea: https://github.com/undertheseanlp/underthesea
- pyvi: https://github.com/trungtv/pyvi
- vncorenlp: https://github.com/vncorenlp/VnCoreNLP

### **Web Scraping:**
- BeautifulSoup4: https://www.crummy.com/software/BeautifulSoup/
- Scrapy: https://scrapy.org/
- Selenium: https://selenium-python.readthedocs.io/

### **Research:**
- Event-aware training papers
- Emerging markets finance
- Vietnamese sentiment analysis

---

## ✅ SUMMARY

**Quyết định:** Tập trung vào chứng khoán Việt Nam với tin tức từ CafeF & VnExpress

**Lý do:**
- ✅ Dễ tiếp cận
- ✅ Phù hợp thị trường
- ✅ Đóng góp nghiên cứu mới
- ✅ Đơn giản hóa pipeline

**Đã cập nhật:**
- ✅ ROADMAP_FULL_PROJECT.md
- ✅ INDEX.md
- ✅ QUICK_START.md
- ✅ LEARNING_GUIDE_FULL_SYSTEM.md

**Đã tạo mới:**
- ✅ 01_NEWS_DATA_VIETNAM.md
- ✅ CHANGELOG_VIETNAM_FOCUS.md

**Tiếp theo:**
- ⏳ 02_VIETNAMESE_TEXT_PROCESSING.md
- ⏳ Implement crawlers
- ⏳ Vietnamese sentiment analysis

---

**Cập nhật bởi:** AI Assistant  
**Ngày:** 28/01/2026  
**Version:** 1.0 - Vietnam Focus
