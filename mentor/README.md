# Mentor – Giải thích code & hướng đi dự án

Thư mục này **chỉ dùng để dạy / giải thích**. Không chỉnh sửa gì trong `docs/` hay source code; toàn bộ nội dung nằm trong `mentor/`.

## Mục tiêu dự án (tóm tắt)

- **Forecasting + anomaly detection** cho cổ phiếu VN30.
- **So sánh 5 loại model:** Transformer-based, Patch-based, LSTM-based, classical ML (và baseline).
- **Multi-step forecasting:** 1, 5, 20 ngày.
- **Anomaly detection** và (sau này) **event-aware** / news.

## Các file trong `mentor/`

| File | Nội dung |
|------|----------|
| [01_WHAT_HAPPENS_IN_SOURCE_FILES.md](01_WHAT_HAPPENS_IN_SOURCE_FILES.md) | Giải thích từng phần trong `src/models/ml.py`, `src/features/build_features.py`, pipeline VN30 – **chuyện gì xảy ra trong từng file**. |
| [02_PROJECT_GOAL_AND_NEXT_STEPS.md](02_PROJECT_GOAL_AND_NEXT_STEPS.md) | Ánh xạ mục tiêu (5 models, multi-step, anomaly, event) với code hiện tại và các bước làm tiếp. |
| [03_FORECASTING_MODELS_EXPLAINED.md](03_FORECASTING_MODELS_EXPLAINED.md) | Giải thích từng model: **tại sao chọn**, **inductive bias**, **khi nào fail**. (Cùng pipeline: 5 models, cùng split/metrics/backtest.) |
| [04_NEWS_PIPELINE_ARCHITECTURE.md](04_NEWS_PIPELINE_ARCHITECTURE.md) | **Tin tức VN:** kiến trúc pipeline (crawl → clean → sentiment → align → DB), cách chuyển sang real-time, layout code. |
| [05_WEB_APP_ARCHITECTURE.md](05_WEB_APP_ARCHITECTURE.md) | **Web app:** backend ML (aggregation → recommendation → explanation), UX, chuyển model output → khuyến nghị rủi ro, Streamlit/FastAPI. |
| [06_PROJECT_STRUCTURE_AND_HOW_TO_RUN.md](06_PROJECT_STRUCTURE_AND_HOW_TO_RUN.md) | **Cấu trúc repo + cách chạy:** giải thích từng folder trong `src/`, kỹ thuật chạy thế nào, input/output, lỗi thường gặp; hướng dẫn chạy project từng bước. |

Đọc theo thứ tự 01 → 02 sẽ rõ: code hiện tại làm gì, thiếu gì, và làm gì tiếp. Pipeline forecasting: `scripts/run_forecasting_pipeline.py`. Pipeline tin tức: `scripts/run_news_pipeline.py`. Web app: `uvicorn api:app --reload` (mở http://localhost:8000). Cấu trúc và cách chạy: **06_PROJECT_STRUCTURE_AND_HOW_TO_RUN.md**.
