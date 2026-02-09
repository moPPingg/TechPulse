# AI-Powered News Analysis

## Vấn đề hiện tại

Tin tức hiện đang được xử lý **hoàn toàn bằng rule/keyword**:

| Thành phần | Cách hiện tại | Hạn chế |
|------------|---------------|---------|
| **Sentiment** | Đếm từ tích cực/tiêu cực (lexicon) | Không hiểu ngữ cảnh: "không giảm" vs "giảm" |
| **Event type** | So khớp keyword (earnings, legal, macro...) | Sai loại tin, không nhận tin phức tạp |
| **Ticker relevance** | Regex match mã cổ phiếu | Không phân biệt mức độ liên quan thực sự |
| **Impact** | Keyword + ngày đăng | Không đánh giá ảnh hưởng thực tế lên giá |

→ **Kết quả**: Phân loại kém, dự đoán dựa trên tin không chính xác.

## Giải pháp: AI đọc tin và phân biệt

### Option 1: LLM API (OpenAI, Anthropic, Google...)

**Ý tưởng**: Gửi từng bài (hoặc batch) cho LLM, yêu cầu trả về structured output:
- `sentiment`: -1..1
- `event_type`: earnings | legal | macro | operations | guidance | ma | dividend | other
- `ticker_relevance`: 0..1
- `impact_summary`: Một câu giải thích tại sao tin quan trọng

**Ưu**: Chất lượng cao, hiểu tiếng Việt, không cần train model  
**Nhược**: Chi phí API, độ trễ, cần API key

### Option 2: PhoBERT / Vietnamese BERT

**Ý tưởng**: Dùng mô hình pre-trained tiếng Việt, fine-tune trên dataset tin tài chính đã gắn nhãn.

**Ưu**: Chạy local, không tốn API, có thể fine-tune  
**Nhược**: Cần dữ liệu gắn nhãn, tài nguyên tính toán

### Option 3: Hybrid (Khuyến nghị)

1. **Giữ** crawl, clean, align như hiện tại
2. **Thêm** bước enrich bằng LLM khi có API key
3. **Fallback** rule-based khi không có LLM hoặc lỗi
4. **Tùy chọn** bật/tắt qua config

```
crawl → clean → sentiment(lexicon) → align → enrich
                                              ↑
                                    [LLM enrich nếu config.enabled]
                                              ↓
                                    event_type, sentiment_refined, impact_summary
```

## Implementation: Module LLM Enrichment

Đã thêm `src/news/llm_enrich.py`:
- Gọi OpenAI/Anthropic (cấu hình qua env)
- Prompt tiếng Việt, output JSON
- Batch để giảm chi phí
- Có thể ghi đè sentiment + event_type + thêm `impact_summary`

## Cấu hình

Trong `configs/news.yaml`:

```yaml
llm_enrich:
  enabled: false          # Bật khi có API key
  provider: openai        # openai | anthropic
  model: gpt-4o-mini      # Rẻ, đủ tốt cho enrich
  batch_size: 5
  max_tokens: 200
```

Biến môi trường:
- `OPENAI_API_KEY` hoặc `ANTHROPIC_API_KEY`

## Kích hoạt

1. Thêm API key vào env
2. Đặt `llm_enrich.enabled: true` trong news.yaml
3. Chạy pipeline: `python scripts/run_news_pipeline.py`

Pipeline sẽ gọi LLM cho các bài chưa có enrichment từ LLM, ghi kết quả vào `article_enrichments` (có thể mở rộng schema để lưu `impact_summary`).
