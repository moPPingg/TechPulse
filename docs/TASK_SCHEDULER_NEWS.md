# Lên lịch cào tin tức mỗi 2–4 giờ (Windows Task Scheduler)

## Cách 1: Create Task (khuyên dùng)

1. Mở **Task Scheduler** (gõ `taskschd.msc` trong Run)
2. Bên phải, chọn **Create Task** (không dùng Create Basic Task)
3. Tab **General**: đặt tên, chọn "Run whether user is logged on or not"
4. Tab **Triggers** → **New**:
   - Begin the task: **On a schedule**
   - Settings: **Daily**
   - Start: chọn ngày giờ bắt đầu (vd: 8:00 sáng)
   - **Repeat task every**: chọn **2 hours** (hoặc 4 hours)
   - **for a duration of**: **Indefinitely**
5. Tab **Actions** → **New**:
   - Action: **Start a program**
   - Program: `python`
   - Add arguments: `run_news_pipeline.py`
   - Start in: `D:\techpulse` (đường dẫn thư mục gốc project)
6. OK → Save

## Cách 2: Create Basic Task + chỉnh Trigger

1. **Create Basic Task** → tên "News Crawl"
2. Trigger: **Daily**, chọn giờ (vd 8:00)
3. Action: **Start a program**
   - Program: `python`
   - Arguments: `run_news_pipeline.py`
   - Start in: `D:\techpulse`
4. Sau khi tạo xong → **Task Scheduler Library** → chuột phải task → **Properties**
5. Tab **Triggers** → **Edit**
6. Bật **Repeat task every** → chọn **2 hours**
7. **for a duration of**: **Indefinitely**
8. OK

## Cách 3: Chạy Python liên tục (terminal)

```bash
cd d:\techpulse
python scripts/run_news_pipeline.py --schedule 2
```

Để chạy nền: mở CMD/PowerShell riêng, chạy lệnh trên, cửa sổ giữ mở.
