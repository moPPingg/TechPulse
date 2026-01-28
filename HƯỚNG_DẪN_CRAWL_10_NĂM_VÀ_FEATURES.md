# 📘 HƯỚNG DẪN CRAWL DỮ LIỆU 10 NĂM & GIẢI THÍCH FEATURES

---

## PHẦN 1: CRAWL DỮ LIỆU 10 NĂM

### 1.1. Tại sao cần dữ liệu 10 năm?

**Lý do thực tế:**

| Số lượng dữ liệu | Sử dụng cho |
|------------------|-------------|
| 1-2 năm (250-500 dòng) | Dự báo ngắn hạn, backtest đơn giản |
| 5 năm (1,250 dòng) | Học pattern dài hạn, kiểm tra qua nhiều chu kỳ |
| **10 năm (2,500 dòng)** | ✅ ML models mạnh, bao quát nhiều tình huống thị trường |

**Dữ liệu 10 năm giúp:**
- Bao quát nhiều chu kỳ kinh tế (tăng trưởng, khủng hoảng, phục hồi)
- ML models học được pattern đa dạng hơn
- Tránh overfitting (học vẹt) - model sẽ tổng quát hơn
- Backtest chiến lược trading đáng tin cậy hơn

### 1.2. Cách crawl 10 năm với code hiện tại

**Code mẫu đơn giản:**

```python
from src.crawl.cafef_scraper import fetch_price_cafef

# Crawl FPT từ 01/01/2015 đến 31/12/2024 (10 năm)
df = fetch_price_cafef(
    symbol='FPT',
    start_date='01/01/2015',  # ← Thay đổi ngày bắt đầu
    end_date='31/12/2024',     # ← Ngày kết thúc
    page_size=3000,            # ← Tăng lên vì có nhiều dữ liệu hơn
    timeout=60                 # ← Tăng timeout vì request lớn hơn
)

# Lưu vào file
df.to_csv('data/raw/vn30/FPT.csv', index=False)
print(f"Đã lấy {len(df)} dòng dữ liệu (khoảng {len(df)/250:.1f} năm)")
```

**Giải thích các tham số:**

```python
page_size=3000
# Tại sao 3000?
# - 1 năm có ~250 ngày giao dịch (trừ thứ 7, CN, lễ)
# - 10 năm = 250 × 10 = 2,500 dòng
# - Để an toàn, đặt 3000 (dư một chút)

timeout=60
# Tại sao 60 giây?
# - Request lớn → server xử lý lâu hơn
# - Mạng chậm → cần thời gian tải
# - 60s đủ cho request lớn nhất
```

### 1.3. Crawl 10 năm cho toàn bộ VN30

**Tạo file mới: `crawl_vn30_10years.py`**

```python
# -*- coding: utf-8 -*-
"""
Script để crawl dữ liệu 10 năm cho VN30
Chạy file này để lấy data từ 2015-2024
"""

from src.pipeline.vnindex30.fetch_vn30 import run_vn30_pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def crawl_10_years():
    """
    Crawl 10 năm dữ liệu cho 30 mã VN30
    
    Thời gian chạy dự kiến: 10-15 phút
    Dữ liệu output: ~2,500 dòng/mã × 30 mã = 75,000 dòng
    """
    
    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU CRAWL DỮ LIỆU 10 NĂM")
    logger.info("=" * 80)
    logger.info("📅 Thời gian: 01/01/2015 → 31/12/2024")
    logger.info("📊 Số mã: 30 (VN30)")
    logger.info("⏱️  Thời gian dự kiến: 10-15 phút")
    logger.info("")
    logger.info("⚠️  LƯU Ý:")
    logger.info("  - Đảm bảo kết nối Internet ổn định")
    logger.info("  - Không tắt máy trong quá trình chạy")
    logger.info("  - Nếu bị lỗi, script sẽ bỏ qua mã đó và tiếp tục")
    logger.info("")
    
    # Chạy pipeline
    run_vn30_pipeline(
        start_date='01/01/2015',  # ← 10 năm trước
        end_date='31/12/2024',
        raw_dir='data/raw/vn30',
        clean_dir='data/clean/vn30',
        features_dir='data/features/vn30'
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 HOÀN THÀNH!")
    logger.info("=" * 80)
    logger.info("📁 Kiểm tra dữ liệu tại:")
    logger.info("   - Raw:      data/raw/vn30/")
    logger.info("   - Clean:    data/clean/vn30/")
    logger.info("   - Features: data/features/vn30/")

if __name__ == "__main__":
    crawl_10_years()
```

**Cách chạy:**
```powershell
# Bước 1: Kích hoạt môi trường
cd "W:\TECH STOCKS"
.\venv\Scripts\Activate.ps1

# Bước 2: Chạy script
python crawl_vn30_10years.py
```

### 1.4. Xử lý khi crawl bị lỗi

**Vấn đề thường gặp:**

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `Timeout` | Request quá lớn | Tăng `timeout=120` |
| `No data returned` | Mã không có dữ liệu trước năm X | Bình thường, bỏ qua |
| `Connection Error` | Mất mạng | Chạy lại script |

**Code xử lý lỗi thông minh:**

```python
from src.crawl.cafef_scraper import fetch_price_cafef
import time

def crawl_with_retry(symbol, start_date, end_date, max_retries=3):
    """
    Crawl với cơ chế retry (thử lại nếu lỗi)
    
    Args:
        symbol: Mã cổ phiếu
        start_date: Ngày bắt đầu
        end_date: Ngày kết thúc
        max_retries: Số lần thử lại tối đa
    
    Returns:
        DataFrame hoặc None nếu thất bại
    """
    for attempt in range(max_retries):
        try:
            print(f"[{symbol}] Attempt {attempt + 1}/{max_retries}...")
            
            df = fetch_price_cafef(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                page_size=3000,
                timeout=60
            )
            
            print(f"[{symbol}] ✅ Success! {len(df)} records")
            return df
            
        except Exception as e:
            print(f"[{symbol}] ❌ Error: {e}")
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                print(f"[{symbol}] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"[{symbol}] Failed after {max_retries} attempts")
                return None

# Sử dụng
df = crawl_with_retry('FPT', '01/01/2015', '31/12/2024')
if df is not None:
    df.to_csv('data/raw/vn30/FPT.csv', index=False)
```

---

## PHẦN 2: GIẢI THÍCH FOLDER `data/features/` - TRỌNG TÂM!

### 2.1. Tổng quan: Raw → Clean → Features

```
┌─────────────────────────────────────────────────────────────┐
│                    LUỒNG DỮ LIỆU                            │
└─────────────────────────────────────────────────────────────┘

data/raw/vn30/FPT.csv (Dữ liệu thô)
├── 7 cột cơ bản
├── Có thể có lỗi (trùng, null, giá sai)
└── Chưa thể dùng cho ML
      │
      ▼ (Qua module CLEAN)
      
data/clean/vn30/FPT.csv (Dữ liệu sạch)
├── 7 cột cơ bản (giống raw)
├── KHÔNG có lỗi (đã validate)
├── Sắp xếp theo thời gian
└── Sẵn sàng tính features
      │
      ▼ (Qua module FEATURES)
      
data/features/vn30/FPT.csv (Dữ liệu có đặc trưng)
├── 45+ cột (7 gốc + ~38 features mới)
├── Sẵn sàng cho Machine Learning
└── Chứa tất cả thông tin cần thiết để dự báo
```

### 2.2. Tại sao cần folder Features?

**Ví dụ thực tế:**

Giả sử bạn muốn dự đoán "ngày mai giá FPT sẽ tăng hay giảm?"

**❌ Nếu chỉ có raw data:**
```python
# Raw data chỉ có:
date: 2024-01-15
open: 100,000
close: 102,000

# ML sẽ nghĩ:
# "102,000 là con số lớn hay nhỏ?"
# "Nó đang tăng hay giảm?"
# "Xu hướng là gì?"
# → KHÔNG BIẾT! Chỉ là con số khô khan
```

**✅ Nếu có features data:**
```python
# Features data có:
date: 2024-01-15
close: 102,000
return_1d: +2.0%          ← "Hôm qua tăng 2%"
return_5d: +8.5%          ← "5 ngày tăng 8.5%"
ma_20: 95,000             ← "Trên MA20 → xu hướng tăng"
rsi_14: 68                ← "Gần vùng quá mua"
macd_hist: 0.5            ← "Động lượng dương"
volatility_20: 2.3%       ← "Biến động thấp = ổn định"

# ML sẽ hiểu:
# "Đang trong xu hướng tăng mạnh"
# "Nhưng gần vùng quá mua"
# "Biến động thấp = rủi ro thấp"
# → DỰ ĐOÁN: Có thể tăng thêm chút rồi điều chỉnh
```

### 2.3. Chi tiết các cột trong Features Data

Khi mở file `data/features/vn30/FPT.csv`, bạn sẽ thấy ~45 cột:

#### **NHÓM 1: Dữ liệu gốc (7 cột)**

```csv
date,open,high,low,close,volume,ticker
2024-01-15,100000,105000,99000,102000,1500000,FPT
```

| Cột | Ý nghĩa |
|-----|---------|
| `date` | Ngày giao dịch |
| `open` | Giá mở cửa |
| `high` | Giá cao nhất trong ngày |
| `low` | Giá thấp nhất trong ngày |
| `close` | Giá đóng cửa |
| `volume` | Khối lượng giao dịch (số cổ phiếu) |
| `ticker` | Mã cổ phiếu (FPT) |

#### **NHÓM 2: Returns - Lợi nhuận (4 cột)**

```python
return_1d   # Lợi nhuận 1 ngày
return_5d   # Lợi nhuận 5 ngày
return_10d  # Lợi nhuận 10 ngày
return_20d  # Lợi nhuận 20 ngày
```

**Ví dụ thực tế:**
```
Ngày:         1        2        3        4        5
Close:     100K     102K     105K     103K     108K
return_1d:   -      +2.0%    +2.9%    -1.9%    +4.9%
return_5d:   -        -        -        -      +8.0%

Cách đọc return_5d = +8.0% ở ngày 5:
→ "So với 5 ngày trước (ngày 1), giá tăng 8%"
→ (108 - 100) / 100 × 100% = 8%
```

**Tác dụng:**
- ML học pattern "nếu 5 ngày trước tăng X%, hôm nay có xu hướng Y%"
- Phát hiện momentum (động lượng)
- Xác định xu hướng ngắn/trung/dài hạn

#### **NHÓM 3: Moving Averages - Trung bình động (4 cột)**

```python
ma_5    # Trung bình 5 ngày (1 tuần)
ma_10   # Trung bình 10 ngày (2 tuần)
ma_20   # Trung bình 20 ngày (1 tháng)
ma_50   # Trung bình 50 ngày (2.5 tháng)
```

**Ví dụ thực tế:**
```
Ngày:     1     2     3     4     5     6  ...  20
Close:   100   102   104   103   105   107 ...  110

ma_5 (ngày 20) = TB(5 ngày gần nhất)
                = (110 + 109 + 108 + 107 + 106) / 5
                = 108

ma_20 (ngày 20) = TB(20 ngày gần nhất)
                 = (110 + 109 + ... + 100) / 20
                 = 105
```

**Cách đọc:**
```
Nếu close > ma_20:
→ Giá đang cao hơn trung bình 1 tháng
→ Xu hướng tăng (uptrend)

Nếu close < ma_20:
→ Giá đang thấp hơn trung bình
→ Xu hướng giảm (downtrend)

Nếu ma_5 cắt lên ma_20 (Golden Cross):
→ Tín hiệu mua mạnh

Nếu ma_5 cắt xuống ma_20 (Death Cross):
→ Tín hiệu bán mạnh
```

**Tác dụng:**
- Xác định xu hướng
- Tìm điểm vào/ra
- ML học pattern "khi giá gần MA, thường có phản ứng như thế nào"

#### **NHÓM 4: EMA - Trung bình trọng số (2 cột)**

```python
ema_12  # EMA 12 ngày
ema_26  # EMA 26 ngày
```

**Khác biệt MA vs EMA:**
```
MA (Simple Moving Average):
- Tất cả ngày có trọng số bằng nhau
- Ví dụ: MA(5) = (100 + 102 + 104 + 103 + 105) / 5

EMA (Exponential Moving Average):
- Ngày gần đây có trọng số lớn hơn
- Ví dụ: EMA(5) = ngày hôm nay × 40% + hôm qua × 30% + ...
                  (tỷ lệ giảm dần theo công thức exponential)

Ưu điểm EMA:
✅ Phản ứng nhanh hơn với thay đổi giá
✅ Phù hợp cho trading ngắn hạn
```

**Tác dụng:**
- Dùng cho MACD (xem bên dưới)
- Trading ngắn hạn
- Bắt trend thay đổi nhanh hơn MA

#### **NHÓM 5: Volatility - Độ biến động (3 cột)**

```python
volatility_5   # Độ biến động 5 ngày
volatility_10  # Độ biến động 10 ngày
volatility_20  # Độ biến động 20 ngày
```

**Công thức:**
```python
# Bước 1: Tính returns (% thay đổi mỗi ngày)
returns = [+2%, -1%, +3%, -0.5%, +1.5%, ...]

# Bước 2: Tính độ lệch chuẩn của returns
volatility = std(returns) × 100

# Ví dụ:
# Nếu returns dao động [+2%, +1.8%, +2.2%, +1.9%, +2.1%]
# → std nhỏ → volatility thấp → ổn định

# Nếu returns dao động [+5%, -3%, +7%, -4%, +6%]
# → std lớn → volatility cao → rủi ro
```

**Cách đọc:**
```
volatility_20 = 1.5%:
→ Trong 20 ngày, giá dao động trung bình ±1.5% mỗi ngày
→ RỦI RO THẤP - ổn định

volatility_20 = 5.0%:
→ Trong 20 ngày, giá dao động trung bình ±5% mỗi ngày
→ RỦI RO CAO - biến động mạnh
```

**Tác dụng:**
- Đo rủi ro
- Trading: Volatility cao → tránh hoặc dùng stop-loss chặt
- ML học: "Khi volatility tăng đột ngột, thường có sự kiện lớn"

#### **NHÓM 6: RSI - Chỉ số sức mạnh (1 cột)**

```python
rsi_14  # RSI 14 ngày
```

**Công thức đơn giản:**
```python
# Bước 1: Tính gain và loss 14 ngày
Gain = [+2, +1, 0, 0, +3, ...]  # Những ngày tăng
Loss = [0, 0, -1, -2, 0, ...]   # Những ngày giảm (đổi dấu)

# Bước 2: Tính trung bình
Avg_Gain = mean(Gain) = 1.5
Avg_Loss = mean(Loss) = 1.0

# Bước 3: RS và RSI
RS = Avg_Gain / Avg_Loss = 1.5 / 1.0 = 1.5
RSI = 100 - (100 / (1 + RS)) = 100 - (100 / 2.5) = 60
```

**Cách đọc:**
```
RSI = 0-100 (chỉ số từ 0 đến 100)

┌────────────────────────────────────┐
│  RSI > 70  │ OVERBOUGHT (quá mua)  │ → Có thể sắp giảm
├────────────────────────────────────┤
│  30-70     │ NEUTRAL (trung lập)   │ → Bình thường
├────────────────────────────────────┤
│  RSI < 30  │ OVERSOLD (quá bán)    │ → Có thể sắp tăng
└────────────────────────────────────┘

Ví dụ:
- RSI = 75 → "Quá nhiều người mua, áp lực bán tăng"
- RSI = 25 → "Quá nhiều người bán, áp lực mua tăng"
```

**Tác dụng:**
- Xác định điểm đảo chiều
- Tránh mua khi quá mua (RSI > 70)
- Tìm cơ hội mua khi quá bán (RSI < 30)

#### **NHÓM 7: MACD - Xu hướng động lượng (3 cột)**

```python
macd         # MACD line = EMA(12) - EMA(26)
macd_signal  # Signal line = EMA(9) của MACD
macd_hist    # Histogram = MACD - Signal
```

**Công thức:**
```python
# Bước 1: Tính 2 EMA
ema_12 = 105  # EMA ngắn hạn
ema_26 = 102  # EMA dài hạn

# Bước 2: MACD
macd = ema_12 - ema_26 = 105 - 102 = 3

# Bước 3: Signal (MA của MACD)
macd_signal = EMA(macd, 9) = 2.5

# Bước 4: Histogram
macd_hist = macd - macd_signal = 3 - 2.5 = 0.5
```

**Cách đọc - RẤT QUAN TRỌNG:**
```
┌────────────────────────────────────────────────────────┐
│ Tín hiệu                │ Ý nghĩa                      │
├────────────────────────────────────────────────────────┤
│ MACD cắt LÊN Signal     │ TÍN HIỆU MUA (BUY)          │
│ MACD cắt XUỐNG Signal   │ TÍN HIỆU BÁN (SELL)         │
├────────────────────────────────────────────────────────┤
│ macd_hist > 0 (dương)   │ Bullish (xu hướng tăng)     │
│ macd_hist < 0 (âm)      │ Bearish (xu hướng giảm)     │
├────────────────────────────────────────────────────────┤
│ Histogram tăng dần      │ Momentum tăng tốc           │
│ Histogram giảm dần      │ Momentum chậm lại           │
└────────────────────────────────────────────────────────┘
```

**Ví dụ thực tế:**
```
Ngày:     1      2      3      4      5
macd:    -0.5   -0.2    0.1    0.4    0.6
signal:   0.0    0.0    0.0    0.2    0.4
hist:    -0.5   -0.2    0.1    0.2    0.2

Phân tích:
- Ngày 3: MACD cắt lên Signal (từ âm sang dương)
  → TÍN HIỆU MUA!
- Ngày 4-5: Histogram dương và tăng
  → Xu hướng tăng đang mạnh lên
```

**Tác dụng:**
- Tín hiệu vào/ra chính xác
- Xác định xu hướng và động lượng
- Là chỉ số "vàng" trong technical analysis

#### **NHÓM 8: Bollinger Bands - Biên độ dao động (4 cột)**

```python
bb_middle  # Middle band = MA(20)
bb_upper   # Upper band = Middle + 2×StdDev
bb_lower   # Lower band = Middle - 2×StdDev
bb_width   # Width = Upper - Lower
```

**Công thức:**
```python
# Bước 1: Tính MA và StdDev
middle = MA(close, 20) = 100
std = StdDev(close, 20) = 5

# Bước 2: Tính bands
upper = middle + 2×std = 100 + 2×5 = 110
lower = middle - 2×std = 100 - 2×5 = 90
width = upper - lower = 110 - 90 = 20
```

**Hình dung:**
```
        115 ┬─────── Upper Band (110)
            │    /\
        110 │   /  \     ← Giá chạm Upper
            │  /    \       → Có thể sắp giảm
        105 │ /      \
            │/        \
        100 ├─────────── Middle Band (MA20)
            │\        /
         95 │ \      /
            │  \    /     ← Giá chạm Lower
         90 │   \  /         → Có thể sắp tăng
            │    \/
         85 ┴─────── Lower Band (90)
```

**Cách đọc:**
```
1. Giá chạm Upper Band:
   → Giá đang "cao" → Có thể điều chỉnh giảm
   
2. Giá chạm Lower Band:
   → Giá đang "thấp" → Có thể hồi phục tăng
   
3. Bands thu hẹp (width giảm):
   → Volatility thấp → SẮP CÓ BIẾN ĐỘNG LỚN
   
4. Bands mở rộng (width tăng):
   → Volatility cao → Đang trong xu hướng mạnh
```

**Tác dụng:**
- Xác định vùng giá cao/thấp tương đối
- Dự đoán khi nào có biến động lớn
- Kết hợp với các chỉ số khác để vào/ra lệnh

#### **NHÓM 9: Volume Features - Khối lượng (3 cột)**

```python
volume_ma_20    # Trung bình volume 20 ngày
volume_ratio    # Tỷ lệ volume hôm nay / volume_ma
volume_change   # % thay đổi volume so với hôm qua
```

**Ý nghĩa volume:**
```
Volume = Khối lượng giao dịch
       = Số cổ phiếu được mua/bán trong ngày

Volume cao + giá tăng → Xu hướng tăng MẠNH (có conviction)
Volume cao + giá giảm → Xu hướng giảm MẠNH (bán tháo)
Volume thấp + giá tăng → Tăng YẾU (không bền vững)
```

**Cách đọc:**
```
volume_ratio = volume / volume_ma_20

Ví dụ:
- volume_ratio = 2.5
  → Hôm nay volume gấp 2.5 lần trung bình
  → CÓ SỰ KIỆN LỚN! (tin tức? thao túng?)

- volume_ratio = 0.3
  → Hôm nay volume chỉ 30% trung bình
  → Thị trường thờ ơ, không quan tâm
```

**Tác dụng:**
- Xác nhận độ mạnh của xu hướng
- Phát hiện anomaly (bất thường)
- Tìm điểm breakout (phá vỡ)

#### **NHÓM 10: Momentum - Động lượng (3 cột)**

```python
momentum_5   # Close(t) - Close(t-5)
momentum_10  # Close(t) - Close(t-10)
momentum_20  # Close(t) - Close(t-20)
```

**Khác biệt với Returns:**
```
Returns:  % thay đổi    → (105-100)/100 × 100% = +5%
Momentum: Chênh lệch số → 105-100 = +5 (đơn vị: nghìn đồng)

Returns:  Dùng để so sánh nhiều cổ phiếu
Momentum: Dùng để đo tốc độ thay đổi giá của 1 cổ phiếu
```

**Tác dụng:**
- Đo "tốc độ" tăng/giảm
- Momentum dương và tăng → Đang tăng tốc
- Momentum dương nhưng giảm → Sắp chững lại

#### **NHÓM 11: Price Range - Biên độ giá (10+ cột)**

```python
daily_range      # high - low (biên độ trong ngày)
daily_range_pct  # daily_range / close × 100%
price_range_5    # max(high,5) - min(low,5)
price_range_10   # max(high,10) - min(low,10)
price_range_20   # max(high,20) - min(low,20)
atr_14           # Average True Range (14 ngày)
hl_ratio         # high / low
close_position   # Vị trí close trong khoảng [low, high]
```

**daily_range - Biên độ trong ngày:**
```
Ví dụ:
open = 100, high = 108, low = 98, close = 105

daily_range = high - low = 108 - 98 = 10
daily_range_pct = 10 / 105 × 100% = 9.5%

Cách đọc:
- daily_range_pct cao (>5%) → Ngày biến động mạnh
- daily_range_pct thấp (<2%) → Ngày ổn định
```

**atr_14 - Average True Range:**
```
ATR đo "biên độ dao động trung bình"

Công thức:
True Range = max(high-low, |high-prev_close|, |low-prev_close|)
ATR = Trung bình True Range 14 ngày

Tác dụng:
- Đặt stop-loss: "Stop = Close - 2×ATR"
- Đo volatility (giống volatility nhưng dùng range thay vì returns)
```

**close_position - Vị trí đóng cửa:**
```
Công thức:
close_position = (close - low) / (high - low)

Giá trị từ 0 đến 1:
- 0.0: Close = Low (đóng cửa ở đáy) → YẾU
- 0.5: Close ở giữa → TRUNG LẬP
- 1.0: Close = High (đóng cửa ở đỉnh) → MẠNH

Ví dụ:
low=98, high=108, close=105
close_position = (105-98)/(108-98) = 7/10 = 0.7
→ Đóng cửa ở 70% range → Khá mạnh
```

### 2.4. Tổng kết: Tại sao features quan trọng?

**So sánh:**

| Tiêu chí | Raw Data | Features Data |
|----------|----------|---------------|
| **Số cột** | 7 | 45+ |
| **ML hiểu** | ❌ Chỉ là con số | ✅ Hiểu ngữ cảnh |
| **Xu hướng** | ❌ Không biết | ✅ MA, EMA, MACD |
| **Rủi ro** | ❌ Không biết | ✅ Volatility, ATR |
| **Động lượng** | ❌ Không biết | ✅ RSI, Momentum |
| **Tín hiệu** | ❌ Không có | ✅ MACD cross, BB touch |
| **Dự báo** | ❌ Rất kém | ✅ Chính xác hơn nhiều |

**Kết luận:**
```
data/features/ là "trái tim" của hệ thống!

Không có features:
→ ML chỉ nhìn thấy con số khô khan
→ Dự báo kém, không hiểu ngữ cảnh

Có features:
→ ML hiểu xu hướng, rủi ro, động lượng
→ Dự báo tốt hơn nhiều
→ Có thể phát hiện patterns phức tạp
```

---

## PHẦN 3: BÀI TẬP THỰC HÀNH

### Bài tập 1: Crawl 10 năm cho 1 mã
```python
# Viết code crawl FPT từ 2015-2024
# In ra:
# 1. Số dòng dữ liệu
# 2. Ngày đầu tiên và cuối cùng
# 3. Tổng volume giao dịch
```

### Bài tập 2: Phân tích features
```python
# Mở file data/features/vn30/FPT.csv
# Tìm ngày có:
# 1. RSI cao nhất (ngày nào thị trường quá mua?)
# 2. Volatility cao nhất (ngày nào biến động mạnh nhất?)
# 3. volume_ratio cao nhất (ngày nào có sự kiện bất thường?)
```

### Bài tập 3: Tín hiệu MACD
```python
# Từ features data, tìm các ngày có:
# 1. MACD cắt lên Signal (tín hiệu mua)
# 2. MACD cắt xuống Signal (tín hiệu bán)
# Hint: macd_hist đổi dấu từ âm sang dương = cắt lên
```

---

## KẾT LUẬN

1. **Crawl 10 năm:**
   - Thay `start_date='01/01/2015'`
   - Tăng `page_size=3000`, `timeout=60`
   - Dữ liệu nhiều hơn → ML học tốt hơn

2. **Folder features:**
   - Chứa dữ liệu "đã dịch" cho ML
   - 45+ cột features thay vì 7 cột raw
   - Mỗi feature có ý nghĩa cụ thể trong finance

3. **Tầm quan trọng:**
   - Features = Ngôn ngữ ML hiểu
   - Không có features → Không thể dự báo tốt
   - Features tốt → Model tốt

**Happy Learning! 🚀**
