# 📚 HƯỚNG DẪN HỌC HỆ THỐNG TECHPULSE TỪ GỐC
## Dành cho người mới bắt đầu - Mentor Style

---

# 🎯 PHẦN 1: BÀI TOÁN LÀ GÌ? TẠI SAO CẦN LÀM?

## 1.1. Vấn đề thực tế

**Tình huống đời thường:**
Bạn muốn đầu tư cổ phiếu nhưng gặp những vấn đề sau:
- Dữ liệu giá cổ phiếu nằm rải rác trên các website
- Dữ liệu thô (raw) thường có lỗi: trùng lặp, thiếu ngày, giá sai
- Muốn biết xu hướng nhưng không biết tính toán chỉ số kỹ thuật
- Không biết khi nào có "bất thường" (cổ phiếu tăng/giảm đột ngột)

**Bài toán của dự án:**
1. **Multi-step forecasting**: Dự báo cổ phiếu sẽ tăng/giảm bao nhiêu trong 1-5-20 ngày tới
2. **Anomaly detection**: Phát hiện "bất thường" - những cú tăng/giảm bất ngờ
3. **Event-driven explanation**: Giải thích "tại sao" cổ phiếu biến động (tin tức? báo cáo tài chính?)

## 1.2. Pipeline là gì?

**Pipeline = Dây chuyền sản xuất dữ liệu**

Giống như nhà máy sản xuất:
```
Nguyên liệu thô → Làm sạch → Gia công → Sản phẩm hoàn chỉnh
     ↓               ↓           ↓              ↓
Dữ liệu từ web → Loại bỏ lỗi → Tính chỉ số → Dữ liệu sẵn sàng cho AI
```

Trong dự án này:
```
[CRAWL] → [CLEAN] → [FEATURES] → [PHÂN TÍCH/ML]
   ↓          ↓           ↓
 Raw data   Clean data  Features data
```

## 1.3. Tại sao cần làm từng bước?

| Bước | Lý do |
|------|-------|
| **Crawl** | Dữ liệu không tự nhiên có - phải lấy từ nguồn |
| **Clean** | Dữ liệu thô luôn có lỗi - không thể dùng trực tiếp |
| **Features** | ML cần "đặc trưng" - không thể học từ giá thô |

---

# 🧩 PHẦN 2: CẤU TRÚC HỆ THỐNG

## 2.1. Sơ đồ tổng thể

```
TECH STOCKS/
├── src/                          # 🧠 Mã nguồn (Source code)
│   ├── crawl/                    # Module lấy dữ liệu
│   │   └── cafef_scraper.py      # Lấy data từ CafeF
│   ├── clean/                    # Module làm sạch
│   │   └── clean_price.py        # Xử lý dữ liệu giá
│   ├── features/                 # Module tính đặc trưng
│   │   └── build_features.py     # Tính chỉ số kỹ thuật
│   └── pipeline/                 # Điều phối toàn bộ
│       └── vnindex30/            
│           └── fetch_vn30.py     # Chạy pipeline VN30
│
├── data/                         # 📊 Dữ liệu
│   ├── raw/vn30/                 # Dữ liệu thô
│   ├── clean/vn30/               # Dữ liệu đã làm sạch
│   └── features/vn30/            # Dữ liệu có đặc trưng
│
└── venv/                         # 🐍 Môi trường Python
```

## 2.2. Luồng dữ liệu chi tiết

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERNET                                  │
│                    (CafeF API Server)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP Request
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 1: CRAWL (cafef_scraper.py)                             │
│  ─────────────────────────────────────                          │
│  Input:  symbol='FPT', start='01/01/2024', end='31/12/2024'     │
│  Output: DataFrame với cột: date, open, high, low, close, vol   │
│  Lưu:    data/raw/vn30/FPT.csv                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ DataFrame
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 2: CLEAN (clean_price.py)                               │
│  ────────────────────────────────                               │
│  Input:  data/raw/vn30/FPT.csv (có thể có lỗi)                  │
│  Xử lý:  - Loại bỏ dòng trùng                                   │
│          - Loại bỏ giá trị null                                 │
│          - Kiểm tra giá âm, giá = 0                             │
│          - Kiểm tra logic OHLC (High >= Low)                    │
│  Output: data/clean/vn30/FPT.csv (sạch)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Clean DataFrame
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 3: FEATURES (build_features.py)                         │
│  ──────────────────────────────────────                         │
│  Input:  data/clean/vn30/FPT.csv (7 cột)                        │
│  Tính:   - Returns (lợi nhuận 1d, 5d, 10d, 20d)                 │
│          - MA (trung bình 5, 10, 20, 50 ngày)                   │
│          - RSI (chỉ số sức mạnh tương đối)                      │
│          - MACD (xu hướng)                                      │
│          - Bollinger Bands (biên độ dao động)                   │
│          - Volatility (độ biến động)                            │
│  Output: data/features/vn30/FPT.csv (45+ cột)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

# 📦 PHẦN 3: MODULE 1 - CRAWL (Lấy dữ liệu)

## 3.1. Kiến thức Python cần biết trước

### 3.1.1. HTTP Request là gì?

**Giải thích đời thường:**
- Khi bạn vào website, trình duyệt gửi "yêu cầu" (request) đến máy chủ
- Máy chủ trả về "phản hồi" (response) - thường là HTML/JSON
- Code Python có thể làm điều tương tự bằng thư viện `requests`

```python
# Ví dụ đơn giản: Lấy dữ liệu từ API
import requests  # Thư viện gửi HTTP request

# Gửi request GET đến URL
response = requests.get("https://example.com/api/data")

# response.text = nội dung trả về (dạng text)
# response.json() = nội dung trả về (dạng dictionary nếu là JSON)
# response.status_code = mã trạng thái (200 = OK, 404 = Not Found)
```

### 3.1.2. Tại sao cần timeout?

```python
# ❌ KHÔNG CÓ TIMEOUT - Nguy hiểm!
response = requests.get(url)  # Có thể đợi vĩnh viễn nếu server không trả lời

# ✅ CÓ TIMEOUT - An toàn
response = requests.get(url, timeout=30)  # Tối đa đợi 30 giây
```

### 3.1.3. JSON là gì?

**JSON = JavaScript Object Notation**
- Định dạng trao đổi dữ liệu phổ biến
- Trông giống dictionary trong Python

```python
# JSON response từ server (dạng text)
json_text = '{"name": "FPT", "price": 100000}'

# Chuyển thành Python dictionary
import json
data = json.loads(json_text)  # {'name': 'FPT', 'price': 100000}

# Hoặc dùng response.json() trực tiếp
data = response.json()
```

### 3.1.4. pandas DataFrame là gì?

```python
import pandas as pd

# DataFrame = Bảng dữ liệu (như Excel)
#
#    date        open    high    low     close   volume
# 0  2024-01-01  100     105     98      103     1000000
# 1  2024-01-02  103     108     101     106     1200000

# Tạo DataFrame từ list dictionary
records = [
    {'date': '2024-01-01', 'open': 100, 'close': 103},
    {'date': '2024-01-02', 'open': 103, 'close': 106}
]
df = pd.DataFrame(records)
```

## 3.2. Phân tích code cafef_scraper.py

### File: `src/crawl/cafef_scraper.py`

```python
import requests          # Gửi HTTP request đến API
import pandas as pd      # Xử lý dữ liệu dạng bảng
from typing import Optional  # Type hints (gợi ý kiểu dữ liệu)
import logging           # Ghi log (nhật ký) cho debug

# Thiết lập logging - giúp theo dõi code đang làm gì
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

**Tại sao cần logging?**
- Khi code chạy, bạn không thấy gì xảy ra bên trong
- Logging = in ra thông tin để biết code đang làm gì
- `__name__` = tên module hiện tại (để biết log từ file nào)

### Hàm chính: fetch_price_cafef()

```python
def fetch_price_cafef(
    symbol: str,           # Mã cổ phiếu (vd: 'FPT')
    start_date: str,       # Ngày bắt đầu (vd: '01/01/2024')
    end_date: str,         # Ngày kết thúc
    page_size: int = 1000, # Số bản ghi tối đa (mặc định 1000)
    timeout: int = 30      # Thời gian chờ tối đa
) -> pd.DataFrame:         # Trả về DataFrame
```

**Giải thích type hints:**
- `symbol: str` = tham số symbol phải là chuỗi
- `-> pd.DataFrame` = hàm sẽ trả về DataFrame
- Không bắt buộc, nhưng giúp đọc code dễ hơn

### Xây dựng URL API

```python
url = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
params = {
    "Symbol": symbol.upper(),    # FPT
    "StartDate": start_date,     # 01/01/2024
    "EndDate": end_date,         # 31/12/2024
    "PageIndex": 1,              # Trang 1
    "PageSize": page_size        # 1000 bản ghi
}
```

**Tại sao có params?**
- API cần biết bạn muốn dữ liệu gì
- params sẽ được thêm vào URL: `?Symbol=FPT&StartDate=01/01/2024&...`

### Gửi request và xử lý response

```python
try:
    # Gửi GET request
    response = requests.get(url, params=params, timeout=timeout)
    
    # Kiểm tra HTTP status (200 = OK)
    response.raise_for_status()  # Nếu lỗi (4xx, 5xx) sẽ raise exception
    
    # Chuyển response thành dictionary
    data = response.json()
    
except requests.Timeout:
    raise requests.RequestException(f"Timeout sau {timeout} giây")
except requests.RequestException as e:
    raise requests.RequestException(f"Lỗi mạng: {e}")
```

**Tại sao cần try/except?**
- Mạng có thể lỗi bất cứ lúc nào
- Server có thể không phản hồi
- Nếu không xử lý exception, chương trình sẽ crash

### Xác thực dữ liệu trả về

```python
# Kiểm tra cấu trúc response
if not isinstance(data, dict):
    raise ValueError("Response không phải dictionary")

if "Data" not in data or not isinstance(data["Data"], dict):
    raise ValueError("Thiếu trường 'Data'")

if "Data" not in data["Data"]:
    raise ValueError("Thiếu trường 'Data' lồng nhau")

records = data["Data"]["Data"]  # Lấy danh sách bản ghi
```

**Tại sao phải kiểm tra nhiều lần?**
- API có thể thay đổi cấu trúc
- Server có thể trả về lỗi thay vì dữ liệu
- Code phải "phòng thủ" trước mọi tình huống

### Chuyển đổi dữ liệu

```python
df = pd.DataFrame(records)

# Đổi tên cột từ tiếng Việt sang tiếng Anh
column_mapping = {
    "Ngay": "date",           # Ngày
    "GiaMoCua": "open",       # Giá mở cửa
    "GiaCaoNhat": "high",     # Giá cao nhất
    "GiaThapNhat": "low",     # Giá thấp nhất
    "GiaDongCua": "close",    # Giá đóng cửa
    "KhoiLuongKhopLenh": "volume"  # Khối lượng giao dịch
}
df = df.rename(columns=column_mapping)

# Chuyển cột date thành kiểu datetime
df["date"] = pd.to_datetime(df["date"])

# Chuyển các cột số
numeric_cols = ["open", "high", "low", "close", "volume"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```

**errors="coerce" nghĩa là gì?**
- Nếu giá trị không thể chuyển thành số, thay bằng NaN (Not a Number)
- Không làm chương trình crash

## 3.3. Bài tập Module 1

### Bài tập 1.1: Hiểu requests
```python
# Viết code gửi request đến một API công khai và in response
# Gợi ý: Dùng https://jsonplaceholder.typicode.com/todos/1
```

### Bài tập 1.2: Xử lý exception
```python
# Sửa code này để không crash khi URL sai
url = "https://khong-ton-tai.com/api"
response = requests.get(url)
print(response.text)
```

### Bài tập 1.3: Làm việc với DataFrame
```python
# Tạo DataFrame với 5 dòng dữ liệu giá cổ phiếu
# Tính giá trung bình (mean) của cột close
```

---

# 🧹 PHẦN 4: MODULE 2 - CLEAN (Làm sạch dữ liệu)

## 4.1. Tại sao phải làm sạch dữ liệu?

### Các vấn đề thường gặp:

| Vấn đề | Ví dụ | Hậu quả nếu không xử lý |
|--------|-------|------------------------|
| **Dữ liệu trùng** | Cùng ngày xuất hiện 2 lần | ML học sai trọng số |
| **Giá trị null** | Ngày 15/3 không có giá | Tính toán bị lỗi |
| **Giá âm** | close = -100 | Vô nghĩa về tài chính |
| **Giá = 0** | volume = 0 | Có thể là lỗi dữ liệu |
| **High < Low** | high=90, low=100 | Vi phạm logic OHLC |

### Logic OHLC là gì?

**OHLC = Open, High, Low, Close**

```
     HIGH (Giá cao nhất trong ngày)
      │
      │     ┌─────┐ 
      │     │     │ ← CLOSE (Giá đóng cửa)
      │     │     │
OPEN ─│─────┤     │ ← Nến Nhật
      │     │     │
      │     └─────┘
      │
     LOW (Giá thấp nhất trong ngày)

Quy tắc bắt buộc:
- HIGH >= tất cả giá khác (open, close, low)
- LOW  <= tất cả giá khác (open, close, high)
```

## 4.2. Kiến thức Python cần biết

### 4.2.1. pathlib - Xử lý đường dẫn file

```python
from pathlib import Path

# Cách cũ (khó đọc, phụ thuộc OS)
path = "data" + "/" + "raw" + "/" + "FPT.csv"  # Linux
path = "data" + "\\" + "raw" + "\\" + "FPT.csv"  # Windows

# Cách mới với pathlib (đẹp, cross-platform)
path = Path("data") / "raw" / "FPT.csv"

# Các phương thức hữu ích
path.exists()      # True nếu file/thư mục tồn tại
path.is_file()     # True nếu là file
path.is_dir()      # True nếu là thư mục
path.name          # "FPT.csv"
path.parent        # Path("data/raw")
path.mkdir(parents=True, exist_ok=True)  # Tạo thư mục
```

### 4.2.2. Làm việc với DataFrame

```python
import pandas as pd

# Đọc CSV
df = pd.read_csv("data.csv")

# Kiểm tra null
df.isnull()           # DataFrame boolean
df.isnull().sum()     # Số null mỗi cột
df.isnull().any()     # True/False mỗi cột có null không

# Xóa null
df.dropna()           # Xóa dòng có bất kỳ null nào
df.dropna(subset=['close'])  # Chỉ xóa nếu cột 'close' null

# Kiểm tra trùng lặp
df.duplicated()       # Boolean Series
df.duplicated().sum() # Số dòng trùng

# Xóa trùng
df.drop_duplicates()  # Giữ dòng đầu tiên

# Sắp xếp
df.sort_values('date')  # Sắp xếp theo ngày

# Reset index
df.reset_index(drop=True)  # Đánh lại số thứ tự 0, 1, 2, ...
```

### 4.2.3. Boolean indexing

```python
# Lọc dữ liệu theo điều kiện
df[df['close'] > 100]     # Lấy dòng có close > 100
df[df['volume'] < 0]      # Lấy dòng có volume âm

# Đếm số dòng thỏa điều kiện
(df['close'] > 100).sum()   # Số dòng có close > 100
(df['volume'] < 0).any()    # True nếu có ít nhất 1 dòng volume < 0

# Kết hợp điều kiện
# & = AND, | = OR, ~ = NOT
df[(df['high'] < df['low']) | (df['close'] < 0)]
```

## 4.3. Phân tích code clean_price.py

### Hàm chính: clean_price()

```python
def clean_price(
    input_path: str,                      # Đường dẫn file input
    output_path: Optional[str] = None,    # Đường dẫn output (có thể None)
    expected_columns: Optional[List[str]] = None,  # Danh sách cột mong đợi
    remove_duplicates: bool = True,       # Có xóa trùng không
    remove_nulls: bool = True,            # Có xóa null không
    validate: bool = True                 # Có kiểm tra chất lượng không
) -> pd.DataFrame:
```

**Optional[str] nghĩa là gì?**
- Tham số có thể là str hoặc None
- `Optional[str] = None` = mặc định là None

### Quy trình làm sạch

```python
# 1. Kiểm tra file tồn tại
input_file = Path(input_path)
if not input_file.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

# 2. Đọc file
df = pd.read_csv(input_path)
initial_rows = len(df)  # Ghi nhớ số dòng ban đầu

# 3. Đổi tên cột
expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
if len(df.columns) == len(expected_columns):
    df.columns = expected_columns

# 4. Chuyển đổi kiểu dữ liệu
df['date'] = pd.to_datetime(df['date'], errors='coerce')
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 5. Xóa trùng lặp
if remove_duplicates:
    df = df.drop_duplicates()

# 6. Xóa null
if remove_nulls:
    df = df.dropna()

# 7. Sắp xếp theo ngày
df = df.sort_values('date').reset_index(drop=True)

# 8. Validate (kiểm tra chất lượng)
if validate:
    issues = validate_price_data(df)
    for issue in issues:
        logger.warning(f"Cảnh báo: {issue}")
```

### Hàm validate: validate_price_data()

```python
def validate_price_data(df: pd.DataFrame) -> List[str]:
    issues = []  # Danh sách vấn đề tìm thấy
    
    # 1. Kiểm tra giá âm
    for col in ['open', 'high', 'low', 'close']:
        if (df[col] < 0).any():
            count = (df[col] < 0).sum()
            issues.append(f"Có {count} giá trị âm trong cột {col}")
    
    # 2. Kiểm tra volume âm
    if (df['volume'] < 0).any():
        issues.append("Có volume âm")
    
    # 3. Kiểm tra logic OHLC
    # High phải >= tất cả
    high_issues = ((df['high'] < df['open']) | 
                   (df['high'] < df['close']) | 
                   (df['high'] < df['low'])).sum()
    if high_issues > 0:
        issues.append(f"{high_issues} dòng high không phải cao nhất")
    
    # 4. Kiểm tra khoảng trống ngày
    df_sorted = df.sort_values('date')
    date_diff = df_sorted['date'].diff()  # Chênh lệch giữa các ngày
    max_gap = date_diff.max()
    if pd.notna(max_gap) and max_gap.days > 30:
        issues.append(f"Khoảng trống lớn nhất: {max_gap.days} ngày")
    
    return issues
```

## 4.4. Bài tập Module 2

### Bài tập 2.1: Tìm dữ liệu lỗi
```python
# Cho DataFrame sau, tìm các dòng có vấn đề
data = {
    'date': ['2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03'],
    'open': [100, 105, 105, -10],
    'high': [110, 108, 108, 115],
    'low': [95, 102, 102, 90],
    'close': [108, 106, 106, 112]
}
df = pd.DataFrame(data)
# Câu hỏi:
# 1. Có bao nhiêu dòng trùng?
# 2. Dòng nào có giá âm?
# 3. Dòng nào vi phạm logic OHLC?
```

### Bài tập 2.2: Viết hàm kiểm tra
```python
# Viết hàm kiểm tra xem close có nằm trong khoảng [low, high] không
def check_close_in_range(df):
    # Your code here
    pass
```

---

# ⚙️ PHẦN 5: MODULE 3 - FEATURES (Tính đặc trưng)

## 5.1. Tại sao cần features?

### Machine Learning không hiểu "giá thô"

```
Dữ liệu thô:          ML thấy:
date: 2024-01-01      Chỉ là con số 100000
close: 100000         Không biết nó cao hay thấp
                      Không biết đang tăng hay giảm
                      Không biết volatility ra sao
```

**Features = "Dịch" dữ liệu thành ngôn ngữ ML hiểu**

```
Features cho ngày 2024-01-01:
- return_1d: +2%      → "Hôm qua tăng 2%"
- ma_20: 95000        → "Trung bình 20 ngày = 95K"
- rsi_14: 75          → "Đang overbought (quá mua)"
- volatility_10: 3%   → "Biến động 3% trong 10 ngày"
```

## 5.2. Các loại features trong dự án

### 5.2.1. Returns (Lợi nhuận)

**Công thức:**
```
Return(t) = (Price(t) - Price(t-n)) / Price(t-n) × 100%

Ví dụ return_5d:
- Hôm nay (t): close = 105
- 5 ngày trước (t-5): close = 100
- Return = (105 - 100) / 100 × 100% = 5%
```

**Code:**
```python
def calculate_returns(df, periods=[1, 5, 10, 20]):
    for period in periods:
        col_name = f'return_{period}d'
        # pct_change(n) = (x[t] - x[t-n]) / x[t-n]
        df[col_name] = df['close'].pct_change(periods=period) * 100
    return df
```

### 5.2.2. Moving Average (Trung bình động)

**Ý nghĩa:**
- MA_20 = Trung bình giá 20 ngày gần nhất
- Nếu giá > MA_20 → Xu hướng tăng
- Nếu giá < MA_20 → Xu hướng giảm

**Công thức:**
```
MA(20) = (Close[t] + Close[t-1] + ... + Close[t-19]) / 20
```

**Code:**
```python
def calculate_moving_averages(df, windows=[5, 10, 20, 50]):
    for window in windows:
        col_name = f'ma_{window}'
        # rolling(n).mean() = trung bình n phần tử gần nhất
        df[col_name] = df['close'].rolling(window=window).mean()
    return df
```

### 5.2.3. RSI (Relative Strength Index)

**Ý nghĩa:**
- RSI đo "sức mạnh" của xu hướng
- RSI từ 0 đến 100
- RSI > 70: Overbought (quá mua) → Có thể giảm
- RSI < 30: Oversold (quá bán) → Có thể tăng

**Công thức:**
```
1. Tính gain và loss:
   - Nếu giá tăng: gain = chênh lệch, loss = 0
   - Nếu giá giảm: gain = 0, loss = |chênh lệch|

2. Tính Average Gain và Average Loss (14 ngày)

3. RS = Average Gain / Average Loss

4. RSI = 100 - (100 / (1 + RS))
```

**Code:**
```python
def calculate_rsi(df, period=14):
    delta = df['close'].diff()                    # Chênh lệch giá
    
    gain = delta.where(delta > 0, 0)              # Giữ gain, thay loss = 0
    loss = -delta.where(delta < 0, 0)             # Giữ loss (đổi dấu)
    
    avg_gain = gain.rolling(window=period).mean() # TB gain 14 ngày
    avg_loss = loss.rolling(window=period).mean() # TB loss 14 ngày
    
    rs = avg_gain / avg_loss                      # Relative Strength
    rsi = 100 - (100 / (1 + rs))                  # RSI formula
    
    df[f'rsi_{period}'] = rsi
    return df
```

### 5.2.4. MACD (Moving Average Convergence Divergence)

**Ý nghĩa:**
- Đo sự hội tụ/phân kỳ của 2 đường trung bình
- Dùng để xác định xu hướng và điểm vào/ra

**Các thành phần:**
```
MACD Line = EMA(12) - EMA(26)    # Đường MACD
Signal Line = EMA(9) of MACD     # Đường tín hiệu
Histogram = MACD - Signal        # Độ chênh lệch

Cách đọc:
- MACD cắt lên Signal → Mua
- MACD cắt xuống Signal → Bán
- Histogram > 0 → Bullish (tăng)
- Histogram < 0 → Bearish (giảm)
```

**EMA vs MA:**
```
MA (Simple):  Tất cả ngày có trọng số bằng nhau
EMA (Exponential): Ngày gần đây có trọng số lớn hơn

Ví dụ trọng số EMA:
Hôm nay:     ████████████  (cao nhất)
Hôm qua:     ██████████
2 ngày trước: ████████
...
```

**Code:**
```python
def calculate_macd(df, fast=12, slow=26, signal=9):
    # ewm = Exponential Weighted Mean
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow              # MACD line
    macd_signal = macd.ewm(span=signal).mean()  # Signal line
    macd_hist = macd - macd_signal          # Histogram
    
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    return df
```

### 5.2.5. Bollinger Bands

**Ý nghĩa:**
- Biên độ dao động dựa trên độ lệch chuẩn
- Giá "thường" nằm trong dải

**Các thành phần:**
```
Middle Band = MA(20)
Upper Band = MA(20) + 2 × StdDev(20)
Lower Band = MA(20) - 2 × StdDev(20)

      ┌─── Upper Band ───┐
      │                  │
──────┼── Middle Band ───┼──────
      │                  │
      └─── Lower Band ───┘

Cách đọc:
- Giá chạm Upper → Có thể sắp giảm
- Giá chạm Lower → Có thể sắp tăng
- Bands thu hẹp → Sắp có biến động lớn
```

**Code:**
```python
def calculate_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    
    df['bb_middle'] = rolling_mean
    df['bb_upper'] = rolling_mean + (rolling_std * num_std)
    df['bb_lower'] = rolling_mean - (rolling_std * num_std)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    return df
```

### 5.2.6. Volatility (Độ biến động)

**Ý nghĩa:**
- Đo độ "dao động" của giá
- Volatility cao = Rủi ro cao, cơ hội cao
- Volatility thấp = Ổn định

**Công thức:**
```
Volatility = Độ lệch chuẩn của Returns trong N ngày

Ví dụ: volatility_20
1. Tính returns 20 ngày gần nhất
2. Tính độ lệch chuẩn của chúng
```

**Code:**
```python
def calculate_volatility(df, windows=[5, 10, 20]):
    for window in windows:
        returns = df['close'].pct_change()  # Daily returns
        df[f'volatility_{window}'] = returns.rolling(window).std() * 100
    return df
```

### 5.2.7. EMA (Exponential Moving Average) - Chi tiết

**Tại sao cần EMA khi đã có MA?**

```
MA (Simple Moving Average):
- Tất cả ngày có trọng số bằng nhau
- Ví dụ MA_5: (Day1 + Day2 + Day3 + Day4 + Day5) / 5
- Chậm phản ứng với thay đổi giá

EMA (Exponential Moving Average):
- Ngày gần đây có trọng số cao hơn
- Phản ứng nhanh với thay đổi giá
- Dùng trong MACD để bắt tín hiệu nhanh
```

**Công thức EMA:**
```
EMA_today = α × Price_today + (1-α) × EMA_yesterday

Trong đó:
α = 2 / (period + 1)  # Smoothing factor

Ví dụ EMA_12:
α = 2 / (12 + 1) = 0.1538

→ Giá hôm nay chiếm 15.38%
→ EMA hôm qua chiếm 84.62%
```

**So sánh trọng số:**
```
MA_5: Mỗi ngày 20%
Day 1: ████████████████████ (20%)
Day 2: ████████████████████ (20%)
Day 3: ████████████████████ (20%)
Day 4: ████████████████████ (20%)
Day 5: ████████████████████ (20%)

EMA_5: Ngày gần có trọng số cao hơn
Day 1: ██████ (3.9%)
Day 2: ████████ (6.5%)
Day 3: ██████████ (10.8%)
Day 4: ██████████████ (17.9%)
Day 5: ████████████████████████████ (60.9%)
```

**Khi nào dùng MA, khi nào dùng EMA?**

| Chỉ số | Ưu điểm | Nhược điểm | Dùng khi |
|--------|---------|------------|----------|
| **MA** | Ổn định, ít nhiễu | Chậm | Xu hướng dài hạn (ma_50, ma_200) |
| **EMA** | Nhanh, nhạy | Nhiễu nhiều | Xu hướng ngắn hạn, MACD |

**Code:**
```python
def calculate_ema(df, spans=[12, 26]):
    """
    Calculate Exponential Moving Average
    """
    for span in spans:
        col_name = f'ema_{span}'
        df[col_name] = df['close'].ewm(span=span, adjust=False).mean()
    return df

# Trong TechPulse: đã có sẵn trong build_features.py
df = calculate_ema(df, spans=[12, 26])
# → ema_12, ema_26 (dùng cho MACD)
```

**Ví dụ thực tế:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Giả sử FPT có giá tăng đột ngột
prices = [80, 82, 81, 84, 85, 95, 94, 93, 92, 91]  # ↑ tăng mạnh ở ngày 6

df = pd.DataFrame({'close': prices})

# Tính MA và EMA
df['ma_5'] = df['close'].rolling(window=5).mean()
df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()

print(df[['close', 'ma_5', 'ema_5']])

# Kết quả:
#   close   ma_5  ema_5
# 0    80    NaN   80.00
# 1    82    NaN   80.67
# 2    81    NaN   80.78
# 3    84    NaN   81.85
# 4    85   82.4   82.90
# 5    95   85.4   87.27  ← EMA phản ứng nhanh hơn
# 6    94   87.8   89.85
# 7    93   90.4   91.23
# 8    92   93.8   91.49
# 9    91   93.0   91.33

# Nhận xét:
# - Khi giá tăng đột ngột (ngày 6: 85→95)
# - EMA phản ứng nhanh: 82.90 → 87.27 (+4.37)
# - MA phản ứng chậm: 82.4 → 85.4 (+3.0)
```

### 5.2.8. Momentum (Động lực giá)

**Momentum là gì?**
```
Momentum = Tốc độ thay đổi giá
         = Giá hôm nay - Giá N ngày trước

Ý nghĩa:
- Đo "động lực" tăng/giảm của giá
- Momentum > 0: Đang tăng (bullish)
- Momentum < 0: Đang giảm (bearish)
- |Momentum| lớn: Động lực mạnh
```

**Công thức:**
```
Momentum_n = P_today - P_{n days ago}

Ví dụ Momentum_5:
Momentum_5 = Giá hôm nay - Giá 5 ngày trước
```

**Ví dụ đời thường:**
```
Giống như xe hơi:
- Momentum dương lớn: Tăng tốc mạnh (60 → 100 km/h)
- Momentum dương nhỏ: Tăng chậm (60 → 65 km/h)
- Momentum = 0: Giữ nguyên tốc độ
- Momentum âm: Giảm tốc (phanh)
```

**Code:**
```python
def calculate_momentum(df, periods=[5, 10, 20]):
    """
    Calculate price momentum
    """
    for period in periods:
        col_name = f'momentum_{period}'
        df[col_name] = df['close'] - df['close'].shift(period)
    return df

# Trong TechPulse: đã có sẵn trong build_features.py
df = calculate_momentum(df, periods=[5, 10, 20])
# → momentum_5, momentum_10, momentum_20
```

**Ví dụ thực tế:**
```python
# FPT 10 ngày
dates = pd.date_range('2024-01-01', periods=10)
prices = [80, 82, 85, 83, 87, 90, 88, 92, 95, 93]

df = pd.DataFrame({'date': dates, 'close': prices})

# Tính Momentum_5
df['momentum_5'] = df['close'] - df['close'].shift(5)

print(df[['date', 'close', 'momentum_5']])

# Kết quả:
#         date  close  momentum_5
# 0  2024-01-01    80         NaN
# 1  2024-01-02    82         NaN
# 2  2024-01-03    85         NaN
# 3  2024-01-04    83         NaN
# 4  2024-01-05    87         NaN
# 5  2024-01-06    90        10.0  ← 90 - 80 = +10
# 6  2024-01-07    88         6.0  ← 88 - 82 = +6
# 7  2024-01-08    92         7.0  ← 92 - 85 = +7
# 8  2024-01-09    95        12.0  ← 95 - 83 = +12
# 9  2024-01-10    93         6.0  ← 93 - 87 = +6

# Giải thích:
# - Ngày 5-8: Momentum dương → Giá tăng mạnh
# - Ngày 9: Momentum +12 (cao nhất) → Động lực mạnh nhất
# - Ngày 10: Momentum giảm xuống +6 → Động lực yếu đi
```

**Cách đọc Momentum:**
```
Momentum > 0:  Giá cao hơn N ngày trước → Xu hướng tăng
Momentum = 0:  Giá giữ nguyên
Momentum < 0:  Giá thấp hơn N ngày trước → Xu hướng giảm

|Momentum| lớn:   Động lực mạnh (tăng/giảm nhanh)
|Momentum| nhỏ:   Động lực yếu (đi ngang)

Momentum tăng:     Tăng tốc (bullish signal)
Momentum giảm:     Giảm tốc (có thể đảo chiều)
```

### 5.2.9. Simple Return vs Log Return

**Trong TechPulse hiện tại: Dùng Simple Return**

```python
# Code trong build_features.py
def calculate_returns(df, periods=[1, 5, 10, 20]):
    for period in periods:
        col_name = f'return_{period}d'
        df[col_name] = df['close'].pct_change(periods=period) * 100
    return df

# Kết quả: return_1d, return_5d, return_10d, return_20d
```

**Simple Return (Đang dùng):**
```
Simple Return = (P_today - P_yesterday) / P_yesterday × 100%

Ví dụ:
Hôm qua: 80,000
Hôm nay: 84,000

Simple Return = (84,000 - 80,000) / 80,000 × 100
              = 5%
```

**Log Return (Có thể thêm):**
```
Log Return = ln(P_today / P_yesterday)

Ví dụ:
Log Return = ln(84,000 / 80,000)
           = ln(1.05)
           = 0.04879  # ≈ 4.88%
```

**So sánh:**

| Đặc điểm | Simple Return | Log Return |
|----------|---------------|------------|
| **Dễ hiểu** | ✅ "Tăng 5%" | ❌ "0.0488" |
| **Cộng được** | ❌ 5% + 5% ≠ 10% thực tế | ✅ log(AB) = log(A) + log(B) |
| **Symmetric** | ❌ +10% rồi -10% ≠ về giá gốc | ✅ Đối xứng |
| **Dùng trong** | Thực tế, báo cáo | ML/Research, papers |

**Ví dụ tính chất cộng:**
```python
import numpy as np

# FPT 3 ngày
prices = [100, 110, 121]

# Simple Returns
r1 = (110 - 100) / 100  # 10%
r2 = (121 - 110) / 110  # 10%
r_total = r1 + r2       # 20%  ← SAI!

actual = (121 - 100) / 100  # 21%  ← Đúng

# Log Returns
log_r1 = np.log(110/100)   # 0.0953
log_r2 = np.log(121/110)   # 0.0953
log_total = log_r1 + log_r2  # 0.1906
actual_log = np.log(121/100) # 0.1906  ← Đúng!

print(f"Simple: {r_total:.1%} vs {actual:.1%}")  # 20.0% vs 21.0%
print(f"Log: {log_total:.4f} vs {actual_log:.4f}")  # Khớp!
```

**Khi nào dùng gì?**

| Use Case | Dùng | Lý do |
|----------|------|-------|
| **ML Training** | Log Return | Tính chất toán học tốt hơn |
| **Báo cáo** | Simple Return | Dễ hiểu: "Tăng 5%" |
| **Research Paper** | Log Return | Chuẩn academic |
| **Dashboard** | Simple Return | User-friendly |

**Thêm Log Return vào TechPulse (optional):**
```python
def calculate_log_returns(df, periods=[1, 5, 10, 20]):
    """
    Calculate log returns (optional - for ML/research)
    """
    import numpy as np
    
    for period in periods:
        col_name = f'log_return_{period}d'
        df[col_name] = np.log(df['close'] / df['close'].shift(period))
    
    return df

# Nếu muốn dùng:
df = calculate_log_returns(df)
# → log_return_1d, log_return_5d, ...
```

### 5.2.10. Drawdown (Rủi ro thực tế)

**Drawdown là gì?**
```
Drawdown = Mức sụt giảm từ đỉnh cao nhất
         = (Giá hiện tại - Đỉnh cao) / Đỉnh cao × 100%

Ý nghĩa:
"Nếu mua ở đỉnh, đang thua lỗ bao nhiêu %?"
```

**Ví dụ đời thường:**
```
Leo núi:
- Bạn leo lên đỉnh: 3000m (Peak)
- Bây giờ xuống: 2500m (Current)
- Drawdown = (2500 - 3000) / 3000 = -16.7%

→ Từ đỉnh, bạn xuống 16.7%
```

**Maximum Drawdown (MDD):**
```
MDD = Drawdown lớn nhất trong cả khoảng thời gian

Ví dụ FPT:
Jan: 100
Feb: 110  ← Peak
Mar: 95   ← Drawdown = -13.6%
Apr: 100
May: 90   ← Drawdown = -18.2%  ← MDD!

→ Maximum Drawdown = -18.2%
→ "Thua lỗ tối đa 18.2% nếu mua ở đỉnh Feb"
```

**Tại sao Drawdown quan trọng?**
```
Volatility:  Đo biến động (cả lên và xuống)
Drawdown:    Đo rủi ro thua lỗ thực tế (chỉ xuống)

Ví dụ:
Stock A: Biến động ±5% mỗi ngày, không thua lỗ lớn
Stock B: Biến động ±2% mỗi ngày, nhưng có đợt giảm 30%

→ Volatility: A > B
→ Drawdown: B > A (rủi ro thật sự!)
```

**Code:**
```python
def calculate_drawdown(df):
    """
    Calculate drawdown and maximum drawdown
    CHƯA CÓ trong TechPulse - Bạn có thể thêm!
    """
    # Running maximum (đỉnh cao nhất đến thời điểm hiện tại)
    running_max = df['close'].cummax()
    
    # Drawdown từng ngày
    df['drawdown'] = (df['close'] - running_max) / running_max * 100
    
    # Maximum Drawdown
    max_dd = df['drawdown'].min()
    
    return df, max_dd
```

**Ví dụ thực tế:**
```python
# FPT 10 ngày
dates = pd.date_range('2024-01-01', periods=10)
prices = [100, 110, 105, 108, 95, 98, 102, 100, 105, 103]

df = pd.DataFrame({'date': dates, 'close': prices})

# Tính running max
df['running_max'] = df['close'].cummax()

# Tính drawdown
df['drawdown'] = (df['close'] - df['running_max']) / df['running_max'] * 100

print(df[['date', 'close', 'running_max', 'drawdown']])

# Kết quả:
#         date  close  running_max  drawdown
# 0  2024-01-01    100          100      0.00%
# 1  2024-01-02    110          110      0.00%  ← New peak
# 2  2024-01-03    105          110     -4.55%  ← Xuống từ đỉnh
# 3  2024-01-04    108          110     -1.82%
# 4  2024-01-05     95          110    -13.64%  ← MDD!
# 5  2024-01-06     98          110    -10.91%
# 6  2024-01-07    102          110     -7.27%
# 7  2024-01-08    100          110     -9.09%
# 8  2024-01-09    105          110     -4.55%
# 9  2024-01-10    103          110     -6.36%

# Maximum Drawdown = -13.64%
# → Nếu mua ở đỉnh 110, thua lỗ tối đa 13.64%
```

**Cách đọc Drawdown:**
```
Drawdown = 0:       Đang ở đỉnh cao nhất
Drawdown < -10%:    Đang sụt giảm đáng kể
MDD < -20%:         Rủi ro cao (bear market)
MDD < -50%:         Rủi ro rất cao (crash)

Ví dụ thị trường:
- Normal: MDD ~ -10% đến -20%
- Bear market: MDD ~ -20% đến -40%
- COVID crash 2020: MDD ~ -40% đến -50%
```

**Drawdown trong Risk Management:**
```
Khi đầu tư, bạn cần biết:
1. Expected Return: Kỳ vọng lãi bao nhiêu?
2. Volatility: Biến động thế nào?
3. Maximum Drawdown: Thua lỗi tối đa bao nhiêu?

Ví dụ:
Portfolio A: Return +20%, Volatility 10%, MDD -15%
Portfolio B: Return +25%, Volatility 15%, MDD -30%

→ B lãi cao hơn nhưng rủi ro (MDD) cũng cao hơn!
→ Phải cân nhắc risk tolerance
```

## 5.3. Bảng tổng hợp features

| Feature | Ý nghĩa | Cách đọc | Có trong code |
|---------|---------|----------|---------------|
| **Returns** |
| return_1d | Lợi nhuận 1 ngày (Simple) | +2% = Hôm qua tăng 2% | ✅ |
| log_return_1d | Lợi nhuận 1 ngày (Log) | 0.02 ≈ 2% (dùng ML/research) | ❌ (Có thể thêm) |
| **Moving Averages** |
| ma_20 | Trung bình 20 ngày (Simple) | Giá > MA_20 → Tăng | ✅ |
| ema_12 | Trung bình 12 ngày (Exponential) | Phản ứng nhanh, dùng MACD | ✅ |
| **Momentum** |
| momentum_5 | Động lực giá 5 ngày | > 0: Tăng, < 0: Giảm | ✅ |
| **Trend Indicators** |
| rsi_14 | Sức mạnh xu hướng | > 70: Overbought, < 30: Oversold | ✅ |
| macd | Xu hướng (EMA_12 - EMA_26) | > 0: Bullish | ✅ |
| macd_hist | Động lượng (MACD - Signal) | Histogram tăng → Tăng tốc | ✅ |
| **Volatility & Risk** |
| volatility_20 | Độ biến động 20 ngày | Cao = Rủi ro cao | ✅ |
| bb_upper | Bollinger Band trên | Giá chạm → Có thể giảm | ✅ |
| bb_width | Độ rộng Bollinger | Rộng = Volatility cao | ✅ |
| drawdown | Sụt giảm từ đỉnh | -10% = Giảm 10% từ peak | ❌ (Có thể thêm) |
| **Volume** |
| volume_ratio | Volume / TB 20 ngày | > 1.5 = Giao dịch sôi động | ✅ |

## 5.4. Bài tập Module 3

### Bài tập 3.1: Tính MA thủ công
```python
# Cho dữ liệu sau, tính MA_3 (trung bình 3 ngày)
prices = [100, 102, 104, 103, 105, 108, 110]
# Kết quả mong đợi: [NaN, NaN, 102, 103, 104, 105.33, 107.67]
```

### Bài tập 3.2: Hiểu RSI
```python
# Giả sử 14 ngày đều tăng, mỗi ngày +1
# RSI sẽ bằng bao nhiêu? Tại sao?
```

### Bài tập 3.3: Đọc MACD
```python
# Cho:
# macd = -0.5, macd_signal = -0.8, macd_hist = 0.3
# Xu hướng hiện tại là gì? Sắp có tín hiệu gì?
```

### Bài tập 3.4: So sánh MA vs EMA
```python
# Cho giá FPT tăng đột ngột:
prices = [80, 82, 81, 84, 85, 95, 94, 93]

# Tính cả MA_5 và EMA_5
# Câu hỏi:
# 1. Cái nào phản ứng nhanh hơn khi giá tăng đột ngột (ngày 6)?
# 2. Tại sao MACD dùng EMA thay vì MA?
```

### Bài tập 3.5: Tính Momentum
```python
# FPT 8 ngày
prices = [100, 102, 105, 103, 108, 110, 107, 112]

# Tính momentum_5 cho ngày cuối cùng
# Giải thích ý nghĩa con số đó
```

### Bài tập 3.6: Hiểu Drawdown
```python
# FPT 7 ngày
prices = [100, 110, 105, 108, 95, 98, 102]

# Câu hỏi:
# 1. Đỉnh cao nhất (peak) là ngày nào?
# 2. Drawdown lớn nhất (MDD) là bao nhiêu?
# 3. Nếu mua ở đỉnh, thua lỗ tối đa bao nhiêu %?
```

### Bài tập 3.7: Simple vs Log Returns
```python
# Giá tăng 10%, sau đó giảm 10%
# Price: 100 → 110 → 99

# Câu hỏi:
# 1. Tính Simple Returns: r1, r2, r_total
# 2. Tính Log Returns: log_r1, log_r2, log_total
# 3. Cái nào cho kết quả chính xác hơn? Tại sao?
```

---

# 🔗 PHẦN 6: PIPELINE TỔNG HỢP

## 6.1. Tại sao cần Pipeline?

**Không có Pipeline:**
```python
# Phải chạy từng bước thủ công
# Bước 1
df = fetch_price_cafef('FPT', '01/01/2024', '31/12/2024')
df.to_csv('data/raw/FPT.csv')

# Bước 2
df = pd.read_csv('data/raw/FPT.csv')
df_clean = clean_price('data/raw/FPT.csv', 'data/clean/FPT.csv')

# Bước 3
df = pd.read_csv('data/clean/FPT.csv')
df_features = calculate_all_features(df)
df_features.to_csv('data/features/FPT.csv')

# Lặp lại 30 lần cho 30 mã??? 😵
```

**Có Pipeline:**
```python
# Một lệnh, chạy tất cả!
run_vn30_pipeline('01/01/2024', '31/12/2024')
# → Tự động crawl + clean + features cho 30 mã
```

## 6.2. Cấu trúc fetch_vn30.py

```python
# Danh sách 30 mã VN30
VN30_SYMBOLS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR',
    'HDB', 'HPG', 'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB',
    'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC',
    'VJC', 'VNM', 'VPB', 'VRE', 'SSB', 'PDR'
]

def run_vn30_pipeline(start_date, end_date):
    # BƯỚC 1: CRAWL
    logger.info("📥 BƯỚC 1/3: CRAWL DỮ LIỆU VN30")
    raw_results = crawl_many(
        symbols=VN30_SYMBOLS,
        start_date=start_date,
        end_date=end_date,
        save_dir='data/raw/vn30'
    )
    
    # BƯỚC 2: CLEAN
    logger.info("🧹 BƯỚC 2/3: CLEAN DỮ LIỆU")
    clean_results = clean_many(
        raw_dir='data/raw/vn30',
        clean_dir='data/clean/vn30'
    )
    
    # BƯỚC 3: FEATURES
    logger.info("⚙️ BƯỚC 3/3: BUILD FEATURES")
    feature_results = build_features(
        clean_dir='data/clean/vn30',
        features_dir='data/features/vn30'
    )
    
    logger.info("🎉 HOÀN THÀNH!")
```

## 6.3. Kết quả cuối cùng

Sau khi chạy pipeline, bạn có:

```
data/features/vn30/ACB.csv
├── Cột gốc:    date, open, high, low, close, volume, ticker (7 cột)
├── Returns:    return_1d, return_5d, return_10d, return_20d (4 cột)
├── MA:         ma_5, ma_10, ma_20, ma_50 (4 cột)
├── EMA:        ema_12, ema_26 (2 cột)
├── Volatility: volatility_5, volatility_10, volatility_20 (3 cột)
├── RSI:        rsi_14 (1 cột)
├── MACD:       macd, macd_signal, macd_hist (3 cột)
├── Bollinger:  bb_middle, bb_upper, bb_lower, bb_width (4 cột)
├── Volume:     volume_ma_20, volume_ratio, volume_change (3 cột)
├── Momentum:   momentum_5, momentum_10, momentum_20 (3 cột)
└── Range:      daily_range, daily_range_pct, price_range_*, atr_14, ... (10+ cột)

TỔNG: ~45 cột features

📚 CHI TIẾT CÁC FEATURES ĐÃ HỌC:

**✅ Có trong code (build_features.py):**
- Returns: return_1d, return_5d, return_10d, return_20d (Simple Returns)
- MA: ma_5, ma_10, ma_20, ma_50 (Simple Moving Average)
- EMA: ema_12, ema_26 (Exponential Moving Average)
- Volatility: volatility_5, volatility_10, volatility_20
- RSI: rsi_14 (Relative Strength Index)
- MACD: macd, macd_signal, macd_hist
- Bollinger: bb_middle, bb_upper, bb_lower, bb_width
- Volume: volume_ma_20, volume_ratio, volume_change
- Momentum: momentum_5, momentum_10, momentum_20
- Range: daily_range, atr_14, price_range_*, ...

**❌ Chưa có (bạn có thể thêm):**
- Log Returns: log_return_1d, log_return_5d, ... (dùng cho ML/research)
- Drawdown: drawdown, max_drawdown (đo rủi ro thực tế)

**📖 Đã học trong LEARNING_GUIDE này:**
- Section 5.2.1-5.2.6: RSI, MACD, Bollinger, Volatility (cơ bản)
- Section 5.2.7: EMA chi tiết (so sánh MA vs EMA, khi nào dùng gì)
- Section 5.2.8: Momentum (công thức, ý nghĩa, code)
- Section 5.2.9: Simple vs Log Returns (so sánh, khi nào dùng gì)
- Section 5.2.10: Drawdown (MDD, rủi ro thực tế)

**💡 TẤT CẢ NỘI DUNG VỀ FEATURES ĐÃ CÓ TRONG FILE NÀY!**
→ Không cần đọc thêm file nào khác
```

---

# 🎯 PHẦN 7: TÓM TẮT & BƯỚC TIẾP THEO

## 7.1. Tóm tắt hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    TECHPULSE PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   INTERNET (CafeF API)                                      │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                           │
│   │   CRAWL     │  Input: symbol, dates                     │
│   │             │  Output: raw DataFrame                    │
│   └─────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                           │
│   │   CLEAN     │  Input: raw DataFrame                     │
│   │             │  Output: clean DataFrame (no errors)      │
│   └─────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                           │
│   │  FEATURES   │  Input: clean DataFrame                   │
│   │             │  Output: 45+ columns for ML               │
│   └─────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                           │
│   │ PHÂN TÍCH   │  ML, Prediction, Anomaly Detection        │
│   │ & ML        │  (Chưa implement trong dự án này)         │
│   └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 7.2. Kiến thức Python đã học

| Chủ đề | Khái niệm |
|--------|-----------|
| **HTTP** | requests, GET, params, timeout, response, JSON |
| **Exception** | try/except, raise, custom exceptions |
| **pathlib** | Path, exists(), is_file(), mkdir() |
| **pandas** | DataFrame, read_csv, to_csv, columns |
| **pandas** | dropna, drop_duplicates, sort_values, reset_index |
| **pandas** | pct_change, rolling, ewm, diff |
| **pandas** | Boolean indexing, isnull, any, sum |
| **logging** | logger.info, logger.warning, logger.error |
| **typing** | List, Dict, Optional, type hints |

## 7.3. Bước tiếp theo (PROPOSAL - VIETNAM FOCUS)

Theo PROPOSAL (đã điều chỉnh cho thị trường Việt Nam), các bước tiếp theo sẽ là:

1. **Thêm nguồn dữ liệu Việt Nam:**
   - ✅ **CafeF News** (tin tức chứng khoán VN)
   - ✅ **VnExpress** (tin tức kinh tế VN)
   - ⏳ Vietnamese sentiment analysis (PhoBERT)
   - ⏳ Macro data VN (GDP, CPI, lãi suất - nếu có API)

2. **Xây dựng mô hình dự báo:**
   - ⏳ Baseline: ARIMA, GARCH, Linear Regression
   - ⏳ ML: XGBoost, LightGBM, Random Forest
   - ⏳ DL: LSTM, GRU
   - ⏳ Transformer: iTransformer, TimesNet (LTSF)

3. **Phát hiện bất thường:**
   - ⏳ Anomaly Transformer
   - ⏳ TranAD
   - ⏳ Isolation Forest

4. **Vietnamese NLP & Multimodal:**
   - ⏳ Vietnamese text processing (underthesea, pyvi)
   - ⏳ Sentiment analysis (PhoBERT, vn-sentiment)
   - ⏳ Event detection từ tin tức VN
   - ⏳ Multimodal fusion (price + Vietnamese text)
   - ⏳ Cross-modal attention mechanism

5. **Event-Aware Training (PAIN POINT):**
   - ⏳ Detect event days (volume spike, news, volatility)
   - ⏳ Weighted loss function cho event days
   - ⏳ Shock-focused metrics (Tail Loss, CVaR)
   - ⏳ Compare: normal vs event-aware training

6. **Regime Detection:**
   - ⏳ Hidden Markov Model (HMM)
   - ⏳ Detect regime changes trong VN30
   - ⏳ Separate models cho different regimes

7. **Giải thích (Efficient XAI):**
   - ⏳ SHAP (SHapley Additive exPlanations)
   - ⏳ TimeSHAP (time series specific)
   - ⏳ Integrated Gradients
   - ⏳ Efficient approximations (pruning, sampling)
   - TimeSHAP

## 7.4. Bài tập tổng hợp

### Bài tập cuối: Mở rộng pipeline

```python
# 1. Thêm feature mới: Stochastic Oscillator
# Công thức:
# %K = (Close - Low14) / (High14 - Low14) × 100
# %D = SMA(%K, 3)

# 2. Thêm feature: On-Balance Volume (OBV)
# OBV = cumsum(volume * sign(return))

# 3. Chạy pipeline cho 5 mã bất kỳ và in summary
```

---

# 📞 LIÊN HỆ & HỖ TRỢ

Nếu có thắc mắc về bất kỳ phần nào, hãy:
1. Đọc lại phần lý thuyết
2. Chạy code ví dụ từng bước
3. Debug bằng print() để xem giá trị
4. Hỏi mentor!

**Happy Learning! 🚀**
