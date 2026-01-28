# 📁 HƯỚNG DẪN CẤU TRÚC DỰ ÁN - CHUẨN CHUYÊN NGHIỆP

## MỤC LỤC
1. [Cấu trúc folder hiện tại](#1-cấu-trúc-folder-hiện-tại)
2. [Cấu trúc folder lý tưởng](#2-cấu-trúc-folder-lý-tưởng)
3. [Nguyên tắc tổ chức code](#3-nguyên-tắc-tổ-chức-code)
4. [Hướng dẫn tạo file mới](#4-hướng-dẫn-tạo-file-mới)
5. [Cách import đúng chuẩn](#5-cách-import-đúng-chuẩn)
6. [Best practices](#6-best-practices)

---

## 1. CẤU TRÚC FOLDER HIỆN TẠI

```
W:\TECH STOCKS\
├── data/                           # ✅ TỐT - Tách biệt dữ liệu
│   ├── raw/vn30/                   # Dữ liệu thô
│   ├── clean/vn30/                 # Dữ liệu sạch
│   └── features/vn30/              # Dữ liệu features
│
├── src/                            # ✅ TỐT - Source code chính
│   ├── __init__.py                 # ✅ Package marker
│   ├── crawl/                      # Module crawl
│   │   ├── __init__.py
│   │   └── cafef_scraper.py
│   ├── clean/                      # Module clean
│   │   ├── __init__.py
│   │   └── clean_price.py
│   ├── features/                   # Module features
│   │   ├── __init__.py
│   │   └── build_features.py
│   └── pipeline/                   # Module pipeline
│       ├── __init__.py
│       ├── runcrawler/
│       │   ├── __init__.py
│       │   └── run_crawler.py
│       └── vnindex30/
│           ├── __init__.py
│           └── fetch_vn30.py
│
├── examples/                       # ✅ TỐT - Ví dụ sử dụng
│   └── demo_vn30.py
│
├── venv/                           # ✅ TỐT - Virtual environment
├── requirements.txt                # ✅ TỐT - Dependencies
├── README.md                       # ✅ TỐT - Documentation
│
├── crawl_vn30_10_nam.py           # ⚠️ NÊN CHUYỂN - Script ở root
├── test.py                         # ⚠️ NÊN CHUYỂN - Test ở root
└── LEARNING_GUIDE_*.md            # ✅ OK - Docs ở root
```

### Vấn đề cần cải thiện:

| Vấn đề | Giải pháp |
|--------|-----------|
| Script ở root (`crawl_vn30_10_nam.py`) | Chuyển vào `scripts/` hoặc `examples/` |
| Test ở root (`test.py`) | Chuyển vào `tests/` |
| Thiếu folder `scripts/` | Tạo folder cho các script tiện ích |
| Thiếu folder `tests/` | Tạo folder cho unit tests |
| Thiếu folder `notebooks/` | Tạo folder cho Jupyter notebooks (nếu dùng) |
| Thiếu folder `configs/` | Tạo folder cho config files |

---

## 2. CẤU TRÚC FOLDER LỶ TƯỞNG

### 2.1. Cấu trúc đề xuất (Best Practice)

```
W:\TECH STOCKS\
│
├── 📁 data/                        # DỮ LIỆU (không commit lên Git)
│   ├── raw/                        # Dữ liệu thô từ API
│   │   └── vn30/
│   │       ├── ACB.csv
│   │       ├── FPT.csv
│   │       └── ...
│   ├── clean/                      # Dữ liệu đã làm sạch
│   │   └── vn30/
│   ├── features/                   # Dữ liệu có features
│   │   └── vn30/
│   └── processed/                  # Dữ liệu đã xử lý cho ML
│       └── vn30/
│
├── 📁 src/                         # SOURCE CODE CHÍNH
│   ├── __init__.py                 # Package marker
│   │
│   ├── 📁 crawl/                   # Module lấy dữ liệu
│   │   ├── __init__.py
│   │   ├── cafef_scraper.py        # Scraper CafeF
│   │   ├── sec_scraper.py          # (Future) SEC EDGAR scraper
│   │   └── gdelt_scraper.py        # (Future) GDELT news scraper
│   │
│   ├── 📁 clean/                   # Module làm sạch
│   │   ├── __init__.py
│   │   ├── clean_price.py          # Clean price data
│   │   ├── clean_news.py           # (Future) Clean news data
│   │   └── validators.py           # (Future) Data validators
│   │
│   ├── 📁 features/                # Module tính features
│   │   ├── __init__.py
│   │   ├── build_features.py       # Build technical indicators
│   │   ├── technical.py            # (Future) Advanced technical indicators
│   │   └── sentiment.py            # (Future) Sentiment features
│   │
│   ├── 📁 models/                  # (Future) ML Models
│   │   ├── __init__.py
│   │   ├── forecasting.py          # Forecasting models
│   │   ├── anomaly.py              # Anomaly detection
│   │   └── explainer.py            # XAI (SHAP, etc.)
│   │
│   ├── 📁 pipeline/                # Orchestration pipelines
│   │   ├── __init__.py
│   │   ├── base_pipeline.py        # Base pipeline class
│   │   ├── data_pipeline.py        # Data pipeline (crawl→clean→features)
│   │   └── vnindex30/
│   │       ├── __init__.py
│   │       └── fetch_vn30.py
│   │
│   └── 📁 utils/                   # Utilities
│       ├── __init__.py
│       ├── logger.py               # Logging utilities
│       ├── file_utils.py           # File I/O utilities
│       └── date_utils.py           # Date utilities
│
├── 📁 scripts/                     # SCRIPTS THỰC THI
│   ├── crawl_vn30_10_nam.py       # Script crawl 10 năm
│   ├── update_daily.py             # Script update hàng ngày
│   ├── backfill_data.py            # Script backfill dữ liệu thiếu
│   └── analyze_features.py         # Script phân tích features
│
├── 📁 tests/                       # UNIT TESTS
│   ├── __init__.py
│   ├── test_crawl.py               # Test crawl module
│   ├── test_clean.py               # Test clean module
│   ├── test_features.py            # Test features module
│   └── test_pipeline.py            # Test pipeline
│
├── 📁 notebooks/                   # JUPYTER NOTEBOOKS
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtesting.ipynb
│
├── 📁 configs/                     # CONFIG FILES
│   ├── config.yaml                 # Main config
│   ├── symbols.yaml                # List of symbols
│   └── features.yaml               # Feature configurations
│
├── 📁 docs/                        # DOCUMENTATION
│   ├── LEARNING_GUIDE_FULL_SYSTEM.md
│   ├── HƯỚNG_DẪN_CRAWL_10_NĂM_VÀ_FEATURES.md
│   ├── API_REFERENCE.md
│   └── ARCHITECTURE.md
│
├── 📁 examples/                    # EXAMPLES
│   ├── demo_vn30.py
│   ├── demo_single_stock.py
│   └── demo_ml_pipeline.py
│
├── 📁 venv/                        # Virtual environment (không commit)
│
├── 📄 .gitignore                   # Git ignore file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package setup (optional)
├── 📄 README.md                    # Project README
├── 📄 LICENSE                      # License file
└── 📄 PROPOSAL_Group2_StockTech.docx
```

### 2.2. Giải thích từng folder

| Folder | Mục đích | Ví dụ file |
|--------|----------|------------|
| **`data/`** | Chứa tất cả dữ liệu (không commit lên Git) | `data/raw/vn30/FPT.csv` |
| **`src/`** | Source code chính của dự án | `src/crawl/cafef_scraper.py` |
| **`scripts/`** | Scripts để chạy các tác vụ cụ thể | `scripts/crawl_vn30_10_nam.py` |
| **`tests/`** | Unit tests và integration tests | `tests/test_crawl.py` |
| **`notebooks/`** | Jupyter notebooks cho phân tích | `notebooks/01_eda.ipynb` |
| **`configs/`** | File cấu hình (YAML, JSON) | `configs/config.yaml` |
| **`docs/`** | Documentation và hướng dẫn | `docs/API_REFERENCE.md` |
| **`examples/`** | Code ví dụ sử dụng | `examples/demo_vn30.py` |

---

## 3. NGUYÊN TẮC TỔ CHỨC CODE

### 3.1. Nguyên tắc SOLID cho Python

#### **S - Single Responsibility (Trách nhiệm đơn)**
```python
# ❌ SAI - 1 file làm quá nhiều việc
# src/data_handler.py
def fetch_and_clean_and_build_features(symbol):
    # Crawl
    df = fetch_price(symbol)
    # Clean
    df = clean_data(df)
    # Features
    df = build_features(df)
    return df

# ✅ ĐÚNG - Mỗi module 1 trách nhiệm
# src/crawl/cafef_scraper.py
def fetch_price(symbol):
    ...

# src/clean/clean_price.py
def clean_data(df):
    ...

# src/features/build_features.py
def build_features(df):
    ...
```

#### **D - Dependency Inversion (Phụ thuộc vào abstraction)**
```python
# ✅ ĐÚNG - Dùng abstraction
# src/pipeline/base_pipeline.py
class BasePipeline:
    def run(self):
        self.crawl()
        self.clean()
        self.build_features()
    
    def crawl(self):
        raise NotImplementedError
    
    def clean(self):
        raise NotImplementedError
    
    def build_features(self):
        raise NotImplementedError

# src/pipeline/vnindex30/vn30_pipeline.py
class VN30Pipeline(BasePipeline):
    def crawl(self):
        # Implementation cụ thể
        ...
```

### 3.2. Cấu trúc file Python chuẩn

```python
"""
Module docstring - Mô tả module làm gì

Example:
    >>> from src.crawl import cafef_scraper
    >>> df = cafef_scraper.fetch_price('FPT', '01/01/2024', '31/12/2024')
"""

# 1. IMPORTS - Theo thứ tự
# Standard library
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict

# Third-party
import pandas as pd
import numpy as np
import requests

# Local imports
from src.utils.logger import get_logger
from src.utils.file_utils import save_csv

# 2. CONSTANTS
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# 3. LOGGER
logger = get_logger(__name__)

# 4. CLASSES
class DataFetcher:
    """Class docstring"""
    pass

# 5. FUNCTIONS
def fetch_price(symbol: str) -> pd.DataFrame:
    """Function docstring"""
    pass

# 6. MAIN (nếu là script)
if __name__ == "__main__":
    pass
```

### 3.3. Quy tắc đặt tên

| Loại | Quy tắc | Ví dụ |
|------|---------|-------|
| **File/Module** | `snake_case.py` | `cafef_scraper.py` |
| **Class** | `PascalCase` | `DataFetcher` |
| **Function** | `snake_case()` | `fetch_price()` |
| **Variable** | `snake_case` | `start_date` |
| **Constant** | `UPPER_SNAKE_CASE` | `DEFAULT_TIMEOUT` |
| **Private** | `_leading_underscore` | `_internal_func()` |

---

## 4. HƯỚNG DẪN TẠO FILE MỚI

### 4.1. Tạo script mới trong `scripts/`

**Ví dụ: Tạo script update dữ liệu hàng ngày**

```python
# scripts/update_daily.py
"""
Script để update dữ liệu VN30 hàng ngày
Chạy script này mỗi ngày để cập nhật dữ liệu mới nhất

Usage:
    python scripts/update_daily.py
"""

import sys
from pathlib import Path

# Thêm project root vào Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import từ src
from src.crawl.cafef_scraper import fetch_price_cafef
from src.clean.clean_price import clean_price
from src.features.build_features import build_features_single
from src.utils.logger import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)


def update_daily(symbols: list):
    """
    Update dữ liệu cho danh sách symbols
    
    Args:
        symbols: List các mã cổ phiếu
    """
    # Lấy ngày hôm nay
    today = datetime.now().strftime('%d/%m/%Y')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%d/%m/%Y')
    
    logger.info(f"Updating data for {len(symbols)} symbols")
    
    for symbol in symbols:
        try:
            logger.info(f"Processing {symbol}...")
            
            # Crawl
            df = fetch_price_cafef(symbol, yesterday, today)
            
            # Append vào file cũ
            # ... (logic append)
            
            logger.info(f"✅ {symbol} updated")
            
        except Exception as e:
            logger.error(f"❌ {symbol} failed: {e}")


if __name__ == "__main__":
    VN30_SYMBOLS = ['ACB', 'FPT', 'VCB', ...]  # Load từ config
    update_daily(VN30_SYMBOLS)
```

### 4.2. Tạo module mới trong `src/`

**Ví dụ: Tạo module utils**

```python
# src/utils/logger.py
"""
Logging utilities cho toàn bộ dự án
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Tạo logger với format chuẩn
    
    Args:
        name: Tên logger (thường dùng __name__)
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.setLevel(level)
    
    return logger


def get_file_logger(name: str, log_file: str) -> logging.Logger:
    """
    Tạo logger ghi vào file
    
    Args:
        name: Tên logger
        log_file: Đường dẫn file log
    
    Returns:
        Logger instance
    """
    logger = get_logger(name)
    
    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger
```

```python
# src/utils/file_utils.py
"""
File I/O utilities
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def save_csv(df: pd.DataFrame, path: str, create_dirs: bool = True):
    """
    Lưu DataFrame vào CSV với error handling
    
    Args:
        df: DataFrame cần lưu
        path: Đường dẫn file
        create_dirs: Tự động tạo thư mục nếu chưa có
    """
    file_path = Path(path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(file_path, index=False, encoding='utf-8')


def load_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Đọc CSV với error handling
    
    Args:
        path: Đường dẫn file
        **kwargs: Tham số cho pd.read_csv
    
    Returns:
        DataFrame hoặc None nếu lỗi
    """
    file_path = Path(path)
    
    if not file_path.exists():
        return None
    
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def ensure_dir(path: str):
    """Đảm bảo thư mục tồn tại"""
    Path(path).mkdir(parents=True, exist_ok=True)
```

### 4.3. Tạo config file

```yaml
# configs/config.yaml
project:
  name: "TechPulse"
  version: "1.0.0"

data:
  raw_dir: "data/raw/vn30"
  clean_dir: "data/clean/vn30"
  features_dir: "data/features/vn30"

crawl:
  timeout: 60
  page_size: 3000
  max_retries: 3
  retry_delay: 5

features:
  returns:
    periods: [1, 5, 10, 20]
  
  moving_averages:
    windows: [5, 10, 20, 50, 200]
  
  ema:
    spans: [12, 26]
  
  volatility:
    windows: [5, 10, 20]
  
  rsi:
    period: 14
  
  macd:
    fast: 12
    slow: 26
    signal: 9
  
  bollinger:
    window: 20
    num_std: 2

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/techpulse.log"
```

```yaml
# configs/symbols.yaml
vn30:
  - ACB
  - BCM
  - BID
  - BVH
  - CTG
  - FPT
  - GAS
  - GVR
  - HDB
  - HPG
  - MBB
  - MSN
  - MWG
  - PLX
  - POW
  - SAB
  - SSI
  - STB
  - TCB
  - TPB
  - VCB
  - VHM
  - VIB
  - VIC
  - VJC
  - VNM
  - VPB
  - VRE
  - SSB
  - PDR
```

---

## 5. CÁCH IMPORT ĐÚNG CHUẨN

### 5.1. Import trong src/

```python
# Trong src/pipeline/vnindex30/fetch_vn30.py

# ✅ ĐÚNG - Import tuyệt đối từ project root
from src.crawl.cafef_scraper import fetch_price_cafef
from src.clean.clean_price import clean_many
from src.features.build_features import build_features

# ❌ SAI - Import tương đối phức tạp
from ...crawl.cafef_scraper import fetch_price_cafef
```

### 5.2. Import trong scripts/

```python
# Trong scripts/crawl_vn30_10_nam.py

import sys
from pathlib import Path

# Thêm project root vào sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Bây giờ có thể import từ src
from src.crawl.cafef_scraper import fetch_price_cafef
from src.clean.clean_price import clean_price
```

### 5.3. Import trong tests/

```python
# Trong tests/test_crawl.py

import sys
from pathlib import Path

# Thêm project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import module cần test
from src.crawl.cafef_scraper import fetch_price_cafef

# Import testing libraries
import unittest
import pandas as pd


class TestCafefScraper(unittest.TestCase):
    def test_fetch_price(self):
        df = fetch_price_cafef('FPT', '01/01/2024', '31/01/2024')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
```

---

## 6. BEST PRACTICES

### 6.1. Sử dụng `__init__.py`

```python
# src/crawl/__init__.py
"""
Crawl module - Lấy dữ liệu từ các nguồn
"""

from .cafef_scraper import fetch_price_cafef

__all__ = ['fetch_price_cafef']
```

Lợi ích:
```python
# Thay vì
from src.crawl.cafef_scraper import fetch_price_cafef

# Có thể viết ngắn hơn
from src.crawl import fetch_price_cafef
```

### 6.2. Sử dụng Type Hints

```python
from typing import Optional, List, Dict, Tuple
import pandas as pd

def fetch_price(
    symbol: str,
    start_date: str,
    end_date: str,
    timeout: int = 30
) -> pd.DataFrame:
    """
    Fetch price data
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        timeout: Request timeout
    
    Returns:
        DataFrame with price data
    """
    pass
```

### 6.3. Sử dụng Docstrings

```python
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI)
    
    RSI is a momentum indicator that measures the speed and magnitude
    of price changes. Values range from 0 to 100.
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period (default: 14)
    
    Returns:
        DataFrame with added 'rsi_{period}' column
    
    Raises:
        ValueError: If 'close' column is missing
    
    Example:
        >>> df = pd.DataFrame({'close': [100, 102, 101, 105]})
        >>> df = calculate_rsi(df, period=14)
        >>> print(df['rsi_14'])
    
    References:
        - https://www.investopedia.com/terms/r/rsi.asp
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    
    # Implementation
    ...
    
    return df
```

### 6.4. Error Handling

```python
# ✅ ĐÚNG - Specific exceptions
try:
    df = fetch_price(symbol)
except requests.Timeout:
    logger.error(f"Timeout fetching {symbol}")
except requests.RequestException as e:
    logger.error(f"Network error: {e}")
except ValueError as e:
    logger.error(f"Invalid data: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise

# ❌ SAI - Catch all
try:
    df = fetch_price(symbol)
except:
    print("Error!")
```

### 6.5. Logging thay vì print

```python
# ❌ SAI
print("Fetching data...")
print(f"Got {len(df)} records")

# ✅ ĐÚNG
logger.info("Fetching data...")
logger.info(f"Got {len(df)} records")
logger.debug(f"DataFrame shape: {df.shape}")
logger.warning(f"Missing {null_count} values")
logger.error(f"Failed to fetch {symbol}")
```

---

## 7. TÓM TẮT CHECKLIST

### ✅ Checklist tạo file mới:

- [ ] Đặt tên file theo `snake_case.py`
- [ ] Đặt ở đúng folder (`src/`, `scripts/`, `tests/`)
- [ ] Có docstring ở đầu file
- [ ] Import theo thứ tự: stdlib → third-party → local
- [ ] Có type hints cho functions
- [ ] Có docstrings cho functions/classes
- [ ] Sử dụng logger thay vì print
- [ ] Có error handling
- [ ] Có `if __name__ == "__main__"` nếu là script

### ✅ Checklist tổ chức code:

- [ ] Mỗi module có trách nhiệm rõ ràng
- [ ] Không có code trùng lặp
- [ ] Functions ngắn gọn (<50 lines)
- [ ] Tên biến/function mô tả rõ ràng
- [ ] Có comments cho logic phức tạp
- [ ] Có unit tests
- [ ] Update README.md khi thêm feature mới

---

## 8. KẾT LUẬN

**Nguyên tắc vàng:**
1. **Separation of Concerns**: Mỗi module làm 1 việc
2. **DRY (Don't Repeat Yourself)**: Không lặp code
3. **KISS (Keep It Simple, Stupid)**: Giữ code đơn giản
4. **YAGNI (You Aren't Gonna Need It)**: Chỉ code những gì cần

**Lợi ích cấu trúc tốt:**
- ✅ Dễ đọc, dễ hiểu
- ✅ Dễ maintain (bảo trì)
- ✅ Dễ test
- ✅ Dễ mở rộng
- ✅ Dễ collaborate (làm việc nhóm)

**Happy Coding! 🚀**
