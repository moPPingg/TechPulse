# TechPulse 📈

A Python-based stock data crawler and analysis pipeline for Vietnamese tech stocks, scraping data from CafeF and providing data cleaning and feature engineering capabilities.

## 🚀 Features

- **Web Scraping**: Automated crawler for CafeF stock price data
- **Data Cleaning**: Comprehensive price data cleaning and validation
- **Feature Engineering**: Build technical indicators and features for analysis
- **Batch Processing**: Crawl multiple stocks with error handling and logging
- **Data Pipeline**: End-to-end pipeline from raw data to processed features

## 📁 Project Structure

```
TECH STOCKS/
├── src/
│   ├── crawl/           # Web scraping modules
│   │   └── cafef_scraper.py
│   ├── clean/           # Data cleaning modules
│   │   └── clean_price.py
│   ├── features/        # Feature engineering
│   │   └── build_features.py
│   └── pipeline/        # Data pipeline orchestration
│       └── run_crawler.py
├── data/
│   ├── raw/            # Raw scraped data
│   ├── processed/      # Cleaned data
│   └── external/       # External data sources
├── venv/               # Virtual environment (not in git)
└── README.md

```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/techpulse.git
cd techpulse
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

- pandas
- numpy
- requests
- python-dateutil
- pytz

## 🎯 Usage

### Scrape Stock Data

```python
from src.pipeline.run_crawler import crawl_many

# Crawl multiple stocks
symbols = ['FPT', 'VNM', 'VCB']
crawl_many(
    symbols=symbols,
    start_date='01/01/2023',
    end_date='31/12/2023',
    save_dir='data/raw'
)
```

### Clean Price Data

```python
from src.clean.clean_price import clean_price_data

# Clean raw price data
df_clean = clean_price_data(
    df_raw,
    ticker='FPT',
    remove_duplicates=True,
    handle_missing=True
)
```

### Build Features

```python
from src.features.build_features import build_features

# Generate technical indicators
df_features = build_features(df_clean, ticker='FPT')
```

## 📚 Documentation

Detailed documentation and learning guides are available in each module:

- `src/clean/LEARNING_GUIDE_clean_price.md` - Data cleaning guide
- `src/features/LEARNING_GUIDE_build_features.md` - Feature engineering guide
- `src/pipeline/đọc hiểu/LEARNING_GUIDE_run_crawler.md` - Crawler pipeline guide

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Always verify data accuracy and comply with website terms of service when scraping.
