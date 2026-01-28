# -*- coding: utf-8 -*-
"""
Demo script sử dụng cấu trúc mới
Minh họa cách import và sử dụng các modules

Author: TechPulse Team
Date: 2026-01-25
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

# Import utilities
from src.utils.logger import get_logger
from src.utils.file_utils import save_csv, load_csv, ensure_dir
from src.utils.date_utils import format_date, get_n_years_ago, get_trading_days

# Import config
import yaml
from datetime import datetime


# Setup logger
logger = get_logger(__name__)


def load_config():
    """Load configuration từ file YAML"""
    config_path = project_root / 'configs' / 'config.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("✅ Loaded config from configs/config.yaml")
    return config


def load_symbols():
    """Load danh sách symbols từ file YAML"""
    symbols_path = project_root / 'configs' / 'symbols.yaml'
    
    with open(symbols_path, 'r', encoding='utf-8') as f:
        symbols = yaml.safe_load(f)
    
    logger.info("✅ Loaded symbols from configs/symbols.yaml")
    return symbols


def demo_date_utils():
    """Demo sử dụng date utilities"""
    logger.info("\n" + "=" * 70)
    logger.info("📅 DEMO: Date Utilities")
    logger.info("=" * 70)
    
    # Lấy ngày 10 năm trước
    today = datetime.now()
    ten_years_ago = get_n_years_ago(10)
    
    logger.info(f"Hôm nay:        {format_date(today)}")
    logger.info(f"10 năm trước:   {format_date(ten_years_ago)}")
    
    # Tính số ngày giao dịch
    trading_days = get_trading_days(ten_years_ago, today)
    logger.info(f"Số ngày giao dịch ước tính: {trading_days} ngày")
    logger.info(f"Số năm: {trading_days / 250:.1f} năm")


def demo_crawl_single_stock(symbol='FPT', config=None):
    """Demo crawl 1 mã cổ phiếu"""
    logger.info("\n" + "=" * 70)
    logger.info(f"📥 DEMO: Crawl dữ liệu {symbol}")
    logger.info("=" * 70)
    
    try:
        # Lấy config
        if config is None:
            config = load_config()
        
        timeout = config['crawl']['timeout']
        page_size = config['crawl']['page_size']
        raw_dir = config['data']['raw_dir']
        
        # Crawl dữ liệu
        logger.info(f"Đang crawl {symbol}...")
        df = fetch_price_cafef(
            symbol=symbol,
            start_date='01/01/2024',
            end_date='31/12/2024',
            page_size=page_size,
            timeout=timeout
        )
        
        logger.info(f"✅ Lấy được {len(df)} dòng")
        
        # Lưu vào file
        ensure_dir(raw_dir)
        output_path = Path(raw_dir) / f'{symbol}.csv'
        save_csv(df, output_path)
        
        logger.info(f"✅ Đã lưu vào: {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi crawl {symbol}: {e}")
        return None


def demo_clean_data(symbol='FPT', config=None):
    """Demo làm sạch dữ liệu"""
    logger.info("\n" + "=" * 70)
    logger.info(f"🧹 DEMO: Clean dữ liệu {symbol}")
    logger.info("=" * 70)
    
    try:
        if config is None:
            config = load_config()
        
        raw_dir = config['data']['raw_dir']
        clean_dir = config['data']['clean_dir']
        
        input_path = Path(raw_dir) / f'{symbol}.csv'
        output_path = Path(clean_dir) / f'{symbol}.csv'
        
        # Check file tồn tại
        if not input_path.exists():
            logger.warning(f"⚠️  File raw chưa có: {input_path}")
            logger.info("💡 Chạy demo_crawl_single_stock() trước")
            return None
        
        # Clean
        logger.info(f"Đang clean {symbol}...")
        df_clean = clean_price(
            input_path=str(input_path),
            output_path=str(output_path),
            remove_duplicates=config['clean']['remove_duplicates'],
            remove_nulls=config['clean']['remove_nulls'],
            validate=config['clean']['validate']
        )
        
        logger.info(f"✅ Clean hoàn tất: {len(df_clean)} dòng")
        logger.info(f"✅ Đã lưu vào: {output_path}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi clean {symbol}: {e}")
        return None


def demo_build_features(symbol='FPT', config=None):
    """Demo tính features"""
    logger.info("\n" + "=" * 70)
    logger.info(f"⚙️  DEMO: Build features {symbol}")
    logger.info("=" * 70)
    
    try:
        if config is None:
            config = load_config()
        
        clean_dir = config['data']['clean_dir']
        features_dir = config['data']['features_dir']
        
        # Check file tồn tại
        input_path = Path(clean_dir) / f'{symbol}.csv'
        if not input_path.exists():
            logger.warning(f"⚠️  File clean chưa có: {input_path}")
            logger.info("💡 Chạy demo_clean_data() trước")
            return None
        
        # Build features
        logger.info(f"Đang tính features cho {symbol}...")
        df_features = build_features_single(
            filename=f'{symbol}.csv',
            clean_dir=clean_dir,
            features_dir=features_dir,
            drop_na=True,
            save_file=True
        )
        
        if df_features is not None:
            logger.info(f"✅ Features hoàn tất:")
            logger.info(f"   - Số dòng: {len(df_features)}")
            logger.info(f"   - Số cột: {len(df_features.columns)}")
            logger.info(f"   - Đã lưu vào: {features_dir}/{symbol}.csv")
            
            # Hiển thị một số features
            logger.info(f"\n📊 Sample features (5 dòng cuối):")
            logger.info(f"\n{df_features[['date', 'close', 'return_1d', 'ma_20', 'rsi_14', 'macd_hist']].tail()}")
            
            return df_features
        else:
            logger.error("❌ Lỗi khi tính features")
            return None
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi build features {symbol}: {e}")
        return None


def demo_full_pipeline(symbol='FPT'):
    """Demo chạy full pipeline cho 1 mã"""
    logger.info("\n" + "=" * 80)
    logger.info(f"🚀 DEMO: Full Pipeline cho {symbol}")
    logger.info("=" * 80)
    
    # Load config
    config = load_config()
    
    # Step 1: Crawl
    df_raw = demo_crawl_single_stock(symbol, config)
    if df_raw is None:
        return
    
    # Step 2: Clean
    df_clean = demo_clean_data(symbol, config)
    if df_clean is None:
        return
    
    # Step 3: Features
    df_features = demo_build_features(symbol, config)
    if df_features is None:
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 HOÀN THÀNH FULL PIPELINE!")
    logger.info("=" * 80)
    logger.info(f"✅ Raw:      {len(df_raw)} dòng")
    logger.info(f"✅ Clean:    {len(df_clean)} dòng")
    logger.info(f"✅ Features: {len(df_features)} dòng, {len(df_features.columns)} cột")


def demo_load_config_and_symbols():
    """Demo load config và symbols"""
    logger.info("\n" + "=" * 70)
    logger.info("⚙️  DEMO: Load Config & Symbols")
    logger.info("=" * 70)
    
    # Load config
    config = load_config()
    logger.info(f"\n📋 Config:")
    logger.info(f"   Project: {config['project']['name']} v{config['project']['version']}")
    logger.info(f"   Timeout: {config['crawl']['timeout']}s")
    logger.info(f"   Page size: {config['crawl']['page_size']}")
    logger.info(f"   Raw dir: {config['data']['raw_dir']}")
    
    # Load symbols
    symbols = load_symbols()
    logger.info(f"\n📊 Symbols:")
    logger.info(f"   VN30: {len(symbols['vn30'])} mã")
    logger.info(f"   Banks: {len(symbols['banks'])} mã")
    logger.info(f"   Tech: {len(symbols['tech'])} mã")
    logger.info(f"\n   VN30 list: {', '.join(symbols['vn30'][:10])}...")


def main():
    """Main function"""
    logger.info("\n" + "=" * 80)
    logger.info("🎓 DEMO: CẤU TRÚC DỰ ÁN MỚI")
    logger.info("=" * 80)
    logger.info("Script này minh họa cách sử dụng cấu trúc mới")
    logger.info("")
    
    # Menu
    print("\nChọn demo:")
    print("  [1] Demo Date Utilities")
    print("  [2] Demo Load Config & Symbols")
    print("  [3] Demo Crawl 1 mã")
    print("  [4] Demo Clean 1 mã")
    print("  [5] Demo Build Features 1 mã")
    print("  [6] Demo Full Pipeline (Crawl → Clean → Features)")
    print("  [0] Thoát")
    print("")
    
    try:
        choice = input("Nhập lựa chọn [0-6]: ").strip()
        
        if choice == '1':
            demo_date_utils()
        elif choice == '2':
            demo_load_config_and_symbols()
        elif choice == '3':
            symbol = input("Nhập mã cổ phiếu (mặc định FPT): ").strip().upper() or 'FPT'
            demo_crawl_single_stock(symbol)
        elif choice == '4':
            symbol = input("Nhập mã cổ phiếu (mặc định FPT): ").strip().upper() or 'FPT'
            demo_clean_data(symbol)
        elif choice == '5':
            symbol = input("Nhập mã cổ phiếu (mặc định FPT): ").strip().upper() or 'FPT'
            demo_build_features(symbol)
        elif choice == '6':
            symbol = input("Nhập mã cổ phiếu (mặc định FPT): ").strip().upper() or 'FPT'
            demo_full_pipeline(symbol)
        elif choice == '0':
            logger.info("👋 Tạm biệt!")
        else:
            logger.warning("❌ Lựa chọn không hợp lệ!")
            
    except KeyboardInterrupt:
        logger.info("\n\n👋 Đã hủy!")
    except Exception as e:
        logger.error(f"\n❌ Lỗi: {e}")


if __name__ == "__main__":
    main()
