# -*- coding: utf-8 -*-
"""
Script để lấy toàn bộ 30 mã cổ phiếu VN30 từ CafeF
và chạy full pipeline: Crawl → Clean → Features

Author: Auto-generated
Date: 2026-01-20
"""
import sys
import io

# Fix encoding cho Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.pipeline.runcrawler.run_crawler import crawl_many
from src.clean.clean_price import clean_many
from src.features.build_features import build_features
import logging

# ============================================================================
# CẤU HÌNH LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DANH SÁCH 30 MÃ VN30 (Cập nhật Q1/2026)
# ============================================================================
# Lưu ý: Danh sách này thay đổi mỗi quý, cần kiểm tra tại:
# https://www.hsx.vn/Modules/Listed/Web/SymbolList/faad6e1b-8646-48aa-8f6f-b6fc092d714d?fid=a938a51449064a84a7b9bd99bf49c97e
VN30_SYMBOLS = [
    'ACB',  # Ngân hàng Á Châu
    'BCM',  # Khoáng sản Bắc Cạn
    'BID',  # Ngân hàng BIDV
    'BVH',  # Bảo Việt Holdings
    'CTG',  # Ngân hàng Vietinbank
    'FPT',  # FPT Corporation
    'GAS',  # PetroVietnam Gas
    'GVR',  # Cao su Việt Nam
    'HDB',  # Ngân hàng HDBank
    'HPG',  # Hòa Phát Group
    'MBB',  # Ngân hàng MB
    'MSN',  # Masan Group
    'MWG',  # Mobile World
    'PLX',  # Petrolimex
    'POW',  # PetroVietnam Power
    'SAB',  # Sabeco
    'SSI',  # SSI Securities
    'STB',  # Ngân hàng Sacombank
    'TCB',  # Ngân hàng Techcombank
    'TPB',  # Ngân hàng TPBank
    'VCB',  # Ngân hàng Vietcombank
    'VHM',  # Vinhomes
    'VIB',  # Ngân hàng VIB
    'VIC',  # Vingroup
    'VJC',  # Vietjet Air
    'VNM',  # Vinamilk
    'VPB',  # Ngân hàng VPBank
    'VRE',  # Vincom Retail
    'SSB',  # Ngân hàng SeABank
    'PDR',  # Phát Đạt
]


def run_vn30_pipeline(
    start_date: str,
    end_date: str,
    raw_dir: str = 'data/raw/vn30',
    clean_dir: str = 'data/clean/vn30',
    features_dir: str = 'data/features/vn30'
):
    """
    Chạy toàn bộ pipeline cho VN30: Crawl → Clean → Features
    
    Pipeline gồm 3 bước:
    1. CRAWL: Lấy dữ liệu từ CafeF API
    2. CLEAN: Làm sạch, validate data quality
    3. FEATURES: Tính toán technical indicators
    
    Args:
        start_date: Ngày bắt đầu, format 'DD/MM/YYYY' (vd: '01/01/2024')
        end_date: Ngày kết thúc, format 'DD/MM/YYYY' (vd: '31/12/2024')
        raw_dir: Thư mục lưu raw data (default: 'data/raw/vn30')
        clean_dir: Thư mục lưu clean data (default: 'data/clean/vn30')
        features_dir: Thư mục lưu features (default: 'data/features/vn30')
    
    Returns:
        None (lưu files vào disk)
    
    Example:
        >>> run_vn30_pipeline('01/01/2024', '31/12/2024')
        # Sẽ tạo 90 files (30 raw + 30 clean + 30 features)
    """
    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU PIPELINE VN30")
    logger.info("=" * 80)
    logger.info(f"📅 Khoảng thời gian: {start_date} → {end_date}")
    logger.info(f"📊 Tổng số mã: {len(VN30_SYMBOLS)}")
    
    # ========================================================================
    # BƯỚC 1: CRAWL DỮ LIỆU TỪ CAFEF
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("📥 BƯỚC 1/3: CRAWL DỮ LIỆU VN30")
    logger.info("=" * 80)
    logger.info("Đang gọi API CafeF để lấy dữ liệu lịch sử...")
    
    try:
        raw_results = crawl_many(
            symbols=VN30_SYMBOLS,
            start_date=start_date,
            end_date=end_date,
            save_dir=raw_dir,
            combine=True,        # Tạo thêm file combined chứa tất cả mã
            skip_on_error=True   # Tiếp tục nếu 1 mã bị lỗi
        )
        
        logger.info(f"✅ Crawl hoàn tất: {len(raw_results)}/{len(VN30_SYMBOLS)} mã thành công")
        
        if not raw_results:
            logger.error("❌ Không có dữ liệu nào được crawl. Dừng pipeline.")
            logger.error("Nguyên nhân có thể:")
            logger.error("  - Không có kết nối Internet")
            logger.error("  - API CafeF đang bảo trì")
            logger.error("  - Khoảng ngày không hợp lệ")
            return
            
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình crawl: {e}")
        return
    
    # ========================================================================
    # BƯỚC 2: CLEAN DỮ LIỆU
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🧹 BƯỚC 2/3: CLEAN VÀ VALIDATE DỮ LIỆU")
    logger.info("=" * 80)
    logger.info("Đang làm sạch dữ liệu:")
    logger.info("  - Loại bỏ duplicates")
    logger.info("  - Loại bỏ null values")
    logger.info("  - Validate OHLC logic")
    logger.info("  - Kiểm tra giá âm, giá = 0")
    
    try:
        clean_results = clean_many(
            raw_dir=raw_dir,
            clean_dir=clean_dir,
            skip_on_error=True,
            remove_duplicates=True,
            remove_nulls=True,
            validate=True
        )
        
        logger.info(f"✅ Clean hoàn tất: {len(clean_results)} files")
        
        if not clean_results:
            logger.warning("⚠️  Không có file clean được. Bỏ qua bước features.")
            return
            
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình clean: {e}")
        return
    
    # ========================================================================
    # BƯỚC 3: BUILD FEATURES (TECHNICAL INDICATORS)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("⚙️  BƯỚC 3/3: BUILD TECHNICAL FEATURES")
    logger.info("=" * 80)
    logger.info("Đang tính toán các chỉ số kỹ thuật:")
    logger.info("  - Returns (1d, 5d, 10d, 20d)")
    logger.info("  - Moving Averages (MA5, MA10, MA20, MA50)")
    logger.info("  - EMA (12, 26)")
    logger.info("  - Volatility (5d, 10d, 20d)")
    logger.info("  - RSI (14)")
    logger.info("  - MACD, Signal, Histogram")
    logger.info("  - Bollinger Bands (upper, middle, lower, width)")
    logger.info("  - Volume features")
    logger.info("  - Momentum indicators")
    logger.info("  - Price range & ATR")
    
    try:
        feature_results = build_features(
            clean_dir=clean_dir,
            features_dir=features_dir,
            skip_on_error=True,
            drop_na=True
        )
        
        logger.info(f"✅ Features hoàn tất: {len(feature_results)} files")
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình build features: {e}")
        return
    
    # ========================================================================
    # TỔNG KẾT KẾT QUẢ
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🎉 HOÀN THÀNH PIPELINE VN30")
    logger.info("=" * 80)
    logger.info(f"📁 Raw data:     {len(raw_results)} files → {raw_dir}/")
    logger.info(f"📁 Clean data:   {len(clean_results)} files → {clean_dir}/")
    logger.info(f"📁 Features:     {len(feature_results)} files → {features_dir}/")
    logger.info("=" * 80)
    
    # Hiển thị sample của 1 file features
    if feature_results:
        sample_file = list(feature_results.keys())[0]
        sample_df = feature_results[sample_file]
        logger.info(f"\n📊 Sample features từ {sample_file}:")
        logger.info(f"   - Tổng số dòng: {len(sample_df)}")
        logger.info(f"   - Tổng số cột: {len(sample_df.columns)}")
        logger.info(f"   - Columns: {list(sample_df.columns[:10])}...")
    
    logger.info("\n✅ Bạn có thể sử dụng data cho:")
    logger.info("   1. Machine Learning (prediction)")
    logger.info("   2. Technical Analysis")
    logger.info("   3. Backtesting trading strategies")
    logger.info("   4. Data visualization")


def fetch_vn30_only(
    start_date: str,
    end_date: str,
    save_dir: str = 'data/raw/vn30'
):
    """
    Chỉ crawl VN30 (KHÔNG clean, KHÔNG tính features)
    
    Dùng khi:
    - Bạn chỉ cần raw data
    - Muốn tự xử lý data theo cách riêng
    - Crawl nhanh để kiểm tra
    
    Args:
        start_date: Ngày bắt đầu, format 'DD/MM/YYYY'
        end_date: Ngày kết thúc, format 'DD/MM/YYYY'
        save_dir: Thư mục lưu (default: 'data/raw/vn30')
    
    Returns:
        List of DataFrames (mỗi mã 1 DataFrame)
    
    Example:
        >>> data = fetch_vn30_only('01/01/2024', '31/12/2024')
        >>> print(f"Lấy được {len(data)} mã")
        >>> # Xem data của FPT
        >>> fpt_data = [df for df in data if df['ticker'].iloc[0] == 'FPT'][0]
        >>> print(fpt_data.head())
    """
    logger.info("=" * 80)
    logger.info("📥 CRAWLING VN30 (CHỈ RAW DATA)")
    logger.info("=" * 80)
    logger.info(f"Tổng số mã: {len(VN30_SYMBOLS)}")
    logger.info(f"Khoảng thời gian: {start_date} → {end_date}")
    
    results = crawl_many(
        symbols=VN30_SYMBOLS,
        start_date=start_date,
        end_date=end_date,
        save_dir=save_dir,
        combine=True,
        skip_on_error=True
    )
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ HOÀN THÀNH! Đã lấy {len(results)}/{len(VN30_SYMBOLS)} mã VN30")
    logger.info(f"📁 Files được lưu tại: {save_dir}/")
    logger.info("=" * 80)
    
    return results


def update_vn30_symbols(new_symbols: list):
    """
    Cập nhật danh sách VN30 (thay đổi mỗi quý)
    
    Args:
        new_symbols: List các mã mới (phải có đúng 30 mã)
    
    Example:
        >>> new_list = ['ACB', 'BID', 'CTG', ...]  # 30 mã
        >>> update_vn30_symbols(new_list)
    """
    global VN30_SYMBOLS
    
    if len(new_symbols) != 30:
        logger.error(f"❌ VN30 phải có đúng 30 mã. Bạn cung cấp {len(new_symbols)} mã.")
        return False
    
    VN30_SYMBOLS = [symbol.upper().strip() for symbol in new_symbols]
    logger.info(f"✅ Đã cập nhật danh sách VN30: {VN30_SYMBOLS}")
    return True


# ============================================================================
# MAIN - CHẠY KHI EXECUTE FILE TRỰC TIẾP
# ============================================================================
if __name__ == "__main__":
    """
    Có 2 cách sử dụng:
    
    CÁCH 1: Chỉ crawl raw data (nhanh, ~2 phút)
    CÁCH 2: Chạy full pipeline (lâu hơn, ~5-10 phút)
    
    Uncomment cách nào bạn muốn dùng ở dưới
    """
    
    # ========================================================================
    # CÁCH 1: CHỈ CRAWL RAW DATA (Nhanh nhất)
    # ========================================================================
    # Uncomment 2 dòng dưới để chạy
    # print("\n🔹 Chế độ: CHỈ CRAWL RAW DATA")
    # fetch_vn30_only('01/01/2024', '20/01/2026')
    
    
    # ========================================================================
    # CÁCH 2: CHẠY FULL PIPELINE (Crawl → Clean → Features)
    # ========================================================================
    # Uncomment 2 dòng dưới để chạy
    print("\n🔹 Chế độ: FULL PIPELINE (Crawl + Clean + Features)")
    run_vn30_pipeline(
        start_date='01/01/2024',
        end_date='20/01/2026'
    )
    
    
    # ========================================================================
    # TÙY CHỌN: Thay đổi thư mục lưu
    # ========================================================================
    # run_vn30_pipeline(
    #     start_date='01/01/2024',
    #     end_date='20/01/2026',
    #     raw_dir='my_data/raw',
    #     clean_dir='my_data/clean',
    #     features_dir='my_data/features'
    # )
