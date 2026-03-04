# -*- coding: utf-8 -*-
"""
Script để lấy toàn bộ 30 mã cổ phiếu VN30 từ CafeF
và chạy full pipeline: Crawl → Clean → Features

Cấu hình đọc từ configs/config.yaml và configs/symbols.yaml
Author: Auto-generated
Date: 2026-01-20
"""
import sys
import io
from pathlib import Path

# Thêm project root vào Python path (để import src.* hoạt động)
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Fix encoding cho Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.pipeline.runcrawler.run_crawler import crawl_many
from src.clean.clean_price import clean_many
from src.features.build_features import build_features
from src.utils.file_utils import load_yaml
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
# FALLBACK - DANH SÁCH VN30 (dùng khi không có symbols.yaml)
# ============================================================================
VN30_SYMBOLS_FALLBACK = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SSI', 'STB', 'TCB', 'TPB',
    'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE', 'SSB', 'PDR',
]


def _get_config_path(filename: str) -> Path:
    """Lấy đường dẫn file config (từ project root)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / 'configs' / filename


def load_pipeline_config() -> dict:
    """
    Đọc cấu hình từ YAML. Fallback về giá trị mặc định nếu file không tồn tại.
    
    Returns:
        dict với keys: symbols, start_date, end_date, raw_dir, clean_dir,
        features_dir, page_size, skip_on_error, clean_opts
    """
    config_path = _get_config_path('config.yaml')
    symbols_path = _get_config_path('symbols.yaml')
    
    config = load_yaml(config_path) if config_path.exists() else {}
    symbols_config = load_yaml(symbols_path) if symbols_path.exists() else {}
    
    # Lấy symbols từ symbols.yaml, fallback VN30_SYMBOLS_FALLBACK
    symbols = symbols_config.get('vn30', VN30_SYMBOLS_FALLBACK)
    symbols = [s.strip().upper() for s in symbols if isinstance(s, str)]
    if not symbols:
        symbols = VN30_SYMBOLS_FALLBACK
    
    # Lấy config crawl
    crawl = config.get('crawl', {})
    data = config.get('data', {})
    clean_cfg = config.get('clean', {})
    pipeline_cfg = config.get('pipeline', {})
    
    return {
        'symbols': symbols,
        'start_date': crawl.get('start_date', '01/01/2015'),
        'end_date': crawl.get('end_date', '31/01/2026'),
        'raw_dir': data.get('raw_dir', 'data/raw/vn30'),
        'clean_dir': data.get('clean_dir', 'data/clean/vn30'),
        'features_dir': data.get('features_dir', 'data/features/vn30'),
        'page_size': crawl.get('page_size', 3000),
        'skip_on_error': pipeline_cfg.get('skip_on_error', True),
        'remove_duplicates': clean_cfg.get('remove_duplicates', True),
        'remove_nulls': clean_cfg.get('remove_nulls', True),
        'validate': clean_cfg.get('validate', True),
    }


def run_vn30_pipeline(
    start_date: str = None,
    end_date: str = None,
    raw_dir: str = None,
    clean_dir: str = None,
    features_dir: str = None,
    symbols: list = None,
    page_size: int = None,
    skip_on_error: bool = None,
    **kwargs
):
    """
    Chạy toàn bộ pipeline cho VN30: Crawl → Clean → Features
    
    Tham số mặc định lấy từ configs/config.yaml. Truyền tham số để override.
    
    Args:
        start_date: Ngày bắt đầu (DD/MM/YYYY). None = đọc từ config
        end_date: Ngày kết thúc (DD/MM/YYYY). None = đọc từ config
        raw_dir, clean_dir, features_dir: Đường dẫn thư mục. None = đọc từ config
        symbols: Danh sách mã. None = đọc từ configs/symbols.yaml
        page_size: Số bản ghi/request. None = đọc từ config
        skip_on_error: Bỏ qua lỗi và tiếp tục. None = đọc từ config
    
    Example:
        >>> run_vn30_pipeline()  # Dùng toàn bộ config từ YAML
        >>> run_vn30_pipeline(start_date='01/01/2020', end_date='31/12/2024')  # Override ngày
    """
    cfg = load_pipeline_config()
    
    # Override bằng tham số truyền vào
    start_date = start_date or cfg['start_date']
    end_date = end_date or cfg['end_date']
    raw_dir = raw_dir or cfg['raw_dir']
    clean_dir = clean_dir or cfg['clean_dir']
    features_dir = features_dir or cfg['features_dir']
    symbols = symbols or cfg['symbols']
    page_size = page_size if page_size is not None else cfg['page_size']
    skip_on_error = skip_on_error if skip_on_error is not None else cfg['skip_on_error']
    
    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU PIPELINE VN30")
    logger.info("=" * 80)
    logger.info(f"📅 Khoảng thời gian: {start_date} → {end_date}")
    logger.info(f"📊 Tổng số mã: {len(symbols)}")
    
    # BƯỚC 1: CRAWL
    logger.info("\n" + "=" * 80)
    logger.info("📥 BƯỚC 1/3: CRAWL DỮ LIỆU VN30")
    logger.info("=" * 80)
    
    try:
        raw_results = crawl_many(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            save_dir=raw_dir,
            combine=True,
            skip_on_error=skip_on_error,
            page_size=page_size,
        )
        
        if not raw_results:
            logger.error("❌ Không có dữ liệu nào được crawl. Dừng pipeline.")
            return
            
    except Exception as e:
        logger.error(f"❌ Lỗi crawl: {e}")
        return
    
    logger.info(f"✅ Crawl hoàn tất: {len(raw_results)}/{len(symbols)} mã thành công")
    
    # BƯỚC 2: CLEAN
    logger.info("\n" + "=" * 80)
    logger.info("🧹 BƯỚC 2/3: CLEAN DỮ LIỆU")
    logger.info("=" * 80)
    
    try:
        clean_results = clean_many(
            raw_dir=raw_dir,
            clean_dir=clean_dir,
            skip_on_error=skip_on_error,
            remove_duplicates=cfg['remove_duplicates'],
            remove_nulls=cfg['remove_nulls'],
            validate=cfg['validate'],
        )
        
        if not clean_results:
            logger.warning("⚠️  Không có file clean được. Bỏ qua bước features.")
            return
            
    except Exception as e:
        logger.error(f"❌ Lỗi clean: {e}")
        return
    
    logger.info(f"✅ Clean hoàn tất: {len(clean_results)} files")
    
    # BƯỚC 3: FEATURES
    logger.info("\n" + "=" * 80)
    logger.info("⚙️  BƯỚC 3/3: BUILD FEATURES")
    logger.info("=" * 80)
    
    try:
        feature_results = build_features(
            clean_dir=clean_dir,
            features_dir=features_dir,
            skip_on_error=skip_on_error,
            drop_na=True,
        )
    except Exception as e:
        logger.error(f"❌ Lỗi build features: {e}")
        return
    
    logger.info(f"✅ Features hoàn tất: {len(feature_results)} files")
    
    # TỔNG KẾT
    logger.info("\n" + "=" * 80)
    logger.info("🎉 HOÀN THÀNH PIPELINE VN30")
    logger.info("=" * 80)
    logger.info(f"📁 Raw:     {len(raw_results)} files → {raw_dir}/")
    logger.info(f"📁 Clean:   {len(clean_results)} files → {clean_dir}/")
    logger.info(f"📁 Features: {len(feature_results)} files → {features_dir}/")
    logger.info("=" * 80)


def fetch_vn30_only(
    start_date: str = None,
    end_date: str = None,
    save_dir: str = None,
    symbols: list = None,
    page_size: int = None,
):
    """
    Chỉ crawl VN30 (KHÔNG clean, KHÔNG tính features).
    Tham số mặc định lấy từ config.
    """
    cfg = load_pipeline_config()
    
    start_date = start_date or cfg['start_date']
    end_date = end_date or cfg['end_date']
    save_dir = save_dir or cfg['raw_dir']
    symbols = symbols or cfg['symbols']
    page_size = page_size if page_size is not None else cfg['page_size']
    
    logger.info("=" * 80)
    logger.info("📥 CRAWLING VN30 (CHỈ RAW DATA)")
    logger.info("=" * 80)
    logger.info(f"Tổng số mã: {len(symbols)} | {start_date} → {end_date}")
    
    results = crawl_many(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        save_dir=save_dir,
        combine=True,
        skip_on_error=cfg['skip_on_error'],
        page_size=page_size,
    )
    
    logger.info(f"✅ HOÀN THÀNH! {len(results)}/{len(symbols)} mã → {save_dir}/")
    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    """
    Chạy full pipeline với cấu hình từ configs/config.yaml và configs/symbols.yaml
    Sửa file YAML để thay đổi ngày, mã cổ phiếu, v.v.
    """
    print("\n🔹 Chế độ: FULL PIPELINE (Crawl + Clean + Features)")
    run_vn30_pipeline()
    
    # Hoặc override:
    # run_vn30_pipeline(start_date='01/01/2020', end_date='31/12/2024')
    # fetch_vn30_only()  # Chỉ crawl raw
