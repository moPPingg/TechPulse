# -*- coding: utf-8 -*-
"""
Script để crawl dữ liệu 10 NĂM cho VN30
Tạo bởi: TechPulse Team
Mục đích: Lấy dữ liệu từ 2015-2024 cho Machine Learning

Cách chạy:
    python crawl_vn30_10_nam.py
"""

import sys
import io
from datetime import datetime

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.pipeline.vnindex30.fetch_vn30 import run_vn30_pipeline
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def crawl_10_nam():
    """
    Crawl dữ liệu 10 NĂM (2015-2024) cho toàn bộ VN30
    
    Thông số:
    - Thời gian: 01/01/2015 → 31/12/2024
    - Số mã: 30 (VN30)
    - Dữ liệu dự kiến: ~2,500 dòng/mã × 30 = 75,000 dòng
    - Thời gian chạy: 10-15 phút
    
    Output:
    - data/raw/vn30/: Dữ liệu thô (7 cột)
    - data/clean/vn30/: Dữ liệu sạch (7 cột)
    - data/features/vn30/: Dữ liệu features (45+ cột)
    """
    
    # Header thông tin
    print("\n" + "=" * 80)
    print("🚀 CRAWL DỮ LIỆU 10 NĂM CHO VN30")
    print("=" * 80)
    print(f"📅 Thời gian:      01/01/2015 → 31/12/2024 (10 năm)")
    print(f"📊 Tổng số mã:     30 mã (VN30)")
    print(f"📈 Dữ liệu dự kiến: ~2,500 dòng/mã = 75,000 dòng tổng")
    print(f"⏱️  Thời gian:      10-15 phút")
    print(f"🕐 Bắt đầu lúc:    {datetime.now().strftime('%H:%M:%S')}")
    print("")
    
    # Cảnh báo quan trọng
    print("⚠️  LƯU Ý QUAN TRỌNG:")
    print("   1. Đảm bảo kết nối Internet ổn định")
    print("   2. KHÔNG TẮT máy/terminal trong quá trình chạy")
    print("   3. Nếu 1 mã bị lỗi, script sẽ tự động bỏ qua và tiếp tục")
    print("   4. Có thể mất 10-15 phút - hãy kiên nhẫn!")
    print("")
    
    # Xác nhận
    try:
        user_input = input("📌 Nhấn ENTER để bắt đầu (hoặc Ctrl+C để hủy): ")
    except KeyboardInterrupt:
        print("\n\n❌ Đã hủy bởi người dùng")
        return
    
    print("\n" + "=" * 80)
    print("⏳ ĐANG CHẠY PIPELINE...")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Chạy pipeline với dữ liệu 10 năm
        run_vn30_pipeline(
            start_date='01/01/2015',      # ← 10 năm trước
            end_date='31/12/2024',         # ← Hiện tại
            raw_dir='data/raw/vn30',
            clean_dir='data/clean/vn30',
            features_dir='data/features/vn30'
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Thông báo hoàn thành
        print("\n" + "=" * 80)
        print("🎉 HOÀN THÀNH!")
        print("=" * 80)
        print(f"⏱️  Thời gian thực tế: {duration/60:.1f} phút ({duration:.0f} giây)")
        print(f"🕐 Kết thúc lúc:      {end_time.strftime('%H:%M:%S')}")
        print("")
        print("📁 Dữ liệu đã được lưu tại:")
        print("   ├─ data/raw/vn30/       (Dữ liệu thô - 7 cột)")
        print("   ├─ data/clean/vn30/     (Dữ liệu sạch - 7 cột)")
        print("   └─ data/features/vn30/  (Dữ liệu features - 45+ cột)")
        print("")
        print("✅ Bạn có thể dùng dữ liệu trong data/features/vn30/ cho:")
        print("   1. Machine Learning (dự báo giá)")
        print("   2. Anomaly Detection (phát hiện bất thường)")
        print("   3. Technical Analysis (phân tích kỹ thuật)")
        print("   4. Backtesting (kiểm thử chiến lược)")
        print("")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline bị ngắt bởi người dùng")
        print("💡 Dữ liệu đã crawl được vẫn được lưu trong các thư mục")
        
    except Exception as e:
        print("\n\n❌ LỖI XẢY RA:")
        print(f"   {e}")
        print("\n💡 Gợi ý:")
        print("   1. Kiểm tra kết nối Internet")
        print("   2. Thử chạy lại script")
        print("   3. Nếu vẫn lỗi, kiểm tra log phía trên")


def crawl_1_ma_demo(symbol='FPT'):
    """
    Demo: Crawl 10 năm cho 1 MÃ duy nhất (để test nhanh)
    
    Args:
        symbol: Mã cổ phiếu (mặc định: FPT)
    
    Cách dùng:
        >>> crawl_1_ma_demo('VCB')  # Crawl VCB
    """
    from src.crawl.cafef_scraper import fetch_price_cafef
    from src.clean.clean_price import clean_price
    from src.features.build_features import build_features_single
    
    print(f"\n🔍 DEMO: Crawl 10 năm cho {symbol}")
    print("=" * 60)
    
    try:
        # Bước 1: Crawl
        print(f"📥 [1/3] Đang crawl {symbol}...")
        df = fetch_price_cafef(
            symbol=symbol,
            start_date='01/01/2015',
            end_date='31/12/2024',
            page_size=3000,  # ← Tăng để chứa 10 năm
            timeout=60       # ← Tăng timeout
        )
        
        # Lưu raw
        raw_path = f'data/raw/vn30/{symbol}.csv'
        df.to_csv(raw_path, index=False)
        print(f"   ✅ Lấy được {len(df)} dòng")
        print(f"   📁 Lưu tại: {raw_path}")
        
        # Bước 2: Clean
        print(f"\n🧹 [2/3] Đang clean {symbol}...")
        from pathlib import Path
        clean_path = f'data/clean/vn30/{symbol}.csv'
        Path('data/clean/vn30').mkdir(parents=True, exist_ok=True)
        
        df_clean = clean_price(
            input_path=raw_path,
            output_path=clean_path,
            remove_duplicates=True,
            remove_nulls=True,
            validate=True
        )
        print(f"   ✅ Còn lại {len(df_clean)} dòng sau khi clean")
        print(f"   📁 Lưu tại: {clean_path}")
        
        # Bước 3: Features
        print(f"\n⚙️  [3/3] Đang tính features cho {symbol}...")
        df_features = build_features_single(
            filename=f'{symbol}.csv',
            clean_dir='data/clean/vn30',
            features_dir='data/features/vn30',
            drop_na=True,
            save_file=True
        )
        
        if df_features is not None:
            print(f"   ✅ Tính được {len(df_features.columns)} cột features")
            print(f"   ✅ Còn lại {len(df_features)} dòng sau khi drop NaN")
            print(f"   📁 Lưu tại: data/features/vn30/{symbol}.csv")
            
            # Hiển thị thống kê
            print(f"\n📊 THỐNG KÊ {symbol}:")
            print(f"   - Khoảng thời gian: {df_features['date'].min().date()} → {df_features['date'].max().date()}")
            print(f"   - Số năm dữ liệu:   {(df_features['date'].max() - df_features['date'].min()).days / 365:.1f} năm")
            print(f"   - Số dòng:          {len(df_features)}")
            print(f"   - Số cột:           {len(df_features.columns)}")
            print(f"   - Giá cao nhất:     {df_features['high'].max():,.0f}")
            print(f"   - Giá thấp nhất:    {df_features['low'].min():,.0f}")
            print(f"   - Volatility TB:    {df_features['volatility_20'].mean():.2f}%")
            print(f"   - RSI trung bình:   {df_features['rsi_14'].mean():.1f}")
            
            print("\n✅ HOÀN THÀNH DEMO!")
            return df_features
        else:
            print("   ❌ Lỗi khi tính features")
            return None
            
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        return None


def kiem_tra_features(symbol='FPT'):
    """
    Kiểm tra và phân tích file features đã có
    
    Args:
        symbol: Mã cổ phiếu cần kiểm tra
    """
    import pandas as pd
    from pathlib import Path
    
    features_path = Path(f'data/features/vn30/{symbol}.csv')
    
    if not features_path.exists():
        print(f"❌ Chưa có file features cho {symbol}")
        print(f"💡 Chạy crawl trước: crawl_1_ma_demo('{symbol}')")
        return
    
    print(f"\n🔍 PHÂN TÍCH FEATURES: {symbol}")
    print("=" * 70)
    
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Thông tin cơ bản
    print(f"\n📊 THÔNG TIN CƠ BẢN:")
    print(f"   Khoảng thời gian: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   Số năm:           {(df['date'].max() - df['date'].min()).days / 365:.1f} năm")
    print(f"   Số dòng:          {len(df):,}")
    print(f"   Số cột:           {len(df.columns)}")
    
    # Danh sách features
    base_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
    feature_cols = [col for col in df.columns if col not in base_cols]
    
    print(f"\n📈 FEATURES ({len(feature_cols)} cột):")
    print(f"   Returns:      {[c for c in feature_cols if 'return' in c]}")
    print(f"   MA:           {[c for c in feature_cols if c.startswith('ma_')]}")
    print(f"   EMA:          {[c for c in feature_cols if c.startswith('ema_')]}")
    print(f"   Volatility:   {[c for c in feature_cols if 'volatility' in c]}")
    print(f"   RSI:          {[c for c in feature_cols if 'rsi' in c]}")
    print(f"   MACD:         {[c for c in feature_cols if 'macd' in c]}")
    print(f"   Bollinger:    {[c for c in feature_cols if 'bb_' in c]}")
    print(f"   Volume:       {[c for c in feature_cols if 'volume' in c and c != 'volume']}")
    print(f"   Momentum:     {[c for c in feature_cols if 'momentum' in c]}")
    
    # Thống kê quan trọng
    print(f"\n💡 THỐNG KÊ QUAN TRỌNG:")
    print(f"   Giá cao nhất:      {df['high'].max():,.0f}")
    print(f"   Giá thấp nhất:     {df['low'].min():,.0f}")
    print(f"   Volume trung bình: {df['volume'].mean():,.0f}")
    print(f"   Return_1d TB:      {df['return_1d'].mean():.2f}%")
    print(f"   Volatility_20 TB:  {df['volatility_20'].mean():.2f}%")
    print(f"   RSI_14 TB:         {df['rsi_14'].mean():.1f}")
    
    # Tìm ngày đặc biệt
    print(f"\n🎯 NGÀY ĐẶC BIỆT:")
    
    # Ngày tăng mạnh nhất
    idx_max_return = df['return_1d'].idxmax()
    print(f"   📈 Tăng mạnh nhất:    {df.loc[idx_max_return, 'date'].date()} "
          f"(+{df.loc[idx_max_return, 'return_1d']:.2f}%)")
    
    # Ngày giảm mạnh nhất
    idx_min_return = df['return_1d'].idxmin()
    print(f"   📉 Giảm mạnh nhất:    {df.loc[idx_min_return, 'date'].date()} "
          f"({df.loc[idx_min_return, 'return_1d']:.2f}%)")
    
    # Ngày volume cao nhất
    idx_max_volume = df['volume'].idxmax()
    print(f"   📊 Volume cao nhất:   {df.loc[idx_max_volume, 'date'].date()} "
          f"({df.loc[idx_max_volume, 'volume']:,.0f})")
    
    # Ngày volatility cao nhất
    idx_max_vol = df['volatility_20'].idxmax()
    print(f"   ⚡ Biến động cao nhất: {df.loc[idx_max_vol, 'date'].date()} "
          f"(volatility={df.loc[idx_max_vol, 'volatility_20']:.2f}%)")
    
    print(f"\n✅ Phân tích hoàn tất!")


if __name__ == "__main__":
    """
    CÁCH SỬ DỤNG SCRIPT NÀY:
    
    1. Crawl TOÀN BỘ VN30 (10 năm):
       >>> python crawl_vn30_10_nam.py
       (hoặc gọi hàm crawl_10_nam() trong Python)
    
    2. Demo nhanh với 1 mã:
       >>> crawl_1_ma_demo('FPT')
    
    3. Kiểm tra features đã có:
       >>> kiem_tra_features('FPT')
    """
    
    print("\n" + "=" * 80)
    print("📚 SCRIPT CRAWL DỮ LIỆU 10 NĂM")
    print("=" * 80)
    print("\nChọn chế độ:")
    print("  [1] Crawl toàn bộ VN30 (10 năm) - Mất 10-15 phút")
    print("  [2] Demo nhanh 1 mã (FPT) - Mất 1-2 phút")
    print("  [3] Kiểm tra features có sẵn")
    print("  [0] Thoát")
    print("")
    
    try:
        choice = input("Nhập lựa chọn [1/2/3/0]: ").strip()
        
        if choice == '1':
            crawl_10_nam()
        elif choice == '2':
            symbol = input("Nhập mã cổ phiếu (mặc định FPT): ").strip().upper() or 'FPT'
            crawl_1_ma_demo(symbol)
        elif choice == '3':
            symbol = input("Nhập mã cổ phiếu (mặc định FPT): ").strip().upper() or 'FPT'
            kiem_tra_features(symbol)
        elif choice == '0':
            print("👋 Tạm biệt!")
        else:
            print("❌ Lựa chọn không hợp lệ!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Đã hủy!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
