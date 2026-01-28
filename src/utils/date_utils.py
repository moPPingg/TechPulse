"""
Date utilities

Các hàm tiện ích cho xử lý ngày tháng
"""

from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd


def format_date(date: datetime, format_str: str = '%d/%m/%Y') -> str:
    """
    Format datetime thành string
    
    Args:
        date: Datetime object
        format_str: Format string (default: DD/MM/YYYY)
    
    Returns:
        Date string
    
    Example:
        >>> dt = datetime(2024, 1, 15)
        >>> format_date(dt)
        '15/01/2024'
    """
    return date.strftime(format_str)


def parse_date(date_str: str, format_str: str = '%d/%m/%Y') -> datetime:
    """
    Parse string thành datetime
    
    Args:
        date_str: Date string
        format_str: Format string
    
    Returns:
        Datetime object
    
    Example:
        >>> dt = parse_date('15/01/2024')
        >>> print(dt.year, dt.month, dt.day)
        2024 1 15
    """
    return datetime.strptime(date_str, format_str)


def get_date_range(
    start_date: str,
    end_date: str,
    format_str: str = '%d/%m/%Y'
) -> Tuple[datetime, datetime]:
    """
    Parse start và end date thành datetime objects
    
    Args:
        start_date: Start date string
        end_date: End date string
        format_str: Format string
    
    Returns:
        Tuple (start_datetime, end_datetime)
    
    Example:
        >>> start, end = get_date_range('01/01/2024', '31/12/2024')
        >>> print(f"From {start} to {end}")
    """
    start_dt = parse_date(start_date, format_str)
    end_dt = parse_date(end_date, format_str)
    return start_dt, end_dt


def get_n_years_ago(n: int, from_date: datetime = None) -> datetime:
    """
    Lấy ngày n năm trước
    
    Args:
        n: Số năm
        from_date: Ngày bắt đầu (default: hôm nay)
    
    Returns:
        Datetime n năm trước
    
    Example:
        >>> ten_years_ago = get_n_years_ago(10)
        >>> print(format_date(ten_years_ago))
    """
    if from_date is None:
        from_date = datetime.now()
    
    return from_date.replace(year=from_date.year - n)


def get_trading_days(
    start_date: datetime,
    end_date: datetime
) -> int:
    """
    Ước tính số ngày giao dịch (loại thứ 7, CN)
    
    Args:
        start_date: Ngày bắt đầu
        end_date: Ngày kết thúc
    
    Returns:
        Số ngày giao dịch ước tính
    
    Note:
        Đây là ước tính đơn giản, không tính ngày lễ
    """
    total_days = (end_date - start_date).days
    weeks = total_days // 7
    remaining_days = total_days % 7
    
    # Ước tính: 5 ngày/tuần
    trading_days = weeks * 5
    
    # Thêm ngày còn lại (trừ weekend)
    for i in range(remaining_days):
        day = start_date + timedelta(days=i)
        if day.weekday() < 5:  # Monday=0, Friday=4
            trading_days += 1
    
    return trading_days


def get_date_list(
    start_date: datetime,
    end_date: datetime,
    freq: str = 'D'
) -> List[datetime]:
    """
    Tạo list các ngày từ start đến end
    
    Args:
        start_date: Ngày bắt đầu
        end_date: Ngày kết thúc
        freq: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
    
    Returns:
        List các datetime objects
    
    Example:
        >>> dates = get_date_list(
        >>>     datetime(2024, 1, 1),
        >>>     datetime(2024, 1, 7)
        >>> )
        >>> print(len(dates))  # 7 days
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    return date_range.tolist()


def is_trading_day(date: datetime) -> bool:
    """
    Kiểm tra có phải ngày giao dịch không (đơn giản)
    
    Args:
        date: Datetime object
    
    Returns:
        True nếu là ngày giao dịch (không phải thứ 7, CN)
    
    Note:
        Không tính ngày lễ, chỉ check weekend
    """
    return date.weekday() < 5  # Monday=0, Friday=4


def get_next_trading_day(date: datetime) -> datetime:
    """
    Lấy ngày giao dịch tiếp theo
    
    Args:
        date: Datetime object
    
    Returns:
        Ngày giao dịch tiếp theo
    """
    next_day = date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day


def get_previous_trading_day(date: datetime) -> datetime:
    """
    Lấy ngày giao dịch trước đó
    
    Args:
        date: Datetime object
    
    Returns:
        Ngày giao dịch trước đó
    """
    prev_day = date - timedelta(days=1)
    while not is_trading_day(prev_day):
        prev_day -= timedelta(days=1)
    return prev_day
