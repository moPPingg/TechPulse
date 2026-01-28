"""
Logging utilities cho toàn bộ dự án

Cung cấp logger chuẩn với format đẹp và dễ đọc
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Tạo logger với format chuẩn cho console
    
    Args:
        name: Tên logger (thường dùng __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom format string (optional)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data...")
        >>> logger.warning("Missing values found")
        >>> logger.error("Failed to fetch data")
    """
    logger = logging.getLogger(name)
    
    # Tránh thêm handler trùng lặp
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(
        format_str,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.setLevel(level)
    
    # Tránh propagate lên root logger
    logger.propagate = False
    
    return logger


def get_file_logger(
    name: str,
    log_file: str,
    level: int = logging.DEBUG,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Tạo logger ghi vào file (ngoài console)
    
    Args:
        name: Tên logger
        log_file: Đường dẫn file log
        level: Logging level
        format_str: Custom format string
    
    Returns:
        Logger instance ghi cả console và file
    
    Example:
        >>> logger = get_file_logger(__name__, 'logs/app.log')
        >>> logger.info("This goes to both console and file")
    """
    # Lấy logger console trước
    logger = get_logger(name, level, format_str)
    
    # Tạo thư mục logs nếu chưa có
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # Format
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
):
    """
    Setup logging cho toàn bộ application
    
    Args:
        level: Default logging level
        log_file: File để ghi log (optional)
    
    Example:
        >>> setup_logging(logging.DEBUG, 'logs/app.log')
    """
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # File handler (nếu có)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        root_logger.addHandler(file_handler)
