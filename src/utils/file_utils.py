"""
File I/O utilities

Các hàm tiện ích cho việc đọc/ghi file
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union
import json
import yaml


def save_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    create_dirs: bool = True,
    **kwargs
) -> None:
    """
    Lưu DataFrame vào CSV với error handling
    
    Args:
        df: DataFrame cần lưu
        path: Đường dẫn file
        create_dirs: Tự động tạo thư mục nếu chưa có
        **kwargs: Tham số cho df.to_csv()
    
    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> save_csv(df, 'data/output.csv')
    """
    file_path = Path(path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default parameters
    if 'index' not in kwargs:
        kwargs['index'] = False
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    
    df.to_csv(file_path, **kwargs)


def load_csv(
    path: Union[str, Path],
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    Đọc CSV với error handling
    
    Args:
        path: Đường dẫn file
        **kwargs: Tham số cho pd.read_csv()
    
    Returns:
        DataFrame hoặc None nếu file không tồn tại
    
    Example:
        >>> df = load_csv('data/input.csv')
        >>> if df is not None:
        >>>     print(df.head())
    """
    file_path = Path(path)
    
    if not file_path.exists():
        return None
    
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Đảm bảo thư mục tồn tại, tạo nếu chưa có
    
    Args:
        path: Đường dẫn thư mục
    
    Returns:
        Path object của thư mục
    
    Example:
        >>> ensure_dir('data/raw/vn30')
        >>> # Thư mục đã được tạo nếu chưa có
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def list_files(
    directory: Union[str, Path],
    pattern: str = '*',
    recursive: bool = False
) -> list:
    """
    List tất cả files trong thư mục
    
    Args:
        directory: Thư mục cần list
        pattern: Pattern để filter (vd: '*.csv')
        recursive: Tìm kiếm đệ quy trong subfolder
    
    Returns:
        List các Path objects
    
    Example:
        >>> files = list_files('data/raw/vn30', '*.csv')
        >>> for file in files:
        >>>     print(file.name)
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    if recursive:
        return list(dir_path.rglob(pattern))
    else:
        return list(dir_path.glob(pattern))


def load_json(path: Union[str, Path]) -> Optional[dict]:
    """
    Đọc file JSON
    
    Args:
        path: Đường dẫn file JSON
    
    Returns:
        Dictionary hoặc None nếu lỗi
    """
    file_path = Path(path)
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON {path}: {e}")
        return None


def save_json(data: dict, path: Union[str, Path], indent: int = 2):
    """
    Lưu dictionary vào file JSON
    
    Args:
        data: Dictionary cần lưu
        path: Đường dẫn file
        indent: Số space indent (default: 2)
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_yaml(path: Union[str, Path]) -> Optional[dict]:
    """
    Đọc file YAML
    
    Args:
        path: Đường dẫn file YAML
    
    Returns:
        Dictionary hoặc None nếu lỗi
    """
    file_path = Path(path)
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML {path}: {e}")
        return None


def get_file_size(path: Union[str, Path]) -> int:
    """
    Lấy kích thước file (bytes)
    
    Args:
        path: Đường dẫn file
    
    Returns:
        Kích thước file (bytes)
    """
    return Path(path).stat().st_size


def get_file_size_mb(path: Union[str, Path]) -> float:
    """
    Lấy kích thước file (MB)
    
    Args:
        path: Đường dẫn file
    
    Returns:
        Kích thước file (MB)
    """
    return get_file_size(path) / (1024 * 1024)
