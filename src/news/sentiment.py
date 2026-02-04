"""
Sentiment extraction for Vietnamese financial text. Research baseline: lexicon-based.
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Minimal Vietnamese financial sentiment lexicon (research baseline; extend via config/file later)
POSITIVE = [
    "tăng", "lên", "khả quan", "tích cực", "lạc quan", "mạnh", "thành công", "lợi nhuận", "tăng trưởng",
    "cải thiện", "vượt", "kỳ vọng", "triển vọng", "mua vào", "khuyến nghị mua", "accumulate",
]
NEGATIVE = [
    "giảm", "xuống", "tiêu cực", "bi quan", "yếu", "thua lỗ", "sụt giảm", "lo ngại", "rủi ro",
    "bán ra", "khuyến nghị bán", "giảm giá", "điều chỉnh giảm", "scandal", "vi phạm",
]
# Normalize: lower, strip
POSITIVE = [w.strip().lower() for w in POSITIVE]
NEGATIVE = [w.strip().lower() for w in NEGATIVE]


def sentiment_lexicon(text: str) -> Tuple[float, str]:
    """
    Returns (score, method). Score in [-1, 1]; method = 'lexicon'.
    """
    if not text or not isinstance(text, str):
        return 0.0, "lexicon"
    t = text.lower()
    pos_count = sum(1 for w in POSITIVE if w in t)
    neg_count = sum(1 for w in NEGATIVE if w in t)
    total = pos_count + neg_count
    if total == 0:
        return 0.0, "lexicon"
    score = (pos_count - neg_count) / total
    return max(-1.0, min(1.0, score)), "lexicon"
