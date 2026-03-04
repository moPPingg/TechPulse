"""
Text cleaning for Vietnamese news. Research-usable: deterministic, no demo shortcuts.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Common boilerplate patterns (Vietnamese)
BOILERPLATE_PATTERNS = [
    r"đọc thêm\s*:.*$",
    r"theo\s+[\w\s]+$",
    r"nguồn\s*:.*$",
    r"\(\s*\)",
    r"\[.*?\]",
]


def strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse spaces and strip."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_boilerplate(text: str, patterns: Optional[list] = None) -> str:
    """Remove common footer/boilerplate lines."""
    if not text:
        return ""
    pats = patterns or BOILERPLATE_PATTERNS
    for p in pats:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return normalize_whitespace(text)


def clean_article_body(
    raw: str,
    strip_html_flag: bool = True,
    normalize_ws: bool = True,
    remove_boilerplate_flag: bool = True,
    max_chars: Optional[int] = None,
) -> str:
    """
    Full clean pipeline for article body. Idempotent for research.
    """
    if not raw:
        return ""
    if strip_html_flag:
        raw = strip_html(raw)
    if normalize_ws:
        raw = normalize_whitespace(raw)
    if remove_boilerplate_flag:
        raw = remove_boilerplate(raw)
    if max_chars and len(raw) > max_chars:
        raw = raw[:max_chars].rsplit(" ", 1)[0] if " " in raw[:max_chars] else raw[:max_chars]
    return raw
