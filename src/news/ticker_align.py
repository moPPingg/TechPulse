"""
Align articles to VN30 tickers by matching symbols and company names in title/body.
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Set
import yaml

logger = logging.getLogger(__name__)

# Default VN30 symbols (fallback if no config)
VN30_DEFAULT = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSI", "STB", "TCB", "TPB",
    "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE", "SSB", "PDR",
]

# Optional: short company names / aliases per ticker (expand via config)
ALIASES: dict = {}  # ticker -> [names]


def load_symbols_and_aliases(symbols_path: Optional[str] = None) -> Tuple[List[str], dict]:
    """Load VN30 list and optional ticker -> [aliases] from YAML."""
    symbols = list(VN30_DEFAULT)
    aliases = {}
    path = Path(symbols_path) if symbols_path else None
    if path and path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if data and isinstance(data, dict):
                vn30 = data.get("vn30", data.get("symbols", []))
                if isinstance(vn30, list):
                    symbols = [str(s).strip().upper() for s in vn30 if s]
                aliases = data.get("ticker_aliases", {})
                if not isinstance(aliases, dict):
                    aliases = {}
        except Exception as e:
            logger.warning("Could not load symbols from %s: %s", path, e)
    return symbols, aliases


def extract_tickers_from_text(
    text: str,
    symbols: List[str],
    aliases: Optional[dict] = None,
) -> List[Tuple[str, str]]:
    """
    Match tickers and aliases in text. Returns [(ticker, relevance)] where relevance in ('symbol', 'alias').
    """
    if not text:
        return []
    aliases = aliases or {}
    found: Set[str] = set()
    result: List[Tuple[str, str]] = []

    # Word boundaries for symbols (avoid FPT inside "FPT Corporation" matching substring in word)
    for sym in symbols:
        if not sym:
            continue
        # Match whole-word only
        if re.search(r"\b" + re.escape(sym) + r"\b", text, re.IGNORECASE):
            found.add(sym.upper())
            result.append((sym.upper(), "symbol"))

    for ticker, names in aliases.items():
        if ticker in found:
            continue
        if not isinstance(names, list):
            names = [names] if names else []
        for name in names:
            if not name or not isinstance(name, str):
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
                found.add(ticker.upper())
                result.append((ticker.upper(), "alias"))
                break
    return result


def align_article_to_tickers(
    title: str,
    body: str,
    symbols_path: Optional[str] = None,
) -> List[Tuple[str, Optional[str]]]:
    """
    Combine title + body and return [(ticker, relevance)] for DB storage.
    """
    symbols, aliases = load_symbols_and_aliases(symbols_path)
    combined = (title or "") + "\n" + (body or "")
    pairs = extract_tickers_from_text(combined, symbols, aliases)
    return [(t, r) for t, r in pairs]
