"""
Green Dragon — Event-Driven Smart Alert System (Email / Gmail)
---------------------------------------------------------------
Monitors three signal sources and fires a Gmail notification when any fires:

    Event A — Liquidity Sweep detected (SMC condition met on latest candle)
    Event B — Market enters Regime 3 (Extreme volatility)
    Event C — Breaking financial news from VnExpress / CafeF / ChungKhoanVN
              with strongly negative FinBERT sentiment score

Setup:
    1. Enable "App Passwords" in your Google Account:
       myaccount.google.com -> Security -> 2-Step Verification -> App Passwords
    2. Generate an App Password for "Mail" -> copy the 16-char code
    3. Set env vars:
         GMAIL_USER=your@gmail.com
         GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
         ALERT_TO_EMAIL=recipient@gmail.com   (can be the same as GMAIL_USER)
    4. python scripts/alert_system.py

Dependencies: requests, beautifulsoup4, pandas, numpy
    pip install requests beautifulsoup4
"""

import os
import smtplib
import logging
import time
import hashlib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config — read from environment variables
# ---------------------------------------------------------------------------
GMAIL_USER         = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
ALERT_TO_EMAIL     = os.environ.get("ALERT_TO_EMAIL", GMAIL_USER)

DATA_DIR            = Path(os.environ.get("DATA_DIR", "data/raw"))
CHECK_INTERVAL_S    = int(os.environ.get("CHECK_INTERVAL_S", "300"))    # 5 min
NEWS_INTERVAL_S     = int(os.environ.get("NEWS_INTERVAL_S", "1800"))    # 30 min
VOL_WINDOW          = int(os.environ.get("VOL_WINDOW", "252"))
VOL_EXTREME_THRESH  = float(os.environ.get("VOL_EXTREME_THRESH", "0.30"))
SWEEP_K             = float(os.environ.get("SWEEP_K", "1.5"))
NEWS_SENTIMENT_THRESH = float(os.environ.get("NEWS_SENTIMENT_THRESH", "-0.40"))

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSI", "STB", "TCB", "TPB",
    "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE", "SSB", "PDR",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Track seen headlines to avoid duplicate alerts
_seen_headlines: set = set()

# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------
def send_email(subject: str, body: str) -> bool:
    """Send a plain-text email via Gmail SMTP (TLS, port 587)."""
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        log.warning("Gmail credentials not set — printing alert locally.")
        print(f"\n[ALERT] {subject}\n{body}\n")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = ALERT_TO_EMAIL
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, ALERT_TO_EMAIL, msg.as_string())
        log.info("Email sent: %s", subject)
        return True
    except smtplib.SMTPException as exc:
        log.error("Email send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# News scrapers — VnExpress / CafeF / ChungKhoanVN
# ---------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}
_TIMEOUT = 10  # seconds per request


def _get(url: str) -> Optional[BeautifulSoup]:
    """Fetch URL and return BeautifulSoup, or None on failure.
    Falls back to verify=False for sites with self-signed certificates."""
    for verify in (True, False):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, verify=verify)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            return BeautifulSoup(resp.text, "html.parser")
        except requests.exceptions.SSLError:
            if not verify:
                log.warning("SSL verify failed (even with verify=False) [%s]", url)
                return None
            # retry without SSL verification
            continue
        except Exception as exc:
            log.warning("Fetch failed [%s]: %s", url, exc)
            return None
    return None


def scrape_vnexpress() -> list[dict]:
    """
    Scrape finance + stock market headlines from VnExpress.
    Returns list of {"title": str, "url": str, "source": "VnExpress"}.
    """
    headlines = []
    urls = [
        "https://vnexpress.net/kinh-doanh/chung-khoan",
        "https://vnexpress.net/kinh-doanh",
    ]
    for url in urls:
        soup = _get(url)
        if soup is None:
            continue
        # VnExpress article titles are in <h3 class="title-news"> or <p class="description">
        for tag in soup.select("h3.title-news a, h2.title-news a"):
            title = tag.get_text(strip=True)
            href  = tag.get("href", "")
            if title and href:
                headlines.append({"title": title, "url": href, "source": "VnExpress"})
        if len(headlines) >= 20:
            break
    return headlines[:20]


def scrape_cafef() -> list[dict]:
    """
    Scrape stock market news from CafeF.
    Returns list of {"title": str, "url": str, "source": "CafeF"}.
    """
    headlines = []
    soup = _get("https://cafef.vn/thi-truong-chung-khoan.chn")
    if soup is None:
        return headlines
    # CafeF uses <h3> and <h2> with <a> inside for article titles
    for tag in soup.select("h3 a, h2 a, .tlitem h3 a"):
        title = tag.get_text(strip=True)
        href  = tag.get("href", "")
        if not title:
            continue
        if not href.startswith("http"):
            href = "https://cafef.vn" + href
        headlines.append({"title": title, "url": href, "source": "CafeF"})
        if len(headlines) >= 20:
            break
    return headlines[:20]


def scrape_tinnhanh() -> list[dict]:
    """
    Scrape stock market news from TinNhanhChungKhoan.vn.
    Returns list of {"title": str, "url": str, "source": "TinNhanhCK"}.
    """
    headlines = []
    soup = _get("https://tinnhanhchungkhoan.vn/")
    if soup is None:
        return headlines
    for tag in soup.select("h3 a, h2 a, .article-title a, .title a"):
        title = tag.get_text(strip=True)
        href  = tag.get("href", "")
        if not title:
            continue
        if not href.startswith("http"):
            href = "https://tinnhanhchungkhoan.vn" + href
        headlines.append({"title": title, "url": href, "source": "TinNhanhCK"})
        if len(headlines) >= 20:
            break
    return headlines[:20]


def fetch_all_headlines() -> list[dict]:
    """
    Aggregate headlines from all three sources. Deduplicate by title hash.
    Returns list of {"title", "url", "source"} dicts.
    """
    all_items = []
    for scraper in [scrape_vnexpress, scrape_cafef, scrape_tinnhanh]:
        try:
            items = scraper()
            all_items.extend(items)
            log.info("%s: fetched %d headlines", scraper.__name__, len(items))
        except Exception as exc:
            log.error("%s failed: %s", scraper.__name__, exc)

    # Deduplicate by normalised title
    seen, unique = set(), []
    for item in all_items:
        key = item["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# Sentiment scoring (lazy-load FinBERT to avoid slowing startup)
# ---------------------------------------------------------------------------
_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            log.info("Loading FinBERT for sentiment scoring...")
            _finbert_pipeline = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,
            )
            log.info("FinBERT loaded.")
        except Exception as exc:
            log.warning("FinBERT unavailable (%s) — sentiment scoring disabled.", exc)
            _finbert_pipeline = False  # mark as unavailable
    return _finbert_pipeline if _finbert_pipeline else None


_LABEL_SCORE = {
    "positive": 1.0, "Positive": 1.0,
    "neutral":  0.0, "Neutral":  0.0,
    "negative":-1.0, "Negative":-1.0,
}


def score_headlines(headlines: list[dict]) -> tuple[list[float], float]:
    """
    Score a list of headline dicts with FinBERT.
    Returns (per_item_scores, aggregate_mean). Falls back to 0.0 if FinBERT unavailable.
    """
    pipe = _get_finbert()
    if pipe is None:
        return [0.0] * len(headlines), 0.0

    texts  = [h["title"] for h in headlines]
    scores = []
    for text in texts:
        try:
            result     = pipe(text, truncation=True, max_length=512)[0]
            direction  = _LABEL_SCORE.get(result["label"], 0.0)
            scores.append(direction * result["score"])
        except Exception:
            scores.append(0.0)

    mean = float(np.mean(scores)) if scores else 0.0
    return scores, mean


# ---------------------------------------------------------------------------
# Price signal detectors
# ---------------------------------------------------------------------------
def _load_csv(symbol: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    Returns True if the latest candle satisfies the three-condition bullish
    liquidity sweep definition used in the paper:
        (1) low_t  < min(low_{t-lookback : t-1})
        (2) close_t > support_level
        (3) volume_t > k * rolling_mean_volume_t
    """
    if len(df) < lookback + 1:
        return False
    latest  = df.iloc[-1]
    history = df.iloc[-(lookback + 1):-1]
    support   = history["low"].min()
    vol_mean  = history["volume"].mean()
    pierce    = latest["low"]    < support
    recovery  = latest["close"]  > support
    vol_spike = latest["volume"] > SWEEP_K * vol_mean
    return bool(pierce and recovery and vol_spike)


def compute_realized_vol(df: pd.DataFrame, window: int = VOL_WINDOW) -> float:
    """Annualised realised volatility from daily log-returns."""
    if len(df) < 2:
        return 0.0
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    return float(log_ret.tail(window).std() * np.sqrt(252))


def classify_regime(vol: float) -> int:
    """Map annualised vol to regime id (1 / 2 / 3)."""
    if vol >= VOL_EXTREME_THRESH:
        return 3
    elif vol >= 0.15:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Main check functions
# ---------------------------------------------------------------------------
def check_price_signals() -> None:
    """Check all VN30 symbols for Liquidity Sweep and Extreme Regime."""
    sweep_events = []
    regime_vols  = []

    for symbol in VN30:
        df = _load_csv(symbol)
        if df is None or len(df) < 30:
            continue
        if detect_liquidity_sweep(df):
            price = df.iloc[-1]["close"]
            date  = df.iloc[-1]["date"].strftime("%Y-%m-%d")
            sweep_events.append((symbol, price, date))
            log.info("Liquidity Sweep: %s @ %s (%.2f)", symbol, date, price)
        regime_vols.append(compute_realized_vol(df))

    # Regime 3 alert
    if regime_vols:
        market_vol = float(np.median(regime_vols))
        if classify_regime(market_vol) == 3:
            subject = "[Green Dragon] CANH BAO: THI TRUONG EXTREME REGIME"
            body = (
                f"Thoi gian : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Che do    : Regime 3 - Extreme\n"
                f"Bien dong : {market_vol:.1%} (nguong >= 30%)\n\n"
                f"De xuat hanh dong:\n"
                f"  - That chat stop-loss cho cac vi the dang mo\n"
                f"  - Giam ty le position sizing\n"
                f"  - Khong mo them vi the long moi cho den khi regime binh thuong hoa\n\n"
                f"[Green Dragon Alert System]"
            )
            send_email(subject, body)

    # Liquidity Sweep alerts
    for symbol, price, date in sweep_events:
        subject = f"[Green Dragon] LIQUIDITY SWEEP — {symbol}"
        body = (
            f"Ma co phieu : {symbol}\n"
            f"Ngay       : {date}\n"
            f"Gia dong cua: {price:,.0f} VND\n\n"
            f"He thong phat hien Bullish Liquidity Sweep tren nen nut cuoi cung.\n"
            f"Dieu kien:\n"
            f"  [1] Wick xuyen qua nen ho tro cau truc (20-ngay swing low)\n"
            f"  [2] Gia dong cua phuc hoi len tren nen ho tro\n"
            f"  [3] Volume bat thuong (k={SWEEP_K}x trung binh 20 ngay)\n\n"
            f"De xuat: Cho Action Score LSTM > 0.635 truoc khi vao lenh.\n\n"
            f"[Green Dragon Alert System]"
        )
        send_email(subject, body)


def check_news_sentiment() -> None:
    """
    Fetch headlines from VnExpress / CafeF / ChungKhoanVN,
    score with FinBERT, and alert if aggregate sentiment is extremely negative.
    Also sends a digest email with the top headlines regardless of sentiment.
    """
    global _seen_headlines

    all_headlines = fetch_all_headlines()
    if not all_headlines:
        log.warning("No headlines fetched from any source.")
        return

    # Filter out already-seen headlines
    new_headlines = []
    for h in all_headlines:
        key = hashlib.md5(h["title"].encode()).hexdigest()
        if key not in _seen_headlines:
            _seen_headlines.add(key)
            new_headlines.append(h)

    if not new_headlines:
        log.info("No new headlines since last check.")
        return

    log.info("%d new headlines to process.", len(new_headlines))

    # Score sentiment
    scores, mean_score = score_headlines(new_headlines)

    # Build headline digest lines
    digest_lines = []
    for h, s in zip(new_headlines[:15], scores[:15]):
        sign = "+" if s > 0.1 else ("-" if s < -0.1 else " ")
        digest_lines.append(
            f"  [{sign}{abs(s):.2f}] [{h['source']:<14}] {h['title'][:80]}"
        )

    digest_text = "\n".join(digest_lines)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Always send a digest email with fresh headlines
    subject_digest = f"[Green Dragon] Tin tuc thi truong — {timestamp}"
    body_digest = (
        f"Cap nhat tin tuc chung khoan VN30\n"
        f"Thoi gian        : {timestamp}\n"
        f"So bai bao moi   : {len(new_headlines)}\n"
        f"Sentiment trung binh: {mean_score:+.3f}  "
        f"({'TICH CUC' if mean_score > 0.1 else 'TIEU CUC' if mean_score < -0.1 else 'TRUNG TINH'})\n"
        f"\nDANH SACH TIN TUC MOI NHAT:\n"
        f"{'-' * 70}\n"
        f"{digest_text}\n"
        f"{'-' * 70}\n\n"
        f"NGUON: VnExpress | CafeF | ChungKhoan.vn\n\n"
        f"[+] = tich cuc  [-] = tieu cuc  [ ] = trung tinh\n"
        f"[Green Dragon Alert System]"
    )
    send_email(subject_digest, body_digest)

    # Extra alert if sentiment is extremely negative
    if mean_score < NEWS_SENTIMENT_THRESH:
        negative_items = [
            (h, s) for h, s in zip(new_headlines, scores) if s < -0.3
        ]
        neg_lines = "\n".join(
            f"  [{s:+.3f}] {h['title'][:90]}\n          {h['url']}"
            for h, s in negative_items[:8]
        )
        subject_alert = f"[Green Dragon] CANH BAO: TIN TUC TIEU CUC MANH — Sentiment {mean_score:+.2f}"
        body_alert = (
            f"CANH BAO SENTIMENT TIEU CUC\n"
            f"Thoi gian              : {timestamp}\n"
            f"Sentiment trung binh   : {mean_score:+.3f} (nguong < {NEWS_SENTIMENT_THRESH})\n"
            f"So tin tieu cuc manh   : {len(negative_items)}/{len(new_headlines)}\n\n"
            f"CAC TIN TIEU CUC NOI BAT:\n"
            f"{'-' * 70}\n"
            f"{neg_lines}\n"
            f"{'-' * 70}\n\n"
            f"De xuat: Kiem tra lai portfolio, xem xet giam exposure.\n\n"
            f"[Green Dragon Alert System]"
        )
        send_email(subject_alert, body_alert)
        log.warning("Extreme negative sentiment alert sent (score=%.3f).", mean_score)


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------
def run_monitor() -> None:
    log.info("Alert system started.")
    log.info("Price check interval : %ds", CHECK_INTERVAL_S)
    log.info("News check interval  : %ds", NEWS_INTERVAL_S)
    log.info("Monitoring %d symbols in %s", len(VN30), DATA_DIR)

    if not GMAIL_USER:
        log.warning(
            "GMAIL_USER not set — alerts will print to stdout only.\n"
            "Set: GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_TO_EMAIL"
        )

    last_news_check = 0.0

    while True:
        now = time.time()

        # Price signals — every CHECK_INTERVAL_S
        try:
            check_price_signals()
        except Exception as exc:
            log.error("check_price_signals error: %s", exc, exc_info=True)

        # News digest — every NEWS_INTERVAL_S (default 30 min)
        if now - last_news_check >= NEWS_INTERVAL_S:
            try:
                check_news_sentiment()
                last_news_check = time.time()
            except Exception as exc:
                log.error("check_news_sentiment error: %s", exc, exc_info=True)

        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    run_monitor()
