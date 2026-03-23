"""
TechPulse API: AI-powered stock investment advisor.

Endpoints:
- GET  /                         → static web app
- GET  /api/symbols              → VN30 symbols
- POST /api/recommend            → Buy/Hold/Avoid + risk + decision_explanation
- GET  /api/stock/{symbol}       → stock detail (forecast, news, indicators)
- GET  /api/stock/{symbol}/signals → full signals (forecast, risk, news sentiment)
- GET  /api/stock/{symbol}/news  → articles with relevance scoring
- GET  /api/stock/{symbol}/chart → OHLC + indicators for candlestick

Run: uvicorn api:app --reload --host 0.0.0.0
Open: http://localhost:8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent
_logger = logging.getLogger(__name__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.app_services.recommendation import UserProfile, get_risk_advice


def _run_news_pipeline():
    """Streaming-style news pipeline: crawl then process only new articles (run every N min)."""
    start = datetime.now()
    _logger.info("⏳ News pipeline (streaming) started at %s", start.strftime("%H:%M:%S"))
    try:
        from src.news.pipeline import run_pipeline
        results = run_pipeline()
        n = results.get("crawl", 0)
        elapsed = (datetime.now() - start).total_seconds()
        _logger.info("✅ News pipeline done: crawl=%s, steps=%s (%.1fs)", n, results, elapsed)
    except Exception as e:
        _logger.exception("❌ News pipeline failed: %s", e)


def _run_data_pipeline():
    """Chạy pipeline giá VN30: Crawl → Clean → Features (mỗi 4 tiếng). Giữ start từ config, end=hôm nay."""
    start = datetime.now()
    _logger.info(f"⏳ Bắt đầu Data Pipeline VN30 lúc {start.strftime('%H:%M:%S')}")
    try:
        from src.pipeline.vnindex30.fetch_vn30 import run_vn30_pipeline, load_pipeline_config
        cfg = load_pipeline_config()
        end_str = datetime.now().strftime("%d/%m/%Y")
        run_vn30_pipeline(start_date=cfg.get("start_date"), end_date=end_str)
        elapsed = (datetime.now() - start).total_seconds()
        _logger.info(f"✅ Data pipeline xong (Crawl+Clean+Features). (Mất {elapsed:.1f}s)")
    except Exception as e:
        _logger.exception(f"❌ Data pipeline failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: run news pipeline continuously (streaming-style) and price pipeline. Shutdown: stop scheduler."""
    scheduler = None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from src.news.pipeline import load_news_config
        scheduler = BackgroundScheduler()
        # Streaming-style: run news crawl + process every N minutes (config: crawl_interval_minutes).
        news_cfg = load_news_config()
        interval_min = news_cfg.get("crawl_interval_minutes", 15)
        scheduler.add_job(_run_news_pipeline, "interval", minutes=interval_min, id="news_pipeline")
        scheduler.add_job(_run_data_pipeline, "interval", hours=4, id="data_pipeline")
        
        # 3. Daily Green Dragon Market Updater (15:05 GMT+7 Mon-Fri)
        try:
            from apscheduler.triggers.cron import CronTrigger
            import pytz
            from src.pipeline.market_data_updater import run_daily_market_update
            vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
            scheduler.add_job(
                run_daily_market_update,
                trigger=CronTrigger(day_of_week='mon-fri', hour=15, minute=5, timezone=vn_tz),
                id="daily_green_dragon_update"
            )
            _logger.info("Green Dragon Daily Updater scheduled for 15:05 GMT+7 (Mon-Fri)")
        except Exception as e:
            _logger.error(f"Failed to schedule daily market update: {e}")
            
        scheduler.start()
        _logger.info("Schedulers started: news every %s min (streaming), data every 4h", interval_min)
    except ImportError:
        _logger.warning("APScheduler not installed. News pipeline disabled.")
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="TechPulse API",
    description="AI-powered stock advisor: Buy/Hold/Avoid + risk metrics + explainability",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSI", "STB", "TCB", "TPB",
    "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE", "SSB", "PDR",
]


# --- Schemas ---

class RecommendRequest(BaseModel):
    name: str = Field(default="Khách", description="Họ tên hoặc nickname")
    capital: float = Field(ge=0, description="Vốn (VND)")
    years_experience: str = Field(
        description="Kinh nghiệm: '< 1 năm' | '1–3 năm' | '3–5 năm' | '5+ năm'"
    )
    risk_tolerance: str = Field(
        description="Khả năng chấp nhận rủi ro: 'Thấp' | 'Trung bình' | 'Cao'"
    )
    leverage: float = Field(default=0, ge=0, le=1, description="Tỷ lệ vốn vay (0–1)")
    symbol: str = Field(description="Mã cổ phiếu VN30")


class RecommendResponse(BaseModel):
    recommendation: str
    risk_of_loss_pct: float
    risk_of_ruin_pct: float
    expected_return_lower: float
    expected_return_upper: float
    confidence_score: float
    explanation: str
    position_size_suggestion: Optional[float] = None
    signal_layers: Optional[Dict[str, Any]] = None
    signal_breakdown: Optional[Dict[str, Any]] = None
    conclusion: Optional[Dict[str, Any]] = None
    decision_explanation: Optional[Dict[str, Any]] = None
    recommendation_id: Optional[str] = None  # PRODUCTION: traceability
    data_freshness: Optional[Dict[str, Any]] = None  # forecast_as_of, generated_at


# --- Routes ---

@app.get("/api/now")
def get_now():
    """Thời gian server để người dùng kiểm tra ngày có cập nhật đúng không."""
    now = datetime.utcnow()
    return {
        "date": now.strftime("%d/%m/%Y"),
        "datetime": now.isoformat() + "Z",
        "timezone": "UTC",
    }


@app.get("/api/symbols")
def get_symbols():
    return {"symbols": VN30}


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    symbol = req.symbol.strip().upper()
    if symbol not in VN30:
        raise HTTPException(status_code=400, detail=f"Mã không thuộc VN30: {req.symbol}")

    risk_map = {"Thấp": "low", "Trung bình": "medium", "Cao": "high"}
    years_map = {"< 1 năm": 0.5, "1–3 năm": 2, "3–5 năm": 4, "5+ năm": 6}

    profile = UserProfile(
        name=req.name or "Khách",
        capital=float(req.capital),
        years_experience=years_map.get(req.years_experience, 2),
        risk_tolerance=risk_map.get(req.risk_tolerance, "medium"),
        leverage=float(req.leverage or 0),
    )
    try:
        advice = get_risk_advice(profile, symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    de = advice.decision_explanation
    decision_explanation = (
        {
            "primary_signal": de.primary_signal,
            "blocking_factors": de.blocking_factors,
            "supporting_factors": de.supporting_factors,
            "action_summary": de.action_summary,
            "news_analysis": getattr(de, "news_analysis", None),
        }
        if de else None
    )
    return RecommendResponse(
        recommendation=advice.recommendation,
        risk_of_loss_pct=advice.risk_of_loss_pct,
        risk_of_ruin_pct=advice.risk_of_ruin_pct,
        expected_return_lower=advice.expected_return_lower,
        expected_return_upper=advice.expected_return_upper,
        confidence_score=advice.confidence_score,
        explanation=advice.explanation,
        position_size_suggestion=getattr(advice, "position_size_suggestion", None),
        signal_layers=getattr(advice, "signal_layers", None) or {},
        signal_breakdown=advice.signal_breakdown or {},
        conclusion=advice.conclusion or {},
        decision_explanation=decision_explanation,
        recommendation_id=getattr(advice, "recommendation_id", None),
        data_freshness=getattr(advice, "data_freshness", None) or {},
    )




def _trading_days_from(start: datetime, count: int) -> List[str]:
    """Sinh danh sách ngày giao dịch (T2–T6) từ start, count ngày."""
    out = []
    d = start
    while len(out) < count:
        if d.weekday() < 5:  # Mon=0 .. Fri=4
            out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _trading_days_in_month(year: int, month: int) -> List[str]:
    """Sinh danh sách ngày giao dịch trong tháng."""
    out = []
    d = datetime(year, month, 1)
    while d.month == month:
        if d.weekday() < 5:
            out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _expand_forecast_for_horizon(forecast: Any, horizon: str, target_date: Optional[str], target_month: Optional[str]) -> Dict:
    """Mở rộng forecast đơn theo horizon (1d/7d/date/month)."""
    if not forecast:
        return {"mode": horizon, "forecasts": []}
    mean = forecast.ensemble_mean
    std = forecast.ensemble_std
    conf = forecast.confidence_score
    weights = forecast.weights or {}

    now = datetime.utcnow()

    if horizon == "1d":
        next_day = now + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return {
            "mode": "1d",
            "as_of_date": getattr(forecast, "as_of_date", "") or str(now.date()),
            "forecasts": [{"date": next_day.strftime("%Y-%m-%d"), "ensemble_mean": mean, "ensemble_std": std, "confidence_score": conf}],
            "model_weights": weights,
        }

    if horizon == "7d":
        dates = _trading_days_from(now + timedelta(days=1), 7)
        return {
            "mode": "7d",
            "as_of_date": getattr(forecast, "as_of_date", "") or str(now.date()),
            "forecasts": [{"date": d, "ensemble_mean": mean, "ensemble_std": std, "confidence_score": conf} for d in dates],
            "model_weights": weights,
        }

    if horizon == "date" and target_date:
        return {
            "mode": "date",
            "target_date": target_date,
            "as_of_date": getattr(forecast, "as_of_date", "") or str(now.date()),
            "forecasts": [{"date": target_date, "ensemble_mean": mean, "ensemble_std": std, "confidence_score": conf}],
            "model_weights": weights,
        }

    if horizon == "month" and target_month:
        parts = target_month.split("-")
        if len(parts) == 2:
            y, m = int(parts[0]), int(parts[1])
            dates = _trading_days_in_month(y, m)
            return {
                "mode": "month",
                "target_month": target_month,
                "as_of_date": getattr(forecast, "as_of_date", "") or str(now.date()),
                "forecasts": [{"date": d, "ensemble_mean": mean, "ensemble_std": std, "confidence_score": conf} for d in dates],
                "model_weights": weights,
            }

    return {"mode": horizon, "as_of_date": getattr(forecast, "as_of_date", ""), "forecasts": [{"date": str(now.date()), "ensemble_mean": mean, "ensemble_std": std, "confidence_score": conf}], "model_weights": weights or {}}




@app.get("/api/stock/{symbol}")
def get_stock_detail(
    symbol: str,
    horizon: str = "1d",
    target_date: Optional[str] = None,
    target_month: Optional[str] = None,
):
    """Stock detail: forecast, news, indicators. horizon: 1d|7d|date|month."""
    symbol = symbol.strip().upper()
    if symbol not in VN30:
        raise HTTPException(status_code=400, detail="Symbol not in VN30")

    # Forecast from cache
    forecast = None
    try:
        from src.inference.service import get_forecast
        forecast = get_forecast(symbol)
    except Exception:
        pass

    # Mở rộng forecast theo horizon
    forecast_expanded = _expand_forecast_for_horizon(forecast, horizon, target_date, target_month)

    # News
    from src.app_services.recommendation import get_news_sentiment
    sentiment, news_count, articles = get_news_sentiment(symbol)

    # Indicators from MarketDataService
    from src.app_services.market_data import get_indicators
    indicators = get_indicators(symbol, days=90)

    return {
        "symbol": symbol,
        "forecast": forecast_expanded,
        "news": {"sentiment": sentiment, "count": news_count, "articles": articles},
        "indicators": indicators,
    }


@app.get("/api/stock/{symbol}/signals")
def get_stock_signals(symbol: str):
    """Full signals: forecast, risk, news sentiment. For dashboard sections."""
    symbol = symbol.strip().upper()
    if symbol not in VN30:
        raise HTTPException(status_code=400, detail="Symbol not in VN30")
    try:
        from src.app_services.recommendation import UserProfile
        from src.app_services.signal_aggregator import aggregate
        profile = UserProfile(name="", capital=0, years_experience=2, risk_tolerance="medium")
        sig = aggregate(symbol, profile)
        if not sig:
            return {"symbol": symbol, "error": "No signals available"}
        return {
            "symbol": symbol,
            "forecast_mean": sig.forecast_mean,
            "forecast_std": sig.forecast_std,
            "confidence": sig.confidence,
            "volatility_pct": sig.volatility_pct,
            "prob_loss_pct": sig.prob_loss_pct,
            "prob_ruin_pct": sig.prob_ruin_pct,
            "expected_return_lower": sig.expected_return_lower,
            "expected_return_upper": sig.expected_return_upper,
            "news_sentiment": sig.news_sentiment,
            "news_count": sig.news_count,
            "model_weights": sig.model_weights,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/live")
def get_live_news(symbol: str = "", limit: int = 15):
    """
    Scrape real-time headlines from VnExpress / CafeF / TinNhanhCK,
    score with FinBERT, and fire an email alert if aggregate sentiment < -0.40.
    Results are cached for 5 minutes to avoid hammering the sources.
    """
    import time as _time
    cache = _live_news_cache

    # Return cached result if fresh (< 5 min)
    if cache["ts"] and (_time.time() - cache["ts"]) < 300 and cache["data"]:
        items = cache["data"]
    else:
        # Import scraper helpers (same module as alert_system.py)
        import sys, importlib
        _ROOT_scripts = str(Path(__file__).parent / "scripts")
        if _ROOT_scripts not in sys.path:
            sys.path.insert(0, _ROOT_scripts)
        alert_mod = importlib.import_module("alert_system")

        raw = alert_mod.fetch_all_headlines()
        scores, mean = alert_mod.score_headlines(raw)

        items = []
        for h, s in zip(raw, scores):
            sentiment = "positive" if s > 0.15 else ("negative" if s < -0.15 else "neutral")
            items.append({
                "title":     h["title"],
                "url":       h["url"],
                "source":    h["source"],
                "score":     round(s, 3),
                "sentiment": sentiment,
            })

        cache["data"] = items
        cache["ts"]   = _time.time()

        # Fire email alert if market sentiment strongly negative
        if mean < -0.40:
            try:
                negative = sorted([i for i in items if i["sentiment"] == "negative"],
                                   key=lambda x: x["score"])[:5]
                body = f"Aggregate FinBERT sentiment: {mean:.2f}\n\n"
                body += "\n".join(f"[{n['source']}] {n['title']}" for n in negative)
                alert_mod.send_email("[Green Dragon] CANH BAO: TIN TUC TIEU CUC MANH", body)
            except Exception as exc:
                _logger.warning("Alert email failed: %s", exc)

    # Optional symbol filter — keep headlines mentioning the ticker
    sym = symbol.strip().upper()
    if sym and sym in VN30:
        filtered = [i for i in items if sym in i["title"].upper()]
        result = filtered if filtered else items  # fallback to all if no match
    else:
        result = items

    return {"articles": result[:limit], "total": len(result)}


# Cache dict for live news (module-level so it persists between requests)
_live_news_cache: dict = {"ts": None, "data": []}


@app.get("/api/stock/{symbol}/news")
def get_stock_news(symbol: str, limit: int = 10, min_relevance: float = 0.0):
    """News articles with relevance scoring. For dashboard news section."""
    symbol = symbol.strip().upper()
    if symbol not in VN30:
        raise HTTPException(status_code=400, detail="Symbol not in VN30")
    try:
        from src.app_services.news_service import get_articles
        arts = get_articles(symbol, days=30, limit=limit, min_relevance=min_relevance)
        return {
            "symbol": symbol,
            "articles": [
                {
                    "title": a.title,
                    "summary": a.summary,
                    "investment_summary": a.investment_summary,
                    "url": a.url,
                    "source": a.source,
                    "sentiment": a.sentiment,
                    "relevance_score": a.relevance_score,
                    "published_at": a.published_at,
                }
                for a in arts
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{symbol}/news/intelligence")
def get_stock_news_intelligence(
    symbol: str,
    days: int = 450,
    min_relevance: float = 0.2,
    limit: int = 10,
    event_type: Optional[str] = None,
):
    """
    News intelligence: aggregated signal + enriched articles.
    event_type: optional filter (earnings, legal, macro, operations, guidance, ma, dividend, other).
    """
    symbol = symbol.strip().upper()
    if symbol not in VN30:
        raise HTTPException(status_code=400, detail="Symbol not in VN30")
    try:
        from src.app_services.news_intelligence import get_stock_news_signal
        signal = get_stock_news_signal(
            symbol, days=days, min_relevance=min_relevance,
            limit_articles=limit, event_type_filter=event_type,
        )
        # Build response with reasoning and per-article breakdown (url, source, publish time, relevance, horizon, sentiment).
        payload = {
            "symbol": signal.symbol,
            "is_general_fallback": getattr(signal, "is_general_fallback", False),
            "signal": {
                "composite_score": signal.composite_score,
                "article_count": signal.article_count,
                "avg_sentiment": signal.avg_sentiment,
                "avg_relevance": signal.avg_relevance,
                "horizon_breakdown": signal.horizon_breakdown,
                "event_breakdown": signal.event_breakdown,
                "net_impact_label": signal.net_impact_label,
                "net_impact_confidence": signal.net_impact_confidence,
            },
            "reasoning": getattr(signal, "reasoning", "") or "",
            "top_contributors": getattr(signal, "top_contributors", []) or [],
            "market_shock": None,
            "top_3_impact": [
                {
                    "title": i.title,
                    "why_it_matters": i.why_it_matters,
                    "impact_direction": i.impact_direction,
                    "time_horizon": i.time_horizon,
                    "confidence": i.confidence,
                    "url": i.url,
                    "event_type": i.event_type,
                }
                for i in signal.top_3_impact
            ],
            "articles": [
                {
                    "article_id": a.article_id,
                    "title": a.title,
                    "summary": a.summary,
                    "url": a.url,
                    "source": a.source,
                    "published_at": a.published_at,
                    "event_type": a.event_type,
                    "ticker_relevance": a.ticker_relevance,
                    "sentiment_score": a.sentiment_score,
                    "sentiment_confidence": a.sentiment_confidence,
                    "impact_horizon": a.impact_horizon,
                    "horizon_weight": getattr(a, "horizon_weight", 1.0),
                    "contribution_weight": getattr(a, "contribution_weight", 0.0),
                    "raw_contribution": getattr(a, "raw_contribution", 0.0),
                }
                for a in signal.top_articles
            ],
        }
        if getattr(signal, "market_shock", None) and signal.market_shock.is_shock:
            payload["market_shock"] = {
                "is_shock": True,
                "reason": signal.market_shock.reason,
                "summary": signal.market_shock.summary,
                "contributing_article_titles": signal.market_shock.contributing_article_titles,
            }
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{symbol}/news/daily")
def get_stock_news_daily(
    symbol: str,
    from_date: str,
    to_date: str,
    min_relevance: float = 0.2,
):
    """
    Per-ticker daily news signals with explainable per-article contribution.
    from_date, to_date: YYYY-MM-DD (inclusive).
    """
    symbol = symbol.strip().upper()
    if symbol not in VN30:
        raise HTTPException(status_code=400, detail="Symbol not in VN30")
    try:
        from src.app_services.news_intelligence import get_ticker_daily_signals
        daily = get_ticker_daily_signals(symbol, from_date=from_date, to_date=to_date, min_relevance=min_relevance)
        return {
            "symbol": symbol,
            "from_date": from_date,
            "to_date": to_date,
            "daily_signals": [
                {
                    "date": d.date,
                    "composite_score": d.composite_score,
                    "article_count": d.article_count,
                    "articles": [
                        {
                            "article_id": a.article_id,
                            "title": a.title,
                            "url": a.url,
                            "source": a.source,
                            "published_at": a.published_at,
                            "ticker_relevance": a.ticker_relevance,
                            "impact_horizon": a.impact_horizon,
                            "horizon_weight": a.horizon_weight,
                            "sentiment_score": a.sentiment_score,
                            "sentiment_confidence": a.sentiment_confidence,
                            "contribution_weight": a.contribution_weight,
                            "raw_contribution": a.raw_contribution,
                        }
                        for a in d.articles
                    ],
                }
                for d in daily
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{symbol}/news/signal")
def get_stock_news_signal_api(symbol: str, days: int = 30, min_relevance: float = 0.2):
    """Lightweight: aggregated news signal only (no article list)."""
    symbol = symbol.strip().upper()
    if symbol not in VN30:
        raise HTTPException(status_code=400, detail="Symbol not in VN30")
    try:
        from src.app_services.news_intelligence import get_stock_news_signal
        signal = get_stock_news_signal(symbol, days=days, min_relevance=min_relevance, limit_articles=0)
        return {
            "symbol": signal.symbol,
            "composite_score": signal.composite_score,
            "article_count": signal.article_count,
            "avg_sentiment": signal.avg_sentiment,
            "avg_relevance": signal.avg_relevance,
            "horizon_breakdown": signal.horizon_breakdown,
            "event_breakdown": signal.event_breakdown,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{symbol}/chart")
def get_stock_chart(symbol: str, days: int = 90, end_date: Optional[str] = None):
    """OHLC + MA cho biểu đồ. end_date: YYYY-MM-DD để lọc đến ngày đó (theo lựa chọn dự báo)."""
    symbol = symbol.strip().upper()
    if symbol not in VN30:
        raise HTTPException(status_code=400, detail="Symbol not in VN30")

    from src.app_services.market_data import get_chart_data
    data = get_chart_data(symbol, days=int(days), end_date=end_date)
    return {"symbol": symbol, **data}


@app.get("/api/v1/chart-data/{ticker}")
def get_chart_data_v1(ticker: str, days: int = 300):
    """
    Unified Endpoint for Green Dragon Next.js TradingView Clone.
    Returns OHLCV, SMC Heuristic Markers, and LSTM Action Scores for the UI overlay.
    """
    import pandas as pd
    import numpy as np
    import torch
    from pathlib import Path

    ticker = ticker.strip().upper()
    if ticker not in VN30:
        raise HTTPException(status_code=400, detail="Ticker not in VN30")

    csv_path = Path(f"data/raw/{ticker}.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Historical data not found")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.sort_values("date").tail(days).reset_index(drop=True)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="Not enough data points")

    # 1. SMC Markers (Heuristic visual extraction)
    try:
        from src.features.smc_visual_utils import detect_heuristic_smc_markers
        # Convert date to string format for UI JSON serialization
        df_smc = df.copy()
        df_smc['date'] = df_smc['date'].dt.strftime("%Y-%m-%d") 
        smc_markers = detect_heuristic_smc_markers(df_smc, window=5)
    except Exception as e:
        _logger.error(f"SMC marker extraction error: {e}")
        smc_markers = {"bos": [], "choch": [], "order_blocks": []}

    # 2. LSTM Action Scores with correct 7-feature / 20-step window inference
    try:
        from src.models.lstm import LSTMModel
        from sklearn.preprocessing import RobustScaler

        LSTM_INPUT_SIZE = 7   # open, high, low, close, volume, ls_binary, ls_strength
        WINDOW_SIZE_INF = 20  # must match training window

        model = LSTMModel(input_size=LSTM_INPUT_SIZE, hidden_size=64, num_layers=2)
        model_path = Path("models/best_lstm_model.pt")
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()

        # --- Merge SMC features (ls_binary, ls_strength) into df ---
        smc_path = Path("data/processed/smc_features.csv")
        if smc_path.exists():
            smc_df = pd.read_csv(smc_path)
            smc_df['date'] = pd.to_datetime(smc_df['date']).dt.normalize()
            smc_t = smc_df[smc_df['symbol'] == ticker][['date', 'ls_binary', 'ls_strength']]
            smc_t = smc_t.drop_duplicates(subset=['date'], keep='last')
            df = pd.merge(df, smc_t, on='date', how='left')
            df['ls_binary']   = df['ls_binary'].fillna(0)
            df['ls_strength'] = df['ls_strength'].fillna(0)
        else:
            df['ls_binary']   = 0.0
            df['ls_strength'] = 0.0

        # --- Normalise exactly like training (pct_change + RobustScaler) ---
        inf_df = df.copy()
        inf_df['open']   = inf_df['open'].pct_change()
        inf_df['high']   = inf_df['high'].pct_change()
        inf_df['low']    = inf_df['low'].pct_change()
        inf_df['close']  = inf_df['close'].pct_change()
        inf_df['volume'] = np.log1p(inf_df['volume']).diff()
        inf_df.dropna(inplace=True)
        scaler = RobustScaler()
        inf_df[['open','high','low','close','volume']] = scaler.fit_transform(
            inf_df[['open','high','low','close','volume']])

        FEAT_COLS_INF = ['open','high','low','close','volume','ls_binary','ls_strength']
        feat_matrix = inf_df[FEAT_COLS_INF].values.astype(np.float32)
        date_vals   = inf_df['date'].dt.strftime('%Y-%m-%d').values
        close_vals  = inf_df['close'].values

        # Map model scores back to original df dates for response
        score_map: dict = {}
        if len(feat_matrix) > WINDOW_SIZE_INF:
            with torch.no_grad():
                for wi in range(WINDOW_SIZE_INF, len(feat_matrix)):
                    window = feat_matrix[wi - WINDOW_SIZE_INF : wi]
                    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                    score = float(model(x).item())
                    score_map[date_vals[wi]] = score

    except Exception as e:
        _logger.error(f"LSTM 7-feat inference error: {e}")
        model = None
        score_map = {}

    action_signals = []
    THRESHOLD = 0.635  # Optuna-discovered Sharpe-optimal threshold
    ohlcv = []

    holding = False
    entry_price = 0.0
    target_price = 0.0
    stop_loss_price = 0.0
    holding_days = 0

    df = df.dropna(subset=["date"])

    for i, row in df.iterrows():
        if pd.isna(row["date"]):
            continue
        dt_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        
        c_open = float(row["open"])
        c_high = float(row["high"])
        c_low = float(row["low"])
        c_close = float(row["close"])
        c_volume = float(row.get("volume", 0))

        ohlcv.append({
            "time":   dt_str,
            "open":   c_open,
            "high":   c_high,
            "low":    c_low,
            "close":  c_close,
            "volume": c_volume
        })

        if holding:
            holding_days += 1
            if holding_days >= 2:
                if c_high >= target_price:
                    action_signals.append({
                        "time": dt_str,
                        "price": target_price,
                        "type": "SELL",
                        "reason": "TP"
                    })
                    holding = False
                elif c_low <= stop_loss_price:
                    action_signals.append({
                        "time": dt_str,
                        "price": stop_loss_price,
                        "type": "SELL",
                        "reason": "SL"
                    })
                    holding = False
                elif holding_days >= 5:
                    action_signals.append({
                        "time": dt_str,
                        "price": c_close,
                        "type": "SELL",
                        "reason": "TE"
                    })
                    holding = False
        else:
            base_score = score_map.get(dt_str, 0.0)
            if base_score > THRESHOLD:
                entry_price = c_close
                target_price = entry_price * 1.03
                stop_loss_price = entry_price * 0.98
                
                action_signals.append({
                    "time": dt_str,
                    "score": round(base_score, 3),
                    "price": entry_price,
                    "type": "BUY"
                })
                
                holding = True
                holding_days = 0
                

    # Cast SMC markers to standard floats/ints as well to fix JSON serialization
    for m in smc_markers["bos"] + smc_markers["choch"]:
        m["price"] = float(m["price"])
        m["start_idx"] = int(m["start_idx"])
        m["end_idx"] = int(m["end_idx"])
        
    for m in smc_markers["order_blocks"]:
        m["top"] = float(m["top"])
        m["bottom"] = float(m["bottom"])

    return {
        "ticker": ticker,
        "ohlcv": ohlcv,
        "smc": smc_markers,
        "action_signals": action_signals,
        "threshold": float(THRESHOLD)
    }

from typing import Optional

class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatContext(BaseModel):
    ticker: str
    date: str
    price: float
    score: float
    smc_context: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: Optional[ChatContext] = None

@app.post("/api/v1/chat")
def chat_endpoint(req: ChatRequest):
    """
    Real AI Chatbot endpoint powered by Gemini.
    Responds to user inquiries and explains specific chart contexts.
    """
    import os
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv() # Load from .env if present
    except ImportError as e:
        return {"reply": f"**<Green Dragon SYSTEM ALERT>**\n\nI detect the Gemini API Key, but my backend server needs a restart to load the newly installed AI modules.\n\n**Action Required:** Please go to the terminal running `uvicorn api:app`, press `Ctrl+C` to stop it, and run the command again. ({e})"}
    
    context = req.context
    messages = req.messages
    last_user_message = messages[-1].content if messages else ""
    
    # 1. Setup Gemini Prompt
    system_instruction = (
        "You are <Green Dragon Quant AI>, an elite quantitative trading assistant. "
        "Your role is to explain institutional footprints, Liquidity Sweeps, and SMC (Smart Money Concepts). "
        "Be extremely professional, concise, and analytical. Use markdown for emphasis."
    )
    
    prompt = f"System Context: {system_instruction}\n\n"
    
    if context:
        confidence_level = "High" if context.score > 0.75 else "Moderate"
        prompt += (
            f"--- CURRENT CHART CONTEXT ---\n"
            f"Ticker: {context.ticker}\n"
            f"Date of Signal: {context.date}\n"
            f"Price at Signal: {context.price} VND\n"
            f"LSTM Action Score: {context.score} (Threshold: 0.635, Confidence: {confidence_level})\n"
            f"SMC Context Detected: {context.smc_context}\n"
            f"-----------------------------\n\n"
        )
    
    prompt += "--- CONVERSATION HISTORY ---\n"
    for msg in messages[:-1]:
        prompt += f"{msg.role.upper()}: {msg.content}\n"
    
    prompt += f"USER: {last_user_message}\n"
    prompt += "GREEN DRAGON AI:\n"
    
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    # 2. Try calling Gemini
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return {"reply": response.text}
        except Exception as e:
            _logger.error(f"Gemini API Error: {e}")
            # Fall through to mock logic if Gemini fails
    
    # 3. Fallback Mock Logic (If API key missing or request fails)
    if context:
        confidence_level = "High" if context.score > 0.75 else "Moderate"
        reply = (f"**<Green Dragon Quant AI>** Analysis for **{context.ticker}** Execution on **{context.date}**\n\n"
                 f"*(Note: Running in offline simulation mode. GEMINI_API_KEY not provided.)*\n\n"
                 f"**Execution Price:** `{context.price:,.0f} VND`\n"
                 f"**LSTM Action Score:** `{context.score}` (Threshold: 0.635)\n"
                 f"**Confidence:** `{confidence_level}`\n\n"
                 f"**Strategic Reasoning:**\n"
                 f"The execution algorithm triggered a **BUY** at this specific juncture because the feature matrix detected a systemic market irregularity perfectly aligning with our tail-risk model.\n\n"
                 f"**Market Context Interception:**\n"
                 f"> *\"{context.smc_context}\"*\n\n"
                 f"My deep learning architecture identifies this as institutional footprinting. By combining the profound Liquidity Sweep signature with the mathematically defined {confidence_level} structural anomaly, the Green Dragon system enters ahead of the retail herd to capture the incoming impulsive reversal.")
        return {"reply": reply}

    # Standard Q&A Fallback
    if "liquidity" in last_user_message.lower():
        reply = "Liquidity points are price levels where retail traders place their stop-loss orders. Institutional algorithms intentionally hunt these levels ('Liquidity Sweeps') to accumulate large positions without causing massive price slippage. Our Green Dragon LSTM is specifically trained to detect these anomalies in OHLCV tick data."
    elif "smc" in last_user_message.lower() or "smart money" in last_user_message.lower():
        reply = "Smart Money Concepts (SMC) track the footprints of large institutional players. Key markers include Break of Structure (BOS) defining trend continuation, Change of Character (CHoCH) preceding reversals, and Order Blocks (OB) where major accumulation occurs. I overlay these on the chart heuristically."
    else:
        reply = "I am the Green Dragon Financial Copilot. Please set `GEMINI_API_KEY` in the `.env` file to enable my full AI reasoning capabilities. Alternatively, click on a specific Green BUY Arrow on the chart to receive a fallback context-aware explanation."

    return {"reply": reply}

# --- Static frontend ---

WEB_DIR = _ROOT / "frontend" / "out"
if not WEB_DIR.exists():
    WEB_DIR = _ROOT / "web"   # fallback for old builds

if WEB_DIR.exists():
    # Serve Next.js static assets at the correct /_next/static/ path
    app.mount("/_next/static", StaticFiles(directory=WEB_DIR / "_next" / "static"), name="nextjs_static")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    # Catch-all: serve any non-API path as index.html (SPA routing)
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        page = WEB_DIR / full_path
        if page.exists() and page.is_file():
            return FileResponse(page)
        return FileResponse(WEB_DIR / "index.html")
else:
    @app.get("/")
    def index():
        return {"message": "Frontend chưa build. Chạy: cd frontend && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
