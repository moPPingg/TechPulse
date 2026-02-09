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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.app_services.recommendation import UserProfile, get_risk_advice


def _run_news_crawl():
    """Chạy bước crawl tin tức (mỗi 2 tiếng)."""
    try:
        from src.news.pipeline import load_news_config, run_crawl
        from src.news.db import get_engine, init_db
        cfg = load_news_config()
        db_path = cfg.get("database", {}).get("path", "data/news/news.db")
        if not Path(db_path).is_absolute():
            db_path = str(_ROOT / db_path)
        init_db(db_path)
        conn = get_engine(db_path)
        n = run_crawl(conn, cfg)
        _logger.info("News crawl (scheduled): %d articles processed", n)
    except Exception as e:
        _logger.warning("News crawl failed: %s", e)


def _run_data_pipeline():
    """Chạy pipeline giá VN30: Crawl → Clean → Features (mỗi 4 tiếng). Giữ start từ config, end=hôm nay."""
    try:
        from src.pipeline.vnindex30.fetch_vn30 import run_vn30_pipeline, load_pipeline_config
        cfg = load_pipeline_config()
        end_str = datetime.now().strftime("%d/%m/%Y")
        run_vn30_pipeline(start_date=cfg.get("start_date"), end_date=end_str)
        _logger.info("Data pipeline (scheduled): Crawl+Clean+Features completed")
    except Exception as e:
        _logger.warning("Data pipeline failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: bật scheduler cào tin + cào giá. Shutdown: tắt scheduler."""
    scheduler = None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(_run_news_crawl, "interval", hours=2, id="news_crawl")
        scheduler.add_job(_run_data_pipeline, "interval", hours=4, id="data_pipeline")
        scheduler.start()
        _logger.info("Schedulers started: news every 2h, data (price) every 4h")
    except ImportError:
        _logger.warning("APScheduler not installed. News crawl every 2h disabled.")
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="TechPulse API",
    description="AI-powered stock advisor: Buy/Hold/Avoid + risk metrics + explainability",
    lifespan=lifespan,
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
        return {
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
                }
                for a in signal.top_articles
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


# --- Static frontend ---

WEB_DIR = _ROOT / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")
else:
    @app.get("/")
    def index():
        return {"message": "Chưa có thư mục web/. Tạo web/index.html và web/static/ rồi chạy lại."}
