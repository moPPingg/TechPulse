"""
Recommendation Engine: Probabilistic Buy/Hold/Avoid with explainability.

TEACHING:
- Combines: ML ensemble forecast + Risk Engine + News sentiment + User profile.
- Decision is probabilistic: we compute "expected utility" adjusted for risk tolerance.
- All logic is traceable; explanation chain explains why each signal matters.

PRODUCTION: See docs/PRODUCTION_REVIEW_AND_REDESIGN.md for architecture,
leakage controls, and metric disclaimers.

Inputs:  UserProfile (capital, experience, leverage, risk_tolerance), symbol
Outputs: RiskAdvice (recommendation, risk metrics, expected return CI, confidence, explanation)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class UserProfile:
    """User context for personalized recommendation."""
    name: str
    capital: float  # VND
    years_experience: float
    risk_tolerance: str  # "low" | "medium" | "high"
    leverage: float = 0.0  # Borrowed money as fraction of capital (0 = no leverage)


@dataclass
class DecisionExplanation:
    """Structured decision explanation for fintech UI."""
    primary_signal: str
    blocking_factors: List[str]
    supporting_factors: List[str]
    action_summary: str
    news_analysis: Optional[str] = None  # Phân tích tin tức bởi AI/model (khi có)


@dataclass
class RiskAdvice:
    """Full recommendation output."""
    recommendation: str  # "Buy" | "Hold" | "Avoid"
    risk_of_loss_pct: float
    risk_of_ruin_pct: float
    expected_return_lower: float  # % CI lower
    expected_return_upper: float  # % CI upper
    confidence_score: float  # 0-1; model confidence (not calibrated accuracy)
    explanation: str
    position_size_suggestion: float = 0.0  # 0-0.5 fraction of capital
    signal_layers: Dict[str, Any] = field(default_factory=dict)  # 4-layer breakdown for UI
    signal_breakdown: Dict[str, Any] = field(default_factory=dict)  # Legacy/trend-vol-news
    conclusion: Dict[str, Any] = field(default_factory=dict)  # Kết luận: advice, situation, why_not_buy, market_analysis
    decision_explanation: Optional[DecisionExplanation] = None  # Structured for dashboard
    recommendation_id: str = ""  # PRODUCTION: traceability
    data_freshness: Dict[str, Any] = field(default_factory=dict)  # forecast_as_of, etc.


def _load_config() -> dict:
    try:
        import yaml
        path = _ROOT / "configs" / "config.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def get_forecast_signal(symbol: str, features_dir: Optional[str] = None) -> Tuple[str, float, float]:
    """
    Fallback when no cached inference: use last row of features.
    Returns (direction, strength, volatility_proxy).
    """
    cfg = _load_config()
    base = Path(features_dir or cfg.get("data", {}).get("features_dir", "data/features/vn30"))
    if not base.is_absolute():
        base = _ROOT / base
    path = base / f"{symbol.upper()}.csv"
    if not path.exists():
        logger.warning("No features file for %s at %s", symbol, path)
        return "flat", 0.0, 0.5

    import pandas as pd
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    if len(df) == 0:
        return "flat", 0.0, 0.5

    row = df.iloc[-1]
    ret = float(row.get("return_1d", 0) or 0)
    vol = float(row.get("volatility_5", 1.0) or 1.0)
    vol = max(vol, 0.1)
    strength = min(1.0, abs(ret) / 2.0)
    if ret > 0.05:
        direction = "up"
    elif ret < -0.05:
        direction = "down"
    else:
        direction = "flat"
    return direction, strength, vol


def _summarize_text(text: str, max_sentences: int = 2, max_chars: int = 200) -> str:
    """
    Tóm tắt extractive: lấy 2-3 câu đầu hoặc max_chars ký tự.
    Giúp người dùng nắm nội dung nhanh, link gốc để kiểm chứng.
    """
    if not text or not str(text).strip():
        return ""
    text = str(text).strip()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if not sentences:
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    out = ". ".join(sentences[:max_sentences])
    if out and not out.endswith("."):
        out += "."
    if len(out) > max_chars:
        out = out[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return out


def get_news_sentiment(symbol: str, news_db_path: Optional[str] = None, days: int = 30) -> Tuple[float, int, List[dict]]:
    """Returns (avg_sentiment, article_count, recent_articles). Delegates to NewsService."""
    try:
        from src.app_services.news_service import get_articles, get_sentiment
        avg, count = get_sentiment(symbol, days=days)
        arts = get_articles(symbol, days=days, limit=10)
        articles = [
            {
                "title": a.title,
                "summary": a.summary,
                "url": a.url,
                "sentiment": a.sentiment,
                "source": a.source,
                "relevance_score": a.relevance_score,
            }
            for a in arts
        ]
        return avg, count, articles
    except Exception as e:
        logger.warning("News sentiment for %s: %s", symbol, e)
        return 0.0, 0, []


def _position_frac(profile: UserProfile) -> float:
    """Fraction of capital in position based on tolerance and leverage."""
    tol = (profile.risk_tolerance or "medium").strip().lower()
    base = 0.1 if tol == "low" else (0.2 if tol == "medium" else 0.3)
    # Leverage increases effective position (and risk)
    leverage_factor = 1.0 + min(profile.leverage or 0, 0.5)  # cap leverage effect
    return min(base * leverage_factor, 0.5)


def _probabilistic_decision(
    ensemble_mean: float,
    risk_of_loss_pct: float,
    risk_of_ruin_pct: float,
    sentiment: float,
    confidence: float,
    risk_tolerance: str,
    news_count: int,
) -> str:
    """
    Probabilistic decision: combine signals with user tolerance.
    No hard thresholds; weighted score.
    """
    tol = (risk_tolerance or "medium").strip().lower()

    # Score components (higher = more Buy)
    score = 0.0
    # Forecast direction: positive mean -> +1, negative -> -1
    if ensemble_mean > 0.1:
        score += 0.3
    elif ensemble_mean > 0:
        score += 0.15
    elif ensemble_mean < -0.1:
        score -= 0.3
    elif ensemble_mean < 0:
        score -= 0.15

    # Risk: lower P(loss) -> higher score
    if risk_of_loss_pct < 40:
        score += 0.2
    elif risk_of_loss_pct > 60:
        score -= 0.2

    # Ruin: high ruin prob -> strong Avoid
    if risk_of_ruin_pct > 15:
        score -= 0.3
    elif risk_of_ruin_pct < 5:
        score += 0.1

    # Sentiment: weak signal
    if news_count > 0:
        score += sentiment * 0.1

    # Confidence: low confidence -> bias toward Hold
    if confidence < 0.4:
        score *= 0.7  # dampen

    # Tolerance adjustment
    if tol == "low":
        if score > 0.3:
            return "Hold"  # Conservative: don't Buy easily
        elif score < -0.2:
            return "Avoid"
        return "Hold"
    elif tol == "high":
        if score > 0.1:
            return "Buy"
        elif score < -0.3:
            return "Hold"  # Even high tolerance avoids extreme
        return "Buy" if score > 0 else "Hold"
    else:
        if score > 0.25:
            return "Buy"
        elif score < -0.25:
            return "Avoid"
        return "Hold"


def __news_event_from_signal(sig: Any) -> "NewsEventSignal":
    """Build NewsEventSignal from StockNewsSignal."""
    from src.app_services.signals import NewsEventSignal
    if not sig:
        return NewsEventSignal(composite_score=0.0, article_count=0, net_impact_label="neutral", net_impact_confidence=0)
    return NewsEventSignal(
        composite_score=sig.composite_score,
        article_count=sig.article_count,
        net_impact_label=sig.net_impact_label or "neutral",
        net_impact_confidence=sig.net_impact_confidence or 0,
    )


def _build_decision_explanation(rec: str, sig: Any) -> DecisionExplanation:
    """Build structured DecisionExplanation for dashboard."""
    # Primary signal
    if sig.forecast_mean > 0.1:
        primary = f"Dự báo xu hướng tăng (kỳ vọng +{sig.forecast_mean:.2f}%)."
    elif sig.forecast_mean < -0.1:
        primary = f"Dự báo xu hướng giảm (kỳ vọng {sig.forecast_mean:.2f}%)."
    else:
        primary = "Dự báo đi ngang, biến động thấp."

    # Blocking factors
    blocking = []
    if sig.prob_loss_pct > 50:
        blocking.append(f"Xác suất lỗ cao ({sig.prob_loss_pct:.0f}%).")
    if sig.prob_ruin_pct > 15:
        blocking.append(f"Rủi ro sụt giảm mạnh vốn ({sig.prob_ruin_pct:.0f}%).")
    if sig.confidence < 0.4:
        blocking.append(f"Độ tin cậy mô hình thấp ({sig.confidence*100:.0f}%).")
    if rec == "Avoid" and not blocking:
        blocking.append("Tín hiệu tổng thể tiêu cực.")

    # Supporting factors
    supporting = []
    if sig.forecast_mean > 0 and rec != "Avoid":
        supporting.append(f"Xu hướng dự báo tích cực (+{sig.forecast_mean:.2f}%).")
    if sig.news_count > 0 and sig.news_sentiment > 0.1:
        supporting.append(f"Tin tức tích cực ({sig.news_count} bài).")
    if sig.prob_loss_pct < 40:
        supporting.append(f"Xác suất lỗ chấp nhận được ({sig.prob_loss_pct:.0f}%).")

    # Action summary
    if rec == "Buy":
        action = "Có thể cân nhắc mua với tỷ trọng phù hợp mức chấp nhận rủi ro."
    elif rec == "Hold":
        action = "Nên quan sát thêm; chờ tín hiệu rõ ràng hơn trước khi quyết định."
    else:
        action = "Không nên mua mới; cân nhắc cắt lỗ nếu đang nắm giữ."

    return DecisionExplanation(
        primary_signal=primary,
        blocking_factors=blocking,
        supporting_factors=supporting,
        action_summary=action,
    )


def get_risk_advice(profile: UserProfile, symbol: str) -> RiskAdvice:
    """
    Main entry: 4 signal layers → Decision Engine → RiskAdvice.
    """
    from src.app_services.signals import (
        get_price_technical_signal,
        get_ml_forecast_signal,
        get_news_event_signal,
        get_risk_uncertainty_signal,
    )
    from src.app_services.decision_engine import decide

    symbol = symbol.strip().upper()

    # 1. Price/Technical Signal
    price_technical = get_price_technical_signal(symbol)

    # 2. ML/DL Forecast Signal
    ml_forecast = get_ml_forecast_signal(symbol, price_technical)

    # 3. News & Event Signal (full signal for news_analysis in explanation)
    from src.app_services.news_intelligence import get_stock_news_signal
    try:
        news_signal = get_stock_news_signal(symbol, days=30, min_relevance=0.0, limit_articles=20)
        news_event = __news_event_from_signal(news_signal)
    except Exception as e:
        logger.warning("News signal failed for %s: %s", symbol, e)
        news_event = __news_event_from_signal(None)
        news_signal = None

    # 4. Risk & Uncertainty Signal
    vol = (price_technical.volatility_pct if price_technical else 1.0) or 1.0
    pos_frac = _position_frac(profile)
    risk_uncertainty = get_risk_uncertainty_signal(ml_forecast, vol, pos_frac)

    # Decision Engine
    decision = decide(profile, price_technical, ml_forecast, news_event, risk_uncertainty)

    # Build 4-layer breakdown for UI
    signal_layers = {
        "price_technical": (
            {
                "direction": price_technical.direction,
                "strength": price_technical.strength,
                "rsi": price_technical.rsi,
                "return_1d": price_technical.return_1d,
                "volatility_pct": price_technical.volatility_pct,
            }
            if price_technical
            else None
        ),
        "ml_forecast": {
            "mean": ml_forecast.mean,
            "std": ml_forecast.std,
            "confidence": ml_forecast.confidence,
            "used_inference": ml_forecast.used_inference,
            "model_weights": ml_forecast.model_weights,
            "as_of_date": getattr(ml_forecast, "as_of_date", "") or "",
        },
        "news_event": {
            "composite_score": news_event.composite_score,
            "article_count": news_event.article_count,
            "net_impact_label": news_event.net_impact_label,
            "net_impact_confidence": news_event.net_impact_confidence,
        },
        "risk_uncertainty": {
            "prob_loss_pct": risk_uncertainty.prob_loss_pct,
            "prob_ruin_pct": risk_uncertainty.prob_ruin_pct,
            "expected_return_lower": risk_uncertainty.expected_return_lower,
            "expected_return_upper": risk_uncertainty.expected_return_upper,
            "volatility_pct": risk_uncertainty.volatility_pct,
        },
    }

    # Legacy signal_breakdown for compatibility
    signal_breakdown = {
        "trend": "positive" if ml_forecast.mean > 0.1 else ("negative" if ml_forecast.mean < -0.1 else "neutral"),
        "volatility": (
            "high"
            if risk_uncertainty.volatility_pct > 2
            else ("low" if risk_uncertainty.volatility_pct < 1 else "medium")
        ),
        "news": (
            {"count": news_event.article_count, "sentiment": news_event.composite_score}
            if news_event.article_count > 0
            else None
        ),
        "model_weights": ml_forecast.model_weights if ml_forecast.used_inference else None,
    }

    # Conclusion and decision_explanation
    class _RiskProxy:
        prob_loss_pct = risk_uncertainty.prob_loss_pct
        prob_ruin_pct = risk_uncertainty.prob_ruin_pct
        expected_return_lower = risk_uncertainty.expected_return_lower
        expected_return_upper = risk_uncertainty.expected_return_upper
        volatility_pct = risk_uncertainty.volatility_pct

    conclusion = build_conclusion(
        recommendation=decision.final_action,
        ensemble_mean=ml_forecast.mean,
        risk=_RiskProxy(),
        sentiment=news_event.composite_score,
        news_count=news_event.article_count,
        confidence=ml_forecast.confidence,
    )

    # Phân tích tin tức: tóm tắt từ top impacts để user thấy "model đọc báo"
    news_analysis = None
    if news_signal and news_signal.article_count > 0:
        top3 = getattr(news_signal, "top_3_impact", []) or []
        label_vi = {"bullish": "tích cực", "bearish": "tiêu cực", "neutral": "trung tính"}.get(news_event.net_impact_label, "trung tính")
        parts = [f"Đã phân tích {news_signal.article_count} bài tin liên quan. Tổng quan: {label_vi}."]
        for i, item in enumerate(top3[:3], 1):
            why = getattr(item, "why_it_matters", None) or ""
            if why:
                parts.append(f"{i}. {why}")
        if len(parts) > 1:
            news_analysis = " ".join(parts)

    decision_explanation = DecisionExplanation(
        primary_signal=f"Dự báo xu hướng {'tăng' if ml_forecast.mean > 0.1 else 'giảm' if ml_forecast.mean < -0.1 else 'đi ngang'} (kỳ vọng {ml_forecast.mean:.2f}%).",
        blocking_factors=(
            [f"Xác suất lỗ cao ({risk_uncertainty.prob_loss_pct:.0f}%)."]
            if risk_uncertainty.prob_loss_pct > 50
            else []
        )
        + (
            [f"Rủi ro sụt giảm mạnh vốn ({risk_uncertainty.prob_ruin_pct:.0f}%)."]
            if risk_uncertainty.prob_ruin_pct > 15
            else []
        )
        + (
            [f"Độ tin cậy mô hình thấp ({ml_forecast.confidence*100:.0f}%)."]
            if ml_forecast.confidence < 0.4
            else []
        ),
        supporting_factors=(
            [f"Xu hướng dự báo tích cực (+{ml_forecast.mean:.2f}%)."]
            if ml_forecast.mean > 0 and decision.final_action != "Avoid"
            else []
        )
        + (
            [f"Tin tức tích cực ({news_event.article_count} bài)."]
            if news_event.article_count > 0 and news_event.net_impact_label == "bullish"
            else []
        )
        + (
            [f"Xác suất lỗ chấp nhận được ({risk_uncertainty.prob_loss_pct:.0f}%)."]
            if risk_uncertainty.prob_loss_pct < 40
            else []
        ),
        action_summary=(
            "Có thể cân nhắc mua với tỷ trọng phù hợp mức chấp nhận rủi ro."
            if decision.final_action == "Buy"
            else (
                "Nên quan sát thêm; chờ tín hiệu rõ ràng hơn trước khi quyết định."
                if decision.final_action == "Hold"
                else "Không nên mua mới; cân nhắc cắt lỗ nếu đang nắm giữ."
            )
        ),
        news_analysis=news_analysis,
    )

    rec_id = str(uuid.uuid4())[:8]
    data_freshness = {
        "forecast_as_of": getattr(ml_forecast, "as_of_date", "") or "",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if price_technical and price_technical.last_date:
        data_freshness["price_last_date"] = price_technical.last_date

    return RiskAdvice(
        recommendation=decision.final_action,
        risk_of_loss_pct=round(risk_uncertainty.prob_loss_pct, 1),
        risk_of_ruin_pct=round(risk_uncertainty.prob_ruin_pct, 1),
        expected_return_lower=round(risk_uncertainty.expected_return_lower, 2),
        expected_return_upper=round(risk_uncertainty.expected_return_upper, 2),
        confidence_score=decision.confidence_score,
        explanation=decision.explanation,
        position_size_suggestion=decision.position_size_suggestion,
        signal_layers=signal_layers,
        signal_breakdown=signal_breakdown,
        conclusion=conclusion,
        decision_explanation=decision_explanation,
        recommendation_id=rec_id,
        data_freshness=data_freshness,
    )


def build_explanation(
    recommendation: str,
    ensemble_mean: float,
    risk: Any,
    sentiment: float,
    news_count: int,
    confidence: float,
    used_inference: bool,
    model_weights: dict,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build plain-language explanation and signal breakdown for UI.
    """
    parts = []
    breakdown = {"trend": None, "volatility": None, "momentum": None, "news": None}

    # Trend
    if ensemble_mean > 0.1:
        trend_label = "tăng (bullish)"
        breakdown["trend"] = "positive"
    elif ensemble_mean < -0.1:
        trend_label = "giảm (bearish)"
        breakdown["trend"] = "negative"
    else:
        trend_label = "đi ngang (sideways)"
        breakdown["trend"] = "neutral"
    parts.append(f"Dự báo lợi nhuận kỳ tới: {trend_label} (kỳ vọng ~{ensemble_mean:.2f}%).")

    # Volatility
    if risk.volatility_pct > 2:
        vol_label = "cao"
        breakdown["volatility"] = "high"
    elif risk.volatility_pct < 1:
        vol_label = "thấp"
        breakdown["volatility"] = "low"
    else:
        vol_label = "trung bình"
        breakdown["volatility"] = "medium"
    parts.append(f"Biến động hiện tại: {vol_label} ({risk.volatility_pct:.1f}%).")

    # News
    if news_count > 0:
        sent_label = "tích cực" if sentiment > 0.1 else ("tiêu cực" if sentiment < -0.1 else "trung tính")
        parts.append(f"Tin tức ({news_count} bài): sentiment {sent_label}.")
        breakdown["news"] = {"count": news_count, "sentiment": sentiment}
    else:
        parts.append("Không có tin tức gần đây.")
        breakdown["news"] = None

    # Risks
    parts.append(
        f"Xác suất lỗ: ~{risk.prob_loss_pct:.0f}%. Rủi ro sụt giảm mạnh vốn: ~{risk.prob_ruin_pct:.0f}%."
    )
    parts.append(
        f"Khoảng tin cậy 95% lợi nhuận: [{risk.expected_return_lower:.2f}%, {risk.expected_return_upper:.2f}%]."
    )

    # Model reasoning (if inference was used)
    if used_inference and model_weights:
        top = sorted(model_weights.items(), key=lambda x: -x[1])[:3]
        parts.append(f"Mô hình chính: {', '.join(f'{n}({w*100:.0f}%)' for n, w in top)}.")
        breakdown["model_weights"] = model_weights

    parts.append(f"Độ tin cậy mô hình: {confidence*100:.0f}%.")
    parts.append(f"Khuyến nghị: {recommendation}.")

    return " ".join(parts), breakdown


def build_conclusion(
    recommendation: str,
    ensemble_mean: float,
    risk: Any,
    sentiment: float,
    news_count: int,
    confidence: float,
) -> Dict[str, Any]:
    """
    Xây dựng kết luận cuối cùng: lời khuyên, tình hình, lý do không nên mua, phân tích thị trường.
    """
    why_not_buy = []
    if recommendation == "Avoid":
        why_not_buy = [
            f"Xác suất lỗ cao (~{risk.prob_loss_pct:.0f}%), rủi ro sụt giảm mạnh vốn ~{risk.prob_ruin_pct:.0f}%.",
            f"Dự báo lợi nhuận âm hoặc rất thấp (kỳ vọng ~{ensemble_mean:.2f}%).",
            "Thị trường đang có xu hướng tiêu cực, chưa phải thời điểm vào lệnh.",
        ]
    elif recommendation == "Hold":
        why_not_buy = [
            f"Chưa có tín hiệu rõ ràng; xác suất lỗ ~{risk.prob_loss_pct:.0f}%.",
            "Nên quan sát thêm trước khi quyết định mua hoặc bán.",
        ]

    situation = []
    if ensemble_mean > 0.1:
        situation.append("Dự báo xu hướng tăng trong kỳ tới.")
    elif ensemble_mean < -0.1:
        situation.append("Dự báo xu hướng giảm, cần thận trọng.")
    else:
        situation.append("Thị trường đi ngang, ít biến động rõ rệt.")
    situation.append(f"Biến động: {'cao' if risk.volatility_pct > 2 else 'thấp' if risk.volatility_pct < 1 else 'trung bình'}.")
    if news_count > 0:
        sent = "tích cực" if sentiment > 0.1 else ("tiêu cực" if sentiment < -0.1 else "trung tính")
        situation.append(f"Tin tức gần đây: {sent}.")

    market_analysis = (
        f"Khoảng lợi nhuận kỳ vọng 95%: [{risk.expected_return_lower:.2f}%, {risk.expected_return_upper:.2f}%]. "
        f"Độ tin cậy mô hình {confidence*100:.0f}%. "
    )
    if risk.prob_loss_pct > 50:
        market_analysis += "Trong những ngày có xác suất lỗ cao, nên hạn chế mua mới hoặc giảm tỷ trọng."
    elif risk.volatility_pct > 2:
        market_analysis += "Biến động cao, giá có thể dao động mạnh trong ngắn hạn."

    advice = ""
    if recommendation == "Buy":
        advice = "Có thể cân nhắc mua với tỷ trọng phù hợp mức chấp nhận rủi ro. Nên chia nhỏ lệnh và đặt stop-loss."
    elif recommendation == "Hold":
        advice = "Nên chờ tín hiệu rõ ràng hơn. Nếu đang nắm giữ, có thể tiếp tục theo dõi."
    else:
        advice = "Không nên mua mới trong giai đoạn này. Nếu đang nắm giữ, cân nhắc cắt lỗ hoặc giảm tỷ trọng."

    return {
        "advice": advice,
        "situation": " ".join(situation),
        "why_not_buy": why_not_buy,
        "market_analysis": market_analysis,
    }
