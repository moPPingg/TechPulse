"""
Decision Engine: Combines 4 signal layers into final recommendation.

PRODUCTION: All weights and thresholds are config-driven via configs/decision.yaml.
See docs/PRODUCTION_REVIEW_AND_REDESIGN.md for audit rationale.

Produces:
- final_action: Buy | Hold | Avoid
- position_size_suggestion: fraction of capital (0-0.5)
- confidence_score: 0-1 (model confidence; NOT calibrated accuracy probability)
- explanation: natural language summary
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import logging

if TYPE_CHECKING:
    from src.app_services.signals import (
        PriceTechnicalSignal,
        MLForecastSignal,
        NewsEventSignal,
        RiskUncertaintySignal,
    )
    from src.app_services.recommendation import UserProfile

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_decision_config() -> dict:
    """Load decision config from YAML. Returns empty dict on failure."""
    try:
        import yaml
        path = _ROOT / "configs" / "decision.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("decision", {})
    except Exception as e:
        logger.warning("Failed to load decision config: %s; using defaults", e)
    return {}


@dataclass
class DecisionOutput:
    """Decision engine output."""
    final_action: str  # "Buy" | "Hold" | "Avoid"
    position_size_suggestion: float  # 0-0.5
    confidence_score: float  # 0-1; model confidence (not calibrated probability)
    explanation: str


def _position_frac(profile: Any, cfg: dict) -> float:
    """Base position fraction from risk tolerance and leverage. Config-driven."""
    tol = (profile.risk_tolerance or "medium").strip().lower()
    pos = cfg.get("position", {})
    base = (
        pos.get("base_low", 0.1) if tol == "low"
        else (pos.get("base_medium", 0.2) if tol == "medium" else pos.get("base_high", 0.3))
    )
    max_lf = pos.get("max_leverage_factor", 0.5)
    leverage_factor = 1.0 + min(profile.leverage or 0, max_lf)
    max_frac = pos.get("max_position_frac", 0.5)
    return min(base * leverage_factor, max_frac)


def decide(
    profile: Any,
    price_technical: "PriceTechnicalSignal | None",
    ml_forecast: "MLForecastSignal",
    news_event: "NewsEventSignal",
    risk_uncertainty: "RiskUncertaintySignal",
) -> DecisionOutput:
    """
    Combine 4 signal layers into final action, position size, confidence, explanation.
    Uses configs/decision.yaml for all weights and thresholds.
    """
    cfg = _load_decision_config()
    tol = (profile.risk_tolerance or "medium").strip().lower()

    pt = cfg.get("price_technical", {})
    mf = cfg.get("ml_forecast", {})
    ne = cfg.get("news_event", {})
    ru = cfg.get("risk_uncertainty", {})
    th = cfg.get("thresholds", {})

    score = 0.0

    # 1. Price/Technical (config-driven)
    if price_technical:
        w_up = pt.get("direction_up_weight", 0.2)
        w_down = pt.get("direction_down_weight", 0.2)
        if price_technical.direction == "up":
            score += w_up * price_technical.strength
        elif price_technical.direction == "down":
            score -= w_down * price_technical.strength
        if price_technical.rsi is not None:
            rsi_bonus = pt.get("rsi_oversold_bonus", 0.05)
            rsi_penalty = pt.get("rsi_overbought_penalty", 0.05)
            if price_technical.rsi < 30:
                score += rsi_bonus
            elif price_technical.rsi > 70:
                score -= rsi_penalty

    # 2. ML/DL Forecast (config-driven)
    thr_bull = mf.get("strong_bullish_threshold", 0.1)
    thr_bear = mf.get("strong_bearish_threshold", -0.1)
    if ml_forecast.mean > thr_bull:
        score += 0.3
    elif ml_forecast.mean > 0:
        score += mf.get("weak_bullish_score", 0.15)
    elif ml_forecast.mean < thr_bear:
        score -= 0.3
    elif ml_forecast.mean < 0:
        score += mf.get("weak_bearish_score", -0.15)

    # 3. News & Event (config-driven)
    score += news_event.composite_score * ne.get("composite_weight", 0.15)
    if news_event.net_impact_label == "bullish":
        score += ne.get("net_impact_bullish_bonus", 0.05) * (news_event.net_impact_confidence / 100)
    elif news_event.net_impact_label == "bearish":
        score -= ne.get("net_impact_bearish_penalty", 0.05) * (news_event.net_impact_confidence / 100)

    # 4. Risk & Uncertainty (config-driven)
    pl_high = ru.get("prob_loss_high_threshold", 60)
    pl_med = ru.get("prob_loss_medium_threshold", 50)
    pl_low = ru.get("prob_loss_low_threshold", 40)
    if risk_uncertainty.prob_loss_pct > pl_high:
        score -= ru.get("prob_loss_high_penalty", 0.25)
    elif risk_uncertainty.prob_loss_pct > pl_med:
        score -= ru.get("prob_loss_medium_penalty", 0.1)
    elif risk_uncertainty.prob_loss_pct < pl_low:
        score += ru.get("prob_loss_low_bonus", 0.1)

    pr_high = ru.get("prob_ruin_high_threshold", 15)
    pr_med = ru.get("prob_ruin_medium_threshold", 10)
    if risk_uncertainty.prob_ruin_pct > pr_high:
        score -= ru.get("prob_ruin_high_penalty", 0.3)
    elif risk_uncertainty.prob_ruin_pct > pr_med:
        score -= ru.get("prob_ruin_medium_penalty", 0.1)

    # Confidence dampening
    conf = ml_forecast.confidence
    dampen_thr = cfg.get("confidence_dampen_threshold", 0.4)
    dampen_fac = cfg.get("confidence_dampen_factor", 0.7)
    if conf < dampen_thr:
        score *= dampen_fac

    # Tolerance-based decision (config-driven thresholds)
    tol_cfg = th.get("low_tolerance", {}) if tol == "low" else (
        th.get("high_tolerance", {}) if tol == "high" else th.get("medium_tolerance", {})
    )
    buy_min = tol_cfg.get("buy_min_score", 0.25)
    avoid_max = tol_cfg.get("avoid_max_score", -0.25)
    if tol == "low":
        buy_min = tol_cfg.get("buy_min_score", 0.35)
        avoid_max = tol_cfg.get("avoid_max_score", -0.2)
    elif tol == "high":
        buy_min = tol_cfg.get("buy_min_score", 0.1)
        avoid_max = tol_cfg.get("avoid_max_score", -0.35)

    if tol == "low":
        action = "Hold" if score > buy_min else ("Avoid" if score < avoid_max else "Hold")
    elif tol == "high":
        action = "Buy" if score > buy_min else ("Hold" if score < avoid_max else ("Buy" if score > 0 else "Hold"))
    else:
        action = "Buy" if score > buy_min else ("Avoid" if score < avoid_max else "Hold")

    # Position size (config-driven)
    base_frac = _position_frac(profile, cfg)
    pos_cfg = cfg.get("position", {})
    hold_frac = pos_cfg.get("hold_fraction", 0.5)
    pl_reduce_thr = pos_cfg.get("prob_loss_reduce_threshold", 45)
    pl_reduce_fac = pos_cfg.get("prob_loss_reduce_factor", 0.7)
    low_conf_fac = pos_cfg.get("low_confidence_reduce_factor", 0.8)

    if action == "Avoid":
        pos = 0.0
    elif action == "Hold":
        pos = base_frac * hold_frac
    else:
        pos = base_frac
        if risk_uncertainty.prob_loss_pct > pl_reduce_thr:
            pos *= pl_reduce_fac
        if conf < 0.5:
            pos *= low_conf_fac

    confidence = min(1.0, max(0.1, 0.5 + abs(score) * 0.5))
    confidence = (confidence + conf) / 2

    parts = _build_explanation_parts(
        action=action,
        position_size=pos,
        price_technical=price_technical,
        ml_forecast=ml_forecast,
        news_event=news_event,
        risk_uncertainty=risk_uncertainty,
    )
    explanation = " ".join(parts)

    return DecisionOutput(
        final_action=action,
        position_size_suggestion=round(pos, 3),
        confidence_score=round(confidence, 2),
        explanation=explanation,
    )


def _build_explanation_parts(
    action: str,
    position_size: float,
    price_technical: Any,
    ml_forecast: Any,
    news_event: Any,
    risk_uncertainty: Any,
) -> List[str]:
    parts = []

    if ml_forecast.mean > 0.1:
        parts.append(f"Dự báo xu hướng tăng (kỳ vọng +{ml_forecast.mean:.2f}%).")
    elif ml_forecast.mean < -0.1:
        parts.append(f"Dự báo xu hướng giảm (kỳ vọng {ml_forecast.mean:.2f}%).")
    else:
        parts.append("Dự báo đi ngang.")

    if price_technical:
        if price_technical.direction == "up":
            parts.append("Tín hiệu kỹ thuật hỗ trợ tăng.")
        elif price_technical.direction == "down":
            parts.append("Tín hiệu kỹ thuật cho thấy áp lực giảm.")

    if news_event.article_count > 0:
        if news_event.net_impact_label == "bullish":
            parts.append(f"Tin tức tích cực ({news_event.article_count} bài).")
        elif news_event.net_impact_label == "bearish":
            parts.append(f"Tin tức tiêu cực ({news_event.article_count} bài).")
        else:
            parts.append(f"Tin tức trung tính ({news_event.article_count} bài).")

    parts.append(
        f"Xác suất lỗ ~{risk_uncertainty.prob_loss_pct:.0f}%, "
        f"rủi ro sụt giảm mạnh ~{risk_uncertainty.prob_ruin_pct:.0f}%. "
        f"Khoảng tin cậy lợi nhuận: [{risk_uncertainty.expected_return_lower:.2f}%, {risk_uncertainty.expected_return_upper:.2f}%]."
    )

    if action == "Buy":
        pct = position_size * 100
        parts.append(f"Khuyến nghị: Mua với tỷ trọng tối đa ~{pct:.0f}% tổng vốn; nên chia nhỏ lệnh.")
    elif action == "Hold":
        parts.append("Khuyến nghị: Chờ tín hiệu rõ ràng hơn trước khi quyết định.")
    else:
        parts.append("Khuyến nghị: Không nên mua mới; cân nhắc cắt lỗ nếu đang nắm giữ.")

    return parts
