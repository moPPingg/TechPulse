"""
LLM-based news enrichment: AI đọc tin và phân loại sentiment, event_type, impact.

Bật qua config news.yaml:
  llm_enrich:
    enabled: true
    provider: openai   # openai | anthropic
    model: gpt-4o-mini

Env: OPENAI_API_KEY hoặc ANTHROPIC_API_KEY
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EVENT_TYPES = ("earnings", "legal", "macro", "operations", "guidance", "ma", "dividend", "other")

_PROMPT = """Phân tích tin tài chính/chứng khoán sau và trả về JSON (chỉ JSON, không giải thích thêm):
{
  "sentiment": <số -1 đến 1: -1=rất tiêu cực, 0=trung tính, 1=rất tích cực>,
  "event_type": "<earnings|legal|macro|operations|guidance|ma|dividend|other>",
  "impact_horizon": "<intraday|short_term|long_term>",
  "ticker_relevance": <số 0 đến 1: độ liên quan đến cổ phiếu cụ thể>,
  "impact_summary": "<1 câu ngắn giải thích ảnh hưởng đến giá cổ phiếu, tiếng Việt>"
}

Tiêu đề: {title}
Nội dung (rút gọn): {body}

JSON:"""


def _call_openai(text: str, model: str, api_key: str, max_tokens: int = 200) -> Optional[dict]:
    """Gọi OpenAI API."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        return _parse_json(content)
    except Exception as e:
        logger.warning("OpenAI enrich error: %s", e)
        return None


def _call_anthropic(text: str, model: str, api_key: str, max_tokens: int = 200) -> Optional[dict]:
    """Gọi Anthropic API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": text}],
        )
        block = resp.content[0] if resp.content else None
        content = (block.text if hasattr(block, "text") else str(block) if block else "").strip()
        return _parse_json(content)
    except Exception as e:
        logger.warning("Anthropic enrich error: %s", e)
        return None


def _parse_json(s: str) -> Optional[dict]:
    """Parse JSON từ response, chấp nhận ```json ... ```."""
    if not s:
        return None
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        start = 1 if lines[0].lower().startswith("```json") else 0
        end = -1 if (lines and lines[-1].strip() == "```") else len(lines)
        s = "\n".join(lines[start:end])
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _validate_enrichment(raw: dict) -> dict:
    """Chuẩn hóa và validate output từ LLM."""
    out = {
        "sentiment": 0.0,
        "event_type": "other",
        "impact_horizon": "short_term",
        "ticker_relevance": 0.5,
        "impact_summary": "",
    }
    if not isinstance(raw, dict):
        return out
    sent = raw.get("sentiment")
    if isinstance(sent, (int, float)):
        out["sentiment"] = max(-1.0, min(1.0, float(sent)))
    evt = str(raw.get("event_type", "other")).lower().strip()
    out["event_type"] = evt if evt in EVENT_TYPES else "other"
    h = str(raw.get("impact_horizon", "short_term")).lower().strip()
    if h in ("intraday", "short_term", "long_term"):
        out["impact_horizon"] = h
    rel = raw.get("ticker_relevance")
    if isinstance(rel, (int, float)):
        out["ticker_relevance"] = max(0.0, min(1.0, float(rel)))
    if isinstance(raw.get("impact_summary"), str):
        out["impact_summary"] = raw["impact_summary"].strip()[:500]
    return out


def enrich_with_llm(
    title: str,
    body: str,
    config: dict,
) -> Optional[dict]:
    """
    Gọi LLM để enrich 1 bài. Trả về dict:
    sentiment, event_type, impact_horizon, ticker_relevance, impact_summary
    hoặc None nếu lỗi / chưa bật.
    """
    cfg = config.get("llm_enrich") or {}
    if not cfg.get("enabled"):
        return None
    provider = (cfg.get("provider") or "openai").lower()
    model = cfg.get("model") or ("gpt-4o-mini" if provider == "openai" else "claude-3-haiku-20240307")
    max_tokens = cfg.get("max_tokens", 200)
    body_trunc = (body or "")[:3000]  # Giới hạn token
    prompt = _PROMPT.format(title=title or "", body=body_trunc)

    api_key = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.debug("OPENAI_API_KEY not set, skip LLM enrich")
            return None
        raw = _call_openai(prompt, model, api_key, max_tokens)
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.debug("ANTHROPIC_API_KEY not set, skip LLM enrich")
            return None
        raw = _call_anthropic(prompt, model, api_key, max_tokens)
    else:
        logger.warning("Unknown llm_enrich provider: %s", provider)
        return None

    if not raw:
        return None
    return _validate_enrichment(raw)


def enrich_batch_with_llm(
    articles: List[Dict[str, Any]],
    config: dict,
) -> List[Optional[dict]]:
    """
    Enrich nhiều bài. Gọi tuần tự (có thể mở rộng batch API sau).
    Trả về list kết quả tương ứng; None nếu lỗi cho bài đó.
    """
    cfg = config.get("llm_enrich") or {}
    if not cfg.get("enabled"):
        return [None] * len(articles)
    delay = cfg.get("delay_seconds", 0.5)
    results = []
    for a in articles:
        r = enrich_with_llm(
            title=a.get("title", ""),
            body=a.get("body_clean") or a.get("body_raw", ""),
            config=config,
        )
        results.append(r)
        if delay > 0:
            time.sleep(delay)
    return results
