"""
Backend API cho web app thuần (HTML/CSS/JS).
- GET  /         → phục vụ web/ (index.html + static)
- GET  /api/symbols   → danh sách mã VN30
- POST /api/recommend → nhận profile + symbol, trả về khuyến nghị + risk + explanation

Chạy: từ thư mục gốc project:
  uvicorn api:app --reload --host 0.0.0.0
Mở: http://localhost:8000
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.app_services.recommendation import UserProfile, get_risk_advice

app = FastAPI(
    title="TechPulse API",
    description="API khuyến nghị cổ phiếu (Buy/Hold/Avoid) + risk metrics",
)

# Danh sách VN30 dùng chung
VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSI", "STB", "TCB", "TPB",
    "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE", "SSB", "PDR",
]


# --- Request/Response models ---

class RecommendRequest(BaseModel):
    name: str = Field(default="Khách", description="Họ tên hoặc nickname")
    capital: float = Field(ge=0, description="Vốn (VND)")
    years_experience: str = Field(description="Kinh nghiệm: '< 1 năm' | '1–3 năm' | '3–5 năm' | '5+ năm'")
    risk_tolerance: str = Field(description="Khả năng chấp nhận rủi ro: 'Thấp' | 'Trung bình' | 'Cao'")
    symbol: str = Field(description="Mã cổ phiếu VN30")


class RecommendResponse(BaseModel):
    recommendation: str  # "Buy" | "Hold" | "Avoid"
    risk_of_loss_pct: float
    risk_of_ruin_pct: float
    explanation: str


# --- API routes ---

@app.get("/api/symbols")
def get_symbols():
    """Trả về danh sách mã VN30 cho dropdown."""
    return {"symbols": VN30}


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Nhận profile + symbol → trả về khuyến nghị, risk of loss, risk of ruin, explanation.
    Logic nằm trong src.app_services.recommendation (đọc features + news DB).
    """
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
    )
    try:
        advice = get_risk_advice(profile, symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RecommendResponse(
        recommendation=advice.recommendation,
        risk_of_loss_pct=advice.risk_of_loss_pct,
        risk_of_ruin_pct=advice.risk_of_ruin_pct,
        explanation=advice.explanation,
    )


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
