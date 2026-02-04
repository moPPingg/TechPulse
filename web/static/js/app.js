/**
 * TechPulse – Frontend thuần (HTML/CSS/JS).
 * - Nạp danh sách mã VN30 từ GET /api/symbols.
 * - Gửi form qua POST /api/recommend, hiển thị khuyến nghị + risk + explanation.
 */

const form = document.getElementById("profile-form");
const messageEl = document.getElementById("message");
const resultEl = document.getElementById("result");
const submitBtn = document.getElementById("submit-btn");
const symbolSelect = document.getElementById("symbol-select");

function setMessage(text, type = "info") {
  messageEl.textContent = text;
  messageEl.className = "message " + type;
}

function showResult(data) {
  const card = document.getElementById("recommendation-card");
  card.textContent = "Khuyến nghị: " + data.recommendation;
  card.className = "rec-card " + data.recommendation.toLowerCase();

  document.getElementById("risk-loss").textContent = data.risk_of_loss_pct + "%";
  document.getElementById("risk-ruin").textContent = data.risk_of_ruin_pct + "%";
  document.getElementById("explanation-text").textContent = data.explanation;

  resultEl.hidden = false;
}

/** Nạp danh sách mã VN30 vào dropdown */
async function loadSymbols() {
  try {
    const res = await fetch("/api/symbols");
    if (!res.ok) return;
    const { symbols } = await res.json();
    if (!Array.isArray(symbols) || symbols.length === 0) return;

    const current = symbolSelect.value;
    symbolSelect.innerHTML = symbols
      .map((s) => `<option value="${s}" ${s === current ? "selected" : ""}>${s}</option>`)
      .join("");
  } catch (_) {
    // Giữ lại option FPT mặc định
  }
}

/** Gửi form → API → hiển thị kết quả */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setMessage("Đang tính toán khuyến nghị...", "info");
  resultEl.hidden = true;
  submitBtn.disabled = true;

  const body = {
    name: form.name.value.trim() || "Khách",
    capital: Number(form.capital.value) || 0,
    years_experience: form.years_experience.value,
    risk_tolerance: form.risk_tolerance.value,
    symbol: form.symbol.value.trim().toUpperCase(),
  };

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      setMessage(data.detail || "Lỗi kết nối. Kiểm tra backend đã chạy chưa.", "error");
      return;
    }

    setMessage("");
    showResult(data);
  } catch (err) {
    setMessage("Lỗi: " + (err.message || "Không kết nối được tới server."), "error");
  } finally {
    submitBtn.disabled = false;
  }
});

loadSymbols();
