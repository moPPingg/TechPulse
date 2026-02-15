/**
 * TechPulse – Bloomberg/TradingView-grade UI
 * Flow: Select ticker → Load full analysis (DATA → SIGNAL → RISK → DECISION → NEWS)
 */

const VN30_FALLBACK = ["ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE", "SSB", "PDR"];

let currentSymbol = "";
let chartInstance = null;

const form = document.getElementById("profile-form");
const messageEl = document.getElementById("message");
const symbolSelect = document.getElementById("symbol-select");
const analysisView = document.getElementById("analysis-view");
const emptyState = document.getElementById("empty-state");

function setMessage(text, type = "info") {
  if (messageEl) {
    messageEl.textContent = text;
    messageEl.className = "message " + type;
  }
}

function getProfile() {
  if (!form) return { name: "Khách", capital: 1e8, years_experience: "1–3 năm", risk_tolerance: "medium", leverage: 0 };
  let capRaw = 100;
  if (form.capital?.value === "custom") {
    capRaw = Number(document.getElementById("capital-custom")?.value) || 100;
  } else {
    capRaw = Number(form.capital?.value) || 100;
  }
  const cap = Math.max(1, capRaw) * 1e6;
  return {
    name: "Khách",
    capital: cap,
    years_experience: form.years_experience?.value || "1–3 năm",
    risk_tolerance: form.risk_tolerance?.value || "medium",
    leverage: Number(form.leverage?.value) || 0,
  };
}

function getForecastParams() {
  const h = document.querySelector('input[name="horizon"]:checked')?.value || "1d";
  const p = new URLSearchParams({ horizon: h });
  const d = document.getElementById("target-date")?.value;
  const m = document.getElementById("target-month")?.value;
  if (h === "date" && d) p.set("target_date", d);
  else if (h === "month" && m) p.set("target_month", m);
  return p.toString();
}

function getChartEndDate() {
  const h = document.querySelector('input[name="horizon"]:checked')?.value || "1d";
  const d = document.getElementById("target-date")?.value;
  const m = document.getElementById("target-month")?.value;
  if (h === "date" && d) return d;
  if (h === "month" && m) {
    const [y, mo] = m.split("-").map(Number);
    return new Date(y, mo, 0).toISOString().slice(0, 10);
  }
  return new Date().toISOString().slice(0, 10);
}

/** Số ngày hiển thị theo horizon: 1D=5, 7D=7, Ngày/Tháng=90 */
function getChartDays() {
  const h = document.querySelector('input[name="horizon"]:checked')?.value || "1d";
  if (h === "1d") return 5;
  if (h === "7d") return 7;
  return 90;
}

/** Chỉ hiển thị link nếu URL hợp lệ (http/https tuyệt đối), escape để tránh vỡ href / XSS */
function safeNewsLink(url, source) {
  if (!url || typeof url !== "string") return "";
  const u = url.trim();
  if (!/^https?:\/\//i.test(u)) return "";
  const safe = u.replace(/"/g, "&quot;").replace(/</g, "%3C").replace(/>/g, "%3E");
  const s = source ? ` <span class="news-source">(${source})</span>` : "";
  return `<a href="${safe}" target="_blank" rel="noopener noreferrer" class="intel-impact-link">Đọc chi tiết ${s}</a>`;
}

async function loadAnalysis(symbol) {
  if (!symbol) return;
  currentSymbol = symbol.trim().toUpperCase();
  setMessage("Đang tải phân tích...", "info");
  analysisView.hidden = true;
  emptyState.hidden = true;

  const profile = getProfile();
  const qs = getForecastParams();
  const chartEnd = getChartEndDate();
  const chartDays = getChartDays();

  const urls = {
    recommend: "/api/recommend",
    stock: `/api/stock/${currentSymbol}${qs ? "?" + qs : ""}`,
    chart: `/api/stock/${currentSymbol}/chart?days=${chartDays}&end_date=${encodeURIComponent(chartEnd)}`,
    newsIntel: `/api/stock/${currentSymbol}/news/intelligence?days=450&limit=8`,
  };

  try {
    const [recRes, stockRes, chartRes, newsRes] = await Promise.all([
      fetch(urls.recommend, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: profile.name,
          capital: profile.capital,
          years_experience: profile.years_experience,
          risk_tolerance: profile.risk_tolerance,
          leverage: profile.leverage,
          symbol: currentSymbol,
        }),
      }),
      fetch(urls.stock),
      fetch(urls.chart),
      fetch(urls.newsIntel),
    ]);

    const rec = await recRes.json().catch(() => ({}));
    const stock = await stockRes.json().catch(() => ({}));
    const chartData = await chartRes.json().catch(() => ({}));
    const newsIntel = await newsRes.json().catch(() => ({}));

    if (!recRes.ok) {
      setMessage(rec.detail || "Lỗi kết nối.", "error");
      emptyState.hidden = false;
      return;
    }

    renderAnalysis(rec, stock, chartData, newsIntel);
    const actualEnd = chartData?.ohlcv?.length ? chartData.ohlcv[chartData.ohlcv.length - 1].date : chartEnd;
    updateChartEndHint(actualEnd, chartEnd);
    analysisView.hidden = false;
    setMessage("");
    updateSymbolListSelection();
  } catch (err) {
    setMessage("Lỗi: " + (err.message || "Không kết nối được."), "error");
    emptyState.hidden = false;
  }
}

function renderAnalysis(rec, stock, chartData, newsIntel) {
  // DATA: Chart + summary
  renderChart(chartData);
  const ind = stock?.indicators || {};
  const dataSummary = document.getElementById("data-summary");
  if (dataSummary) {
    let html = "";
    if (ind.close != null) html += `<span class="data-item"><span class="label">Giá</span><span class="value mono">${Number(ind.close).toLocaleString("vi-VN")}</span></span>`;
    if (ind.last_date) html += `<span class="data-item"><span class="label">Cập nhật</span><span class="value">${ind.last_date}</span></span>`;
    dataSummary.innerHTML = html || '<span class="data-item">—</span>';
  }

  // SIGNAL: 4 layers from rec.signal_layers (fallback to stock/indicators)
  const layers = rec.signal_layers || {};
  const pt = layers.price_technical;
  const ptEl = document.getElementById("signal-price-technical");
  if (ptEl) {
    if (pt) {
      const dir = pt.direction || "flat";
      const dirLabel = { up: "↑ Tăng", down: "↓ Giảm", flat: "— Đi ngang" }[dir] || dir;
      let t = [`<span class="signal-value">${dirLabel}</span>`];
      if (pt.rsi != null) t.push(`<span class="signal-muted">RSI ${pt.rsi.toFixed(1)}</span>`);
      if (pt.return_1d != null) t.push(`<span class="signal-muted">Ret ${pt.return_1d.toFixed(2)}%</span>`);
      if (pt.volatility_pct != null) t.push(`<span class="signal-muted">Vol ${pt.volatility_pct.toFixed(2)}%</span>`);
      ptEl.innerHTML = t.join(" ");
    } else {
      let t = [];
      if (ind.rsi_14 != null) t.push(`RSI ${ind.rsi_14.toFixed(1)}`);
      if (ind.volatility_5 != null) t.push(`Vol ${ind.volatility_5.toFixed(2)}%`);
      if (ind.return_1d != null) t.push(`Ret ${ind.return_1d.toFixed(2)}%`);
      ptEl.innerHTML = t.length ? t.map(s => `<span class="signal-value">${s}</span>`).join(" ") : "—";
    }
  }
  const ml = layers.ml_forecast;
  const mlEl = document.getElementById("signal-ml-forecast");
  if (mlEl) {
    if (ml) {
      const c = ml.mean >= 0 ? "up" : "down";
      mlEl.innerHTML = `<span class="signal-value ${c}">${ml.mean.toFixed(2)}%</span><span class="signal-muted">± ${(ml.std || 0).toFixed(2)}%</span><span class="signal-muted">conf ${((ml.confidence || 0) * 100).toFixed(0)}%</span>`;
    } else {
      const fc = stock?.forecast?.forecasts?.[0];
      if (fc && fc.ensemble_mean != null) {
        const c = fc.ensemble_mean >= 0 ? "up" : "down";
        mlEl.innerHTML = `<span class="signal-value ${c}">${fc.ensemble_mean.toFixed(2)}%</span><span class="signal-muted">± ${(fc.ensemble_std || 0).toFixed(2)}%</span>`;
      } else mlEl.textContent = "—";
    }
  }
  const ne = layers.news_event;
  const neEl = document.getElementById("signal-news-event");
  if (neEl) {
    if (ne) {
      const c = ne.composite_score > 0 ? "up" : ne.composite_score < 0 ? "down" : "";
      const label = { bullish: "Bullish", bearish: "Bearish", neutral: "Neutral" }[ne.net_impact_label] || "Neutral";
      neEl.innerHTML = `<span class="signal-value ${c}">${ne.composite_score.toFixed(2)}</span><span class="signal-muted">${ne.article_count || 0} bài · ${label}</span>`;
    } else {
      const sig = newsIntel?.signal;
      if (sig) {
        const c = sig.composite_score > 0 ? "up" : sig.composite_score < 0 ? "down" : "";
        neEl.innerHTML = `<span class="signal-value ${c}">${sig.composite_score.toFixed(2)}</span><span class="signal-muted">${sig.article_count || 0} bài</span>`;
      } else neEl.textContent = stock?.news?.count ? `${stock.news.count} bài` : "—";
    }
  }
  const ru = layers.risk_uncertainty;
  const ruEl = document.getElementById("signal-risk-uncertainty");
  if (ruEl) {
    if (ru) {
      const pl = ru.prob_loss_pct != null ? ru.prob_loss_pct.toFixed(1) + "%" : "—";
      const pr = ru.prob_ruin_pct != null ? ru.prob_ruin_pct.toFixed(1) + "%" : "—";
      const lo = ru.expected_return_lower != null ? ru.expected_return_lower.toFixed(2) : "—";
      const hi = ru.expected_return_upper != null ? ru.expected_return_upper.toFixed(2) : "—";
      ruEl.innerHTML = `<span class="signal-muted">P(lỗ) ${pl}</span> <span class="signal-muted">P(ruin) ${pr}</span> <span class="signal-muted">CI [${lo}, ${hi}]%</span>`;
    } else {
      ruEl.innerHTML = `<span class="signal-muted">P(lỗ) ${rec.risk_of_loss_pct ?? "—"}%</span> <span class="signal-muted">P(ruin) ${rec.risk_of_ruin_pct ?? "—"}%</span>`;
    }
  }

  // RISK
  const riskEl = document.getElementById("risk-metrics");
  if (riskEl) {
    riskEl.innerHTML = `
      <span class="risk-item"><span class="label">P(lỗ)</span><span class="value mono">${rec.risk_of_loss_pct ?? "—"}%</span></span>
      <span class="risk-item"><span class="label">P(ruin)</span><span class="value mono">${rec.risk_of_ruin_pct ?? "—"}%</span></span>
      <span class="risk-item"><span class="label">Kỳ vọng</span><span class="value mono">${rec.expected_return_lower != null && rec.expected_return_upper != null ? `${rec.expected_return_lower}% – ${rec.expected_return_upper}%` : "—"}</span></span>
      <span class="risk-item"><span class="label">Tin cậy</span><span class="value mono">${rec.confidence_score != null ? (rec.confidence_score * 100).toFixed(0) + "%" : "—"}</span></span>
    `;
  }

  // DECISION (action, position size, confidence)
  const cardEl = document.getElementById("decision-card");
  const posEl = document.getElementById("decision-position");
  const confEl = document.getElementById("decision-confidence");
  const reasonEl = document.getElementById("decision-reasoning");
  if (cardEl) {
    cardEl.textContent = rec.recommendation || "—";
    cardEl.className = "decision-card decision-" + (rec.recommendation || "").toLowerCase();
  }
  if (posEl) {
    const pct = rec.position_size_suggestion != null ? (rec.position_size_suggestion * 100).toFixed(1) : "—";
    posEl.querySelector(".value").textContent = pct !== "—" ? pct + "% tổng vốn" : "—";
  }
  if (confEl) {
    const c = rec.confidence_score != null ? (rec.confidence_score * 100).toFixed(0) + "%" : "—";
    confEl.querySelector(".value").textContent = c;
  }
  if (reasonEl) {
    const de = rec.decision_explanation;
    if (de) {
      let html = `<div class="reason-primary">${de.primary_signal || ""}</div>`;
      if (de.news_analysis) html += `<div class="reason-news-analysis"><strong>📰 Phân tích tin tức:</strong> ${de.news_analysis}</div>`;
      if (de.blocking_factors?.length) html += `<div class="reason-blocking"><strong>Cần thận trọng:</strong><ul>${de.blocking_factors.map(f => `<li>${f}</li>`).join("")}</ul></div>`;
      if (de.supporting_factors?.length) html += `<div class="reason-supporting"><strong>Ủng hộ:</strong><ul>${de.supporting_factors.map(f => `<li>${f}</li>`).join("")}</ul></div>`;
      if (de.action_summary) html += `<div class="reason-action">${de.action_summary}</div>`;
      reasonEl.innerHTML = html;
    } else reasonEl.innerHTML = `<p>${rec.explanation || "—"}</p>`;
  }

  // INVESTMENT INTELLIGENCE
  const summaryEl = document.getElementById("intel-summary");
  const impactEl = document.getElementById("intel-impact-list");
  if (summaryEl || impactEl) {
    const sig = newsIntel?.signal;
    const top3 = newsIntel?.top_3_impact || [];
    const isGeneral = newsIntel?.is_general_fallback;
    if (sig) {
      const label = sig.net_impact_label || "neutral";
      const conf = sig.net_impact_confidence ?? 0;
      const labelVi = { bullish: "Bullish", bearish: "Bearish", neutral: "Neutral" }[label] || "Neutral";
      const labelClass = "intel-net-" + label;
      if (summaryEl) {
        const fallbackNote = isGeneral ? '<span class="intel-fallback-note">Tin thị trường chung</span>' : '';
        summaryEl.innerHTML = `<div class="intel-summary-box ${labelClass}"><span class="intel-net-label">Net news impact:</span> <strong>${labelVi}</strong> <span class="intel-net-conf">(+${conf}%)</span> ${fallbackNote}</div>`;
      }
      if (impactEl && top3.length) {
        const dirLabels = { bullish: "↑ Bullish", bearish: "↓ Bearish", neutral: "— Neutral" };
        const dirClass = { bullish: "bullish", bearish: "bearish", neutral: "neutral" };
        impactEl.innerHTML = top3.map((item, i) => `
          <div class="intel-impact-card">
            <div class="intel-impact-header">
              <span class="intel-impact-num">${i + 1}</span>
              <span class="intel-impact-direction ${dirClass[item.impact_direction] || "neutral"}">${dirLabels[item.impact_direction] || "—"}</span>
              <span class="intel-impact-conf">${Math.round((item.confidence || 0) * 100)}%</span>
            </div>
            <p class="intel-impact-why">${item.why_it_matters || "—"}</p>
            <div class="intel-impact-meta">
              <span class="intel-impact-horizon">${item.time_horizon || "—"}</span>
              ${safeNewsLink(item.url, item.source)}
            </div>
          </div>
        `).join("");
      } else if (impactEl) {
        const arts = newsIntel?.articles || [];
        if (arts.length) {
          impactEl.innerHTML = arts.slice(0, 5).map(a => `
            <div class="intel-impact-card">
              <p class="intel-impact-why">${(a.title || "").slice(0, 120)}${(a.title || "").length > 120 ? "…" : ""}</p>
              <div class="intel-impact-meta">
                ${safeNewsLink(a.url, a.source)}
              </div>
            </div>
          `).join("");
        } else {
          impactEl.innerHTML = '<p class="empty-intel">Chưa có tin tức khớp mã này. Thử mã <strong>FPT</strong> hoặc <strong>VCB</strong>. Chạy <code>python scripts/run_news_pipeline.py</code> (đầy đủ, không chỉ enrich) để thu thập thêm tin.</p>';
        }
      }
    } else {
      if (summaryEl) summaryEl.innerHTML = '<div class="intel-summary-box intel-net-neutral"><span class="intel-net-label">Net news impact:</span> <strong>Neutral</strong> <span class="intel-net-conf">(no data)</span></div>';
      if (impactEl) impactEl.innerHTML = '<p class="empty-intel">Chưa có dữ liệu tin tức. Chạy <code>python scripts/run_news_pipeline.py</code> (đầy đủ 5 bước). Sau đó thử mã FPT hoặc VCB.</p>';
    }
  }
}

/**
 * Chart: TradingView Lightweight Charts - vẽ lại từ đầu
 * Candlestick, dark theme, price marker xanh/đỏ, crosshair, tooltip, zoom/pan, grid mờ, responsive
 */
function renderChart(chartData) {
  const container = document.getElementById("chart-container");
  if (!container) return;
  container.innerHTML = "";

  const createChart = (typeof lightweightCharts !== "undefined" && lightweightCharts.createChart) ||
    (typeof LightweightCharts !== "undefined" && LightweightCharts.createChart);
  if (!createChart || !chartData?.ohlcv?.length) {
    container.innerHTML = '<p class="empty-msg">Không có dữ liệu biểu đồ.</p>';
    return;
  }

  if (chartInstance) {
    if (chartInstance._tooltipUnsub) chartInstance._tooltipUnsub();
    if (chartInstance._resizeObs) chartInstance._resizeObs.disconnect();
    chartInstance.remove();
    chartInstance = null;
  }

  const w = Math.max(container.clientWidth || 600, 400);
  const h = 340;

  const chart = createChart(container, {
    layout: {
      background: { color: "#0b0f14" },
      textColor: "#9ca3af",
      fontFamily: "JetBrains Mono, monospace",
      fontSize: 11,
    },
    grid: {
      vertLines: { color: "rgba(75, 85, 99, 0.15)" },
      horzLines: { color: "rgba(75, 85, 99, 0.15)" },
    },
    crosshair: {
      vertLine: {
        color: "rgba(59, 130, 246, 0.6)",
        width: 1,
        labelBackgroundColor: "#1f2937",
      },
      horzLine: {
        color: "rgba(59, 130, 246, 0.6)",
        width: 1,
        labelBackgroundColor: "#1f2937",
      },
    },
    rightPriceScale: {
      borderColor: "rgba(75, 85, 99, 0.4)",
      scaleMargins: { top: 0.06, bottom: 0.12 },
    },
    timeScale: {
      borderColor: "rgba(75, 85, 99, 0.4)",
      timeVisible: true,
      rightBarSpacing: 10,
      barSpacing: 12, // Increased spacing for "gaps"
      minBarSpacing: 2,
    },
    handleScroll: { vertTouchDrag: true, horzTouchDrag: true, mouseWheel: true, pressedMouseMove: true },
    width: w,
    height: h,
  });

  const ohlcv = chartData.ohlcv.map(d => ({ time: d.date, open: d.open, high: d.high, low: d.low, close: d.close }));

  const candle = chart.addCandlestickSeries({
    upColor: "#22c55e",
    downColor: "#ef4444",
    borderVisible: false, // Disable border to prevent "touching" and make body narrower relative to space
    wickUpColor: "#22c55e",
    wickDownColor: "#ef4444",
    wickVisible: true,
    lastValueVisible: true,
    priceLineVisible: false,
  });
  candle.setData(ohlcv);

  const last = ohlcv[ohlcv.length - 1];
  const prev = ohlcv[ohlcv.length - 2];
  const lastClose = last?.close ?? 0;
  const isUp = prev && lastClose >= prev.close;
  if (lastClose > 0) {
    candle.createPriceLine({
      price: lastClose,
      color: isUp ? "#22c55e" : "#ef4444",
      lineWidth: 1,
      axisLabelVisible: true,
      lineStyle: 2, // Dashed
    });
  }

  if (chartData.ma?.length) {
    const ma20 = chartData.ma.filter(d => d.ma_20 != null).map(d => ({ time: d.date, value: d.ma_20 }));
    const ma50 = chartData.ma.filter(d => d.ma_50 != null).map(d => ({ time: d.date, value: d.ma_50 }));
    if (ma20.length) chart.addLineSeries({ color: "#3b82f6", lineWidth: 1, title: "MA20", lastValueVisible: true, priceLineVisible: false, crosshairMarkerVisible: false }).setData(ma20);
    if (ma50.length) chart.addLineSeries({ color: "#f59e0b", lineWidth: 1, title: "MA50", lastValueVisible: true, priceLineVisible: false, crosshairMarkerVisible: false }).setData(ma50);
  }

  const tooltip = document.getElementById("chart-tooltip");
  const map = {};
  chartData.ohlcv.forEach(d => { map[d.date] = d; });
  const fmt = (v) => (v != null && !isNaN(v) ? Number(v).toLocaleString("vi-VN", { minimumFractionDigits: 0, maximumFractionDigits: 2 }) : "-");

  const unsub = chart.subscribeCrosshairMove((param) => {
    if (!tooltip) return;
    if (!param?.point || param.point.x < 0 || param.point.y < 0 || !param.time) {
      tooltip.style.display = "none";
      return;
    }
    // Lấy dữ liệu OHLC từ series candle
    let open = 0, high = 0, low = 0, close = 0;
    const candleData = param.seriesData.get(candle);
    if (candleData) {
      open = candleData.open;
      high = candleData.high;
      low = candleData.low;
      close = candleData.close;
    }

    // Lấy dữ liệu Volume từ series volume (nếu có - chưa implement ở dưới nhưng cứ để logic)
    // Tạm thời nếu ko có volume series riêng thì lấy từ map (nếu map có)
    const tStr = String(param.time); // yyyy-mm-dd
    const bar = map[tStr.slice(0, 10)];
    const vol = bar?.volume ?? 0;

    const color = (close >= open) ? "#22c55e" : "#ef4444";
    const dir = (close >= open) ? "up" : "down";

    // Format tooltip
    tooltip.innerHTML = `
      <div class="tooltip-header">
         <span class="tooltip-date">${tStr}</span>
         <span class="tooltip-status tooltip-${dir}">${dir === 'up' ? '▲' : '▼'}</span>
      </div>
      <div class="tooltip-grid">
        <div class="tooltip-item"><span class="tooltip-label">Open</span> <span class="tooltip-val">${fmt(open)}</span></div>
        <div class="tooltip-item"><span class="tooltip-label">High</span> <span class="tooltip-val">${fmt(high)}</span></div>
        <div class="tooltip-item"><span class="tooltip-label">Low</span>  <span class="tooltip-val">${fmt(low)}</span></div>
        <div class="tooltip-item"><span class="tooltip-label">Close</span> <span class="tooltip-val tooltip-${dir}">${fmt(close)}</span></div>
        <div class="tooltip-item tooltip-full"><span class="tooltip-label">Volume</span> <span class="tooltip-val">${fmt(vol)}</span></div>
      </div>
    `;

    // Positioning
    const w = tooltip.offsetWidth;
    const h = tooltip.offsetHeight;
    const containerRect = container.getBoundingClientRect();

    let left = param.point.x + 15;
    let top = param.point.y + 15;

    if (left + w > containerRect.width) {
      left = param.point.x - w - 15;
    }
    if (top + h > containerRect.height) {
      top = param.point.y - h - 15;
    }

    // Ensure not overflowing top/left
    left = Math.max(5, left);
    top = Math.max(5, top);

    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
    tooltip.style.display = "block";
  });

  chart.timeScale().fitContent();

  const resizeObs = new ResizeObserver((entries) => {
    const nw = Math.max(entries[0]?.contentRect?.width ?? 400, 400);
    if (nw > 0 && chartInstance) chartInstance.applyOptions({ width: nw });
  });
  resizeObs.observe(container);

  chartInstance = chart;
  chartInstance._tooltipUnsub = unsub;
  chartInstance._resizeObs = resizeObs;
}

function updateSymbolListSelection() {
  const list = document.getElementById("symbol-list");
  if (list) list.querySelectorAll(".symbol-btn").forEach(b => b.classList.toggle("selected", b.dataset.symbol === currentSymbol));
}

function renderSymbolList(symbols) {
  const list = document.getElementById("symbol-list");
  if (!list) return;
  list.innerHTML = symbols.map(s => `<button type="button" class="symbol-btn" data-symbol="${s}">${s}</button>`).join("");
  list.querySelectorAll(".symbol-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (symbolSelect) symbolSelect.value = btn.dataset.symbol;
      loadAnalysis(btn.dataset.symbol);
    });
  });
}

async function loadSymbols() {
  try {
    const res = await fetch("/api/symbols");
    if (res.ok) {
      const { symbols } = await res.json();
      if (Array.isArray(symbols) && symbols.length) {
        if (symbolSelect) symbolSelect.innerHTML = symbols.map(s => `<option value="${s}"${s === "FPT" ? " selected" : ""}>${s}</option>`).join("");
        renderSymbolList(symbols);
        return;
      }
    }
  } catch (_) { }
  if (symbolSelect) symbolSelect.innerHTML = VN30_FALLBACK.map(s => `<option value="${s}"${s === "FPT" ? " selected" : ""}>${s}</option>`).join("");
  renderSymbolList(VN30_FALLBACK);
}

function setupHorizonInputs() {
  const dateIn = document.getElementById("target-date");
  const monthIn = document.getElementById("target-month");
  const today = new Date();
  if (dateIn) {
    dateIn.min = new Date(today.getFullYear() - 1, 0, 1).toISOString().slice(0, 10);
    dateIn.max = today.toISOString().slice(0, 10);
    if (!dateIn.value) dateIn.value = today.toISOString().slice(0, 10);
    const onDateChange = (e) => {
      e.stopPropagation();
      const r = document.querySelector('input[name="horizon"][value="date"]');
      if (r) { r.checked = true; if (currentSymbol) loadAnalysis(currentSymbol); }
    };
    dateIn.addEventListener("change", onDateChange);
    dateIn.addEventListener("input", onDateChange);
    dateIn.addEventListener("click", (e) => e.stopPropagation());
  }
  if (monthIn) {
    monthIn.max = today.toISOString().slice(0, 7);
    if (!monthIn.value) monthIn.value = today.toISOString().slice(0, 7);
    monthIn.addEventListener("change", (e) => {
      e.stopPropagation();
      const r = document.querySelector('input[name="horizon"][value="month"]');
      if (r) { r.checked = true; if (currentSymbol) loadAnalysis(currentSymbol); }
    });
    monthIn.addEventListener("click", (e) => e.stopPropagation());
  }
  document.querySelectorAll('input[name="horizon"]').forEach(el => el.addEventListener("change", () => {
    if (currentSymbol) loadAnalysis(currentSymbol);
  }));
}

function updateChartEndHint(actualEndStr, requestedEndStr) {
  const hint = document.getElementById("chart-end-hint");
  if (!hint) return;
  if (!actualEndStr) { hint.textContent = ""; return; }
  const d = new Date(actualEndStr);
  let text = "Hiển thị đến: " + d.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" });
  if (requestedEndStr && actualEndStr < requestedEndStr) {
    text += " (dữ liệu mới nhất có sẵn — chạy pipeline giá để cập nhật)";
  }
  hint.textContent = text;
}

function setupCapitalToggle() {
  const sel = document.getElementById("capital-select");
  const customIn = document.getElementById("capital-custom");
  if (!sel || !customIn) return;
  const toggle = () => {
    const isCustom = sel.value === "custom";
    customIn.hidden = !isCustom;
    if (isCustom) customIn.focus();
  };
  sel.addEventListener("change", () => { toggle(); if (currentSymbol) loadAnalysis(currentSymbol); });
  customIn.addEventListener("input", () => { if (currentSymbol) loadAnalysis(currentSymbol); });
  customIn.addEventListener("change", () => { if (currentSymbol) loadAnalysis(currentSymbol); });
  toggle();
}

symbolSelect?.addEventListener("change", (e) => loadAnalysis(e.target.value));
form?.addEventListener("change", () => { if (currentSymbol) loadAnalysis(currentSymbol); });
setupCapitalToggle();
loadSymbols().then(() => {
  const sym = symbolSelect?.value || "FPT";
  if (sym) loadAnalysis(sym);
});
setupHorizonInputs();

const now = new Date();
const el = document.getElementById("today-date");
if (el) el.textContent = "Hôm nay: " + now.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" });
