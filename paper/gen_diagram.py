"""
Regenerates paper/images/system_architecture.png.
Two-track pipeline:
  LEFT  — Production Alert Pipeline (yfinance → News → FinBERT → Regime Classifier)
  CENTER — Core ML Pipeline         (OHLCV → SMC → R^7 → LSTM → Action Score)
Both tracks converge at the Gmail Alert output.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 12)
ax.set_ylim(2.6, 15.8)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Drawing helpers ───────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, label,
         fc='white', ec='#2c3e50', tc='white',
         fs=9.5, fw='bold', lw=1.5, ls='solid', pad=0.18, zorder=4):
    bp = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle=f'round,pad={pad}',
                        fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(bp)
    ax.text(cx, cy, label, color=tc, fontsize=fs, fontweight=fw,
            ha='center', va='center', zorder=zorder + 1,
            multialignment='center')


def diamond(cx, cy, hw, hh, label,
            fc='#FCF3CF', ec='#D4AC0D', tc='black', fs=10, lw=2):
    pts = [(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)]
    ax.add_patch(mpatches.Polygon(pts, closed=True,
                                  fc=fc, ec=ec, lw=lw, zorder=4))
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=tc, zorder=5,
            multialignment='center')


def solid_arrow(x1, y1, x2, y2, color='black', lw=1.5, ms=13):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=ms,
                                connectionstyle='arc3,rad=0'))


def dashed_arrow(x1, y1, x2, y2, color='#888888', lw=1.2, ms=11):
    ap = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle='->', color=color,
                         lw=lw, linestyle='dashed', mutation_scale=ms,
                         zorder=3)
    ax.add_patch(ap)


def lbl(x, y, text, ha='center', va='center', fs=8.0, color='black'):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            zorder=7, multialignment='center',
            bbox=dict(boxstyle='round,pad=0.12', fc='white',
                      ec='none', alpha=0.90))


# ── Layout constants ──────────────────────────────────────────────────────────
XL   = 1.9    # left track center x (Production Alert Pipeline)
XC   = 6.6    # center track x      (Core ML Pipeline)
XGUI = 10.6   # Visual Frontend GUI x

Y_FETCHER = 15.0  # yfinance Fetcher
Y_OHLCV   = 13.7  # Atomic OHLCV Data Matrix
Y_SMC_T   = 12.75 # SMC box top
Y_SMC_M   = 11.95 # SMC feature boxes center
Y_SMC_B   = 11.15 # SMC box bottom
Y_R7      = 10.1  # R^7 Tensor
Y_LSTM    = 8.6   # LSTM Predictive Core
Y_DIAM    = 6.6   # Decision diamond center
Y_BOT     = 4.7   # Bottom outputs

# Left track Y positions (aligned with corresponding center nodes)
Y_NEWS    = 11.95  # same as SMC_M
Y_FINBERT = 10.1   # same as R^7
Y_REGIME  = 8.6    # same as LSTM

# ── Background track labels ───────────────────────────────────────────────────
ax.text(XL, 15.65, 'Production Alert Pipeline',
        ha='center', va='center', fontsize=8.5, color='#555555',
        style='italic', fontweight='bold')
ax.text(XC, 15.65, 'Core ML Pipeline',
        ha='center', va='center', fontsize=8.5, color='#555555',
        style='italic', fontweight='bold')

# Vertical divider between left and center track
ax.plot([3.55, 3.55], [4.0, 15.5],
        color='#dddddd', lw=1.0, linestyle='dotted', zorder=1)

# ── LEFT TRACK ────────────────────────────────────────────────────────────────

# 1. yfinance Data Fetcher
rbox(XL, Y_FETCHER, 2.9, 0.72,
     'yfinance\nData Fetcher',
     fc='#AED6F1', ec='#2471A3', tc='black', fs=9.0)

# 2. News Sources
rbox(XL, Y_NEWS, 2.9, 0.90,
     'News Sources\nVnExpress · CafeF\nTinNhanhCK',
     fc='#FAD7A0', ec='#E67E22', tc='black', fs=8.2, fw='normal')

# 3. FinBERT Sentiment
rbox(XL, Y_FINBERT, 2.9, 0.72,
     'FinBERT\nSentiment Score',
     fc='#D5F5E3', ec='#1E8449', tc='black', fs=9.0)

# 4. Regime Classifier
rbox(XL, Y_REGIME, 2.9, 0.82,
     'Dual-Trigger\nRegime Classifier\n(Vol + Sentiment)',
     fc='#FEF9E7', ec='#F39C12', tc='black', fs=8.2, fw='normal')

# ── CENTER TRACK ──────────────────────────────────────────────────────────────

# 5. Atomic OHLCV Data Matrix
rbox(XC, Y_OHLCV, 4.3, 0.72,
     'Atomic OHLCV Data Matrix',
     fc='#5DADE2', ec='#2471A3')

# 6. SMC Quantification Layer (dashed outer box)
smc_xl, smc_w = 4.05, 7.15
smc_yb, smc_h = Y_SMC_B - 0.08, Y_SMC_T - Y_SMC_B + 0.16
ax.add_patch(FancyBboxPatch((smc_xl, smc_yb), smc_w, smc_h,
                            boxstyle='round,pad=0.10',
                            fc='#F9F9F9', ec='#999999',
                            lw=1.3, linestyle='dashed', zorder=2))
ax.text(smc_xl + smc_w - 0.08, Y_SMC_T + 0.02,
        'SMC Quantification Layer',
        ha='right', va='top', fontsize=8.0,
        color='black', style='italic', zorder=3)

# Three feature boxes inside SMC
for label, cx in [('Liquidity\nSweeps', 5.15),
                  ('Order\nBlocks',     6.90),
                  ('Fair Value\nGaps',  8.65)]:
    rbox(cx, Y_SMC_M, 1.55, 0.72, label,
         fc='white', ec='#666666', tc='black',
         fs=8.5, fw='normal', lw=1.2)

# 7. R^7 Feature Tensor
rbox(XC, Y_R7, 3.0, 0.72,
     r'$\mathbb{R}^7$ Feature Tensor',
     fc='#A569BD', ec='#7D3C98', pad=0.28)

# 8. LSTM Predictive Core
rbox(XC, Y_LSTM, 3.8, 0.72,
     'LSTM Predictive Core',
     fc='#5DADE2', ec='#2471A3')

# 9. Decision diamond
diamond(XC, Y_DIAM, 2.4, 1.45,
        'Action Score\n> 0.635')

# 10. Gmail Alert (green, left of diamond)
rbox(4.1, Y_BOT, 3.0, 0.88,
     'Gmail Alert\n(Liquidity Sweep / Regime 3)',
     fc='#D5F5E3', ec='#27AE60', tc='black', lw=1.8, fs=8.8)

# 11. Hold / Await Setup (gray dashed, right of diamond)
rbox(8.8, Y_BOT, 2.6, 0.72,
     'Hold / Await Setup',
     fc='white', ec='#AAAAAA', tc='black',
     fw='normal', lw=1.3, ls='dashed')

# 12. Visual Frontend GUI (right side)
rbox(XGUI, Y_R7, 2.4, 0.72,
     'Visual\nFrontend GUI',
     fc='white', ec='#2471A3', tc='black',
     fw='normal', lw=1.5)

# ── ARROWS: Left track internal ───────────────────────────────────────────────

# yfinance → OHLCV (diagonal right)
solid_arrow(XL + 1.45, Y_FETCHER,
            XC - 2.15, Y_OHLCV + 0.36)
lbl((XL + 1.45 + XC - 2.15) / 2 + 0.15,
    (Y_FETCHER + Y_OHLCV) / 2 + 0.15,
    'Daily Fetch\n(16:00 ICT)', fs=7.5)

# News → FinBERT
solid_arrow(XL, Y_NEWS - 0.45,
            XL, Y_FINBERT + 0.36)

# FinBERT → Regime
solid_arrow(XL, Y_FINBERT - 0.36,
            XL, Y_REGIME + 0.41)
lbl(XL + 0.7, (Y_FINBERT + Y_REGIME) / 2,
    'score ∈ [-1,1]', fs=7.5)

# Regime → Gmail Alert (diagonal to bottom-left)
solid_arrow(XL + 0.5, Y_REGIME - 0.41,
            4.1 - 1.2, Y_BOT + 0.44)
lbl(XL + 1.0, (Y_REGIME + Y_BOT) / 2,
    'Regime 3\nTrigger', fs=7.5, color='#E67E22')

# ── ARROWS: Center track internal ────────────────────────────────────────────

# OHLCV left side → R^7  ("5 OHLCV Features")
ax.plot([4.55, 4.55], [Y_OHLCV - 0.36, Y_R7],
        color='black', lw=1.5, zorder=3, solid_capstyle='round')
solid_arrow(4.55, Y_R7, 5.1, Y_R7, ms=11)
lbl(4.0, (Y_OHLCV + Y_R7) / 2,
    '5 OHLCV\nFeatures', ha='center', fs=7.8)

# OHLCV right side → SMC box  ("Price Action")
solid_arrow(XC + 0.8, Y_OHLCV - 0.36,
            7.5,       Y_SMC_T + 0.08)
lbl(7.8, (Y_OHLCV - 0.3 + Y_SMC_T) / 2 + 0.1,
    'Price Action', ha='left', fs=7.8)

# SMC bottom → R^7  ("2 SMC Features")
solid_arrow(5.5, Y_SMC_B - 0.08,
            XC - 0.6, Y_R7 + 0.36)
lbl(5.6, (Y_SMC_B + Y_R7) / 2,
    '2 SMC\nFeatures', ha='right', fs=7.8)

# R^7 → LSTM
solid_arrow(XC, Y_R7 - 0.36, XC, Y_LSTM + 0.36)

# LSTM → Diamond
solid_arrow(XC, Y_LSTM - 0.36, XC, Y_DIAM + 1.45)

# Diamond left → Gmail Alert  ("Yes")
solid_arrow(XC - 2.4, Y_DIAM,
            4.1 + 0.6, Y_BOT + 0.44)
lbl(XC - 2.6, Y_DIAM - 0.55, 'Yes', fs=9.5)

# Diamond right → Hold/Await  ("No")
solid_arrow(XC + 2.4, Y_DIAM,
            8.8, Y_BOT + 0.36)
lbl(XC + 2.5, Y_DIAM - 0.55, 'No', fs=9.5)

# ── ARROWS: Center → Right (GUI) ─────────────────────────────────────────────

# Order Blocks → Visual Frontend GUI
dashed_arrow(6.90, Y_SMC_M - 0.36,
             XGUI - 0.65, Y_R7 + 0.36)
lbl(9.3, (Y_SMC_M + Y_R7) / 2 + 0.1,
    'Heuristic\nMapping', ha='left', fs=7.5)

# Fair Value Gaps → Visual Frontend GUI
dashed_arrow(8.65, Y_SMC_M - 0.36,
             XGUI + 0.45, Y_R7 + 0.36)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig('paper/images/system_architecture.png',
            dpi=300, bbox_inches='tight', pad_inches=0.06)
print("Saved paper/images/system_architecture.png")
