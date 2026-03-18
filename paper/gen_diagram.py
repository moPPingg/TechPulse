"""
Regenerates paper/images/system_architecture.png.
Matches the original flow-chart design exactly with tight margins
(no excess white space).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 11.5))
ax.set_xlim(0, 10)
ax.set_ylim(2.6, 13.8)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Drawing helpers ───────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, label,
         fc='white', ec='#2c3e50', tc='white',
         fs=10, fw='bold', lw=1.5, ls='solid', pad=0.18, zorder=4):
    """Rounded rectangle centred at (cx, cy)."""
    bp = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle=f'round,pad={pad}',
                        fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(bp)
    ax.text(cx, cy, label, color=tc, fontsize=fs, fontweight=fw,
            ha='center', va='center', zorder=zorder + 1,
            multialignment='center')


def diamond(cx, cy, hw, hh, label,
            fc='#FCF3CF', ec='#D4AC0D', tc='black', fs=10.5, lw=2):
    """Diamond (rotated square) centred at (cx, cy)."""
    pts = [(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)]
    ax.add_patch(mpatches.Polygon(pts, closed=True,
                                  fc=fc, ec=ec, lw=lw, zorder=4))
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=tc, zorder=5,
            multialignment='center')


def solid_arrow(x1, y1, x2, y2, color='black', lw=1.5, ms=14):
    """Solid arrow from tail (x1,y1) to tip (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=ms,
                                connectionstyle='arc3,rad=0'))


def dashed_arrow(x1, y1, x2, y2, color='#999999', lw=1.2, ms=12):
    """Dashed arrow using FancyArrowPatch for reliable dash rendering."""
    ap = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle='->', color=color,
                         lw=lw, linestyle='dashed', mutation_scale=ms,
                         zorder=3)
    ax.add_patch(ap)


def lbl(x, y, text, ha='center', va='center', fs=8.5, color='black'):
    """Floating label with white background."""
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            zorder=7, multialignment='center',
            bbox=dict(boxstyle='round,pad=0.12', fc='white',
                      ec='none', alpha=0.88))


# ── Layout constants ──────────────────────────────────────────────────────────
XC       = 3.8    # main-flow center x
XGUI     = 8.35   # Visual Frontend GUI center x

Y_OHLCV  = 13.1
Y_SMC_T  = 12.05  # top edge of SMC dashed box
Y_SMC_M  = 11.2   # centre of feature boxes inside SMC
Y_SMC_B  = 10.35  # bottom edge of SMC dashed box
Y_R7     = 9.25
Y_LSTM   = 7.75
Y_DIAM   = 5.75   # diamond centre
Y_BOT    = 3.85   # bottom terminal boxes

# ── 1. Atomic OHLCV Data Matrix ───────────────────────────────────────────────
rbox(XC, Y_OHLCV, 4.3, 0.72,
     'Atomic OHLCV Data Matrix',
     fc='#5DADE2', ec='#2471A3')

# ── 2. SMC Quantification Layer (dashed outer box) ───────────────────────────
smc_x, smc_w = 2.25, 7.0
smc_y, smc_h = Y_SMC_B - 0.08, Y_SMC_T - Y_SMC_B + 0.16
ax.add_patch(FancyBboxPatch((smc_x, smc_y), smc_w, smc_h,
                            boxstyle='round,pad=0.10',
                            fc='#F9F9F9', ec='#999999',
                            lw=1.3, linestyle='dashed', zorder=2))
ax.text(smc_x + smc_w - 0.08, Y_SMC_T + 0.02,
        'SMC Quantification Layer',
        ha='right', va='top', fontsize=8.2,
        color='black', style='italic', zorder=3)

# Three feature boxes inside SMC
for label, cx in [('Liquidity\nSweeps', 3.35),
                  ('Order\nBlocks',     5.45),
                  ('Fair Value\nGaps',  7.55)]:
    rbox(cx, Y_SMC_M, 1.68, 0.72, label,
         fc='white', ec='#666666', tc='black',
         fs=9, fw='normal', lw=1.2)

# ── 3. R^7 Feature Tensor (purple pill) ──────────────────────────────────────
rbox(XC, Y_R7, 2.9, 0.70,
     r'$\mathbb{R}^7$ Feature Tensor',
     fc='#A569BD', ec='#7D3C98', pad=0.28)

# ── 4. LSTM Predictive Core ───────────────────────────────────────────────────
rbox(XC, Y_LSTM, 3.7, 0.72,
     'LSTM Predictive Core',
     fc='#5DADE2', ec='#2471A3')

# ── 5. Action Score decision diamond ─────────────────────────────────────────
diamond(XC, Y_DIAM, 2.35, 1.42,
        'Action Score > 0.635')

# ── 6. Automated Alert Mechanism (green) ─────────────────────────────────────
rbox(1.55, Y_BOT, 2.7, 0.88,
     'Automated Alert\nMechanism',
     fc='#D5F5E3', ec='#27AE60', tc='black', lw=1.8)

# ── 7. Hold / Await Setup (gray dashed) ──────────────────────────────────────
rbox(5.6, Y_BOT, 2.6, 0.72,
     'Hold / Await Setup',
     fc='white', ec='#AAAAAA', tc='black',
     fw='normal', lw=1.3, ls='dashed')

# ── 8. Visual Frontend GUI (right side) ──────────────────────────────────────
rbox(XGUI, Y_R7, 2.5, 0.72,
     'Visual\nFrontend GUI',
     fc='white', ec='#2471A3', tc='black',
     fw='normal', lw=1.5)

# ── Arrows ────────────────────────────────────────────────────────────────────

# OHLCV → SMC box  ("Price Action")
solid_arrow(XC + 0.5, Y_OHLCV - 0.36,
            5.8,       Y_SMC_T + 0.10)
lbl(5.55, (Y_OHLCV - 0.36 + Y_SMC_T + 0.1) / 2 + 0.15,
    'Price Action', ha='left', fs=8.2)

# OHLCV left edge → R^7  ("5 OHLCV Features")
#   Draw as a vertical line on the left, then a short horizontal arrowhead.
ax.plot([1.75, 1.75], [Y_OHLCV - 0.36, Y_R7],
        color='black', lw=1.5, zorder=3, solid_capstyle='round')
solid_arrow(1.75, Y_R7, 2.45, Y_R7, ms=12)   # horizontal tip into R^7
lbl(1.05, (Y_OHLCV + Y_R7) / 2,
    '5 OHLCV\nFeatures', ha='center', fs=8.2)

# SMC bottom → R^7  ("2 SMC Features")
solid_arrow(3.5, Y_SMC_B - 0.08,
            XC - 0.3, Y_R7 + 0.35)
lbl(3.2, (Y_SMC_B + Y_R7 + 0.35) / 2,
    '2 SMC\nFeatures', ha='right', fs=8.2)

# R^7 → LSTM
solid_arrow(XC, Y_R7 - 0.35, XC, Y_LSTM + 0.36)

# LSTM → Diamond (top vertex)
solid_arrow(XC, Y_LSTM - 0.36, XC, Y_DIAM + 1.42)

# Diamond left → Automated Alert  ("Yes")
solid_arrow(XC - 2.35, Y_DIAM,
            1.55, Y_BOT + 0.44)
lbl(XC - 2.35 + 0.12, Y_DIAM - 0.42, 'Yes', fs=9.5)

# Diamond right → Hold/Await  ("No")
solid_arrow(XC + 2.35, Y_DIAM,
            5.6, Y_BOT + 0.36)
lbl(XC + 2.35 - 0.12, Y_DIAM - 0.42, 'No', fs=9.5)

# Order Blocks → Visual Frontend GUI  (dashed, "Heuristic Mapping")
dashed_arrow(5.45, Y_SMC_M - 0.36,
             XGUI - 0.55, Y_R7 + 0.36)
lbl(7.15, (Y_SMC_M - 0.36 + Y_R7 + 0.36) / 2 + 0.22,
    'Heuristic\nMapping', ha='left', fs=7.8)

# Fair Value Gaps → Visual Frontend GUI  (dashed, "Heuristic Mapping")
dashed_arrow(7.55, Y_SMC_M - 0.36,
             XGUI + 0.45, Y_R7 + 0.36)
lbl(8.4, (Y_SMC_M - 0.36 + Y_R7 + 0.36) / 2 - 0.22,
    'Heuristic\nMapping', ha='left', fs=7.8)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig('paper/images/system_architecture.png',
            dpi=300, bbox_inches='tight', pad_inches=0.06)
print("Saved paper/images/system_architecture.png")
