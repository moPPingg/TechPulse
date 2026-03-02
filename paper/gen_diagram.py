import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

# Ensure matplotlib is working in headless mode
import matplotlib
matplotlib.use('Agg')

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

def draw_box(ax, x, y, width, height, text, bg_color, text_color='white'):
    box = patches.FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.2",
                                 ec="black", fc=bg_color, lw=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, color=text_color, fontsize=12, weight='bold', ha='center', va='center')

# Draw Input 1
draw_box(ax, 1, 4, 2.5, 1, "OHLCV Data", "#2980B9")

# Draw Input 2
draw_box(ax, 4.5, 4, 2.5, 1, "News Sentiment\n(Multimodal)", "#2980B9")

# Draw LSTM
draw_box(ax, 2.75, 2, 2.5, 1, "LSTM\nPredictive Core", "#27AE60")

# Draw Predictions
draw_box(ax, 2.75, 0, 2.5, 1, "Trading\nPredictions", "#8E44AD")

arrow_props = dict(facecolor='black', edgecolor='black', shrink=0.05, width=2, headwidth=8)

# Input 1 -> LSTM
ax.annotate('', xy=(3.5, 3.2), xytext=(2.25, 4), arrowprops=arrow_props)

# Input 2 -> LSTM
ax.annotate('', xy=(4.5, 3.2), xytext=(5.75, 4), arrowprops=arrow_props)

# LSTM -> Predictions
ax.annotate('', xy=(4.0, 1.2), xytext=(4.0, 2.0), arrowprops=arrow_props)

plt.xlim(0, 8)
plt.ylim(-0.5, 5.5)

plt.savefig(r'd:\TechPulse\paper\system_architecture.png', bbox_inches='tight', dpi=300)
print("System architecture diagram generated successfully.")
