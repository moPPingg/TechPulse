import matplotlib.pyplot as plt
import seaborn as sns

# ── Shared annotation style helpers ──────────────────────────────────────────
def ann_bbox(color='#2c3e50', alpha=0.92):
    """Return a white-background bbox dict for annotate/text calls."""
    return dict(boxstyle='round,pad=0.35', facecolor='white',
                edgecolor=color, alpha=alpha, linewidth=0.8)


def set_academic_style():
    """Sets publication-quality matplotlib rcParams suitable for Q1 journals."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',        # Times-like math font
        'font.size': 14,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 17,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,        # extra outer padding
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': plt.cycler('color', sns.color_palette('colorblind')),
        # Use explicit tight_layout() in each script; autolayout causes
        # conflicts when combined with manual tight_layout() calls.
        'figure.autolayout': False,
        'axes.titlepad': 20,
    })
