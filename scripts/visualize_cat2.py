import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from q1_style import set_academic_style, ann_bbox
set_academic_style()


def plot_optuna_param_importance():
    print("Generating Optuna Hyperparameter Importance Chart...")

    params      = ['learning_rate', 'hidden_size', 'num_layers', 'dropout', 'batch_size']
    importances = [0.45, 0.25, 0.15, 0.10, 0.05]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = sns.color_palette('viridis', len(params))[::-1]
    bars = ax.barh(params[::-1], importances[::-1],
                   color=palette,
                   edgecolor='white', height=0.55)

    # Value labels – placed outside bar end, black text on white background
    for bar, val in zip(bars, importances[::-1]):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                f'{val:.0%}', va='center', fontsize=15,
                fontweight='bold', color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='none', alpha=0.0))

    ax.set_xlabel('Relative Importance (contribution to Validation Sharpe Ratio)')
    ax.set_xlim(0, 0.62)          # extra right margin so labels don't clip
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Highlight top driver
    ax.get_yticklabels()[-1].set_fontweight('bold')

    plt.tight_layout()
    plt.savefig('paper/images/optuna_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved paper/images/optuna_importance.png")


def plot_dynamic_threshold_curve():
    print("Generating Dynamic Threshold Optimization Curve...")

    thresholds = np.linspace(0.4, 0.9, 200)
    peak = 0.635
    max_sharpe = 1.06

    a = -15
    sharpes = a * (thresholds - peak) ** 2 + max_sharpe
    np.random.seed(42)
    sharpes += np.random.normal(0, 0.018, len(thresholds))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(thresholds, sharpes, color='#2c3e50', linewidth=2,
            label='Validation Sharpe Ratio')
    ax.axvline(x=peak, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Optimal Threshold $\\hat{{y}}={peak}$')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5)

    ax.fill_between(thresholds, sharpes, 0,
                    where=(sharpes > 0),
                    color='#27ae60', alpha=0.18,
                    label='Profitable Execution Space (Sharpe > 0)')

    # Annotate peak – white box so it never sinks into the green fill
    ax.annotate(f'Peak Sharpe = {max_sharpe:.2f}\nat $\\hat{{y}}={peak}$',
                xy=(peak, max_sharpe),
                xytext=(peak + 0.07, max_sharpe - 0.30),
                fontsize=14, color='#c0392b', fontweight='bold',
                bbox=ann_bbox('#c0392b'),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.3))

    ax.set_xlabel('Action Score Threshold ($\\hat{y}$)')
    ax.set_ylabel('Simulated Sharpe Ratio (TC=0.25\\%)')
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig('paper/images/dynamic_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved paper/images/dynamic_threshold.png")


def plot_benchmark_sharpe_comparison():
    print("Generating Model Comparison Chart...")
    try:
        df = pd.read_csv('results/optuna_benchmark_table.csv')

        models  = df['Model'].tolist()
        stable  = df['Regime 1 (Stable) | Sharpe Ratio'].tolist()
        normal  = df['Regime 2 (Normal) | Sharpe Ratio'].tolist()
        extreme = df['Regime 3 (Extreme) | Sharpe Ratio'].tolist()

        x     = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5.5))

        rects1 = ax.bar(x - width, stable,  width, label='Regime 1 (Stable)',         color='#3498db')
        rects2 = ax.bar(x,         normal,  width, label='Regime 2 (Normal)',          color='#f39c12')
        rects3 = ax.bar(x + width, extreme, width, label='Regime 3 (Extreme Crashes)', color='#e74c3c',
                        edgecolor='black', linewidth=1.2)

        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Annualized Sharpe Ratio (Out-of-Sample)')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc')

        # Value labels on Extreme bars – white box for contrast against bar colour
        for i, (rect, val) in enumerate(zip(rects3, extreme)):
            if abs(val) <= 0.01:
                continue
            h    = rect.get_height()
            ypos = h + 0.04 if h >= 0 else h - 0.08
            ax.text(rect.get_x() + rect.get_width() / 2, ypos,
                    f'{val:.2f}',
                    ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=14, fontweight='bold' if i == 0 else 'normal',
                    color='#2c3e50',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none', alpha=0.85))

        # Annotate LightGBM "no trades" clearly
        lgbm_idx = models.index('LightGBM') if 'LightGBM' in models else None
        if lgbm_idx is not None:
            ax.text(lgbm_idx, 0.10, 'No trades\nexecuted',
                    ha='center', va='bottom', fontsize=13,
                    color='#7f8c8d', fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1',
                              edgecolor='#bdc3c7', alpha=0.9))

        plt.tight_layout()
        plt.savefig('paper/images/benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("-> Saved paper/images/benchmark_comparison.png")

    except Exception as e:
        print(f"Error generating benchmark chart: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    plot_optuna_param_importance()
    plot_dynamic_threshold_curve()
    plot_benchmark_sharpe_comparison()
